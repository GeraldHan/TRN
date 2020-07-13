import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from modulars import FCNet, Encoder, Decoder
import modulars
from torch.nn.utils import weight_norm
import copy

class Net(nn.Module):
    def __init__(self, **kwargs):
        """
        kwargs:
             vocab,
             dim_hidden, 
             dim_vision,
             mid_size, 
             glimpses,
             dropout_prob,
             device,
        """
        super(Net, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v) 

        glimpses = 6
        objects = 10

        self.num_classes = len(self.vocab['answer_token_to_idx'])
        

        self.num_token = len(self.vocab['question_token_to_idx'])

        self.classifier = Classifier(
            in_features= self.dim_hidden,
            mid_features=self.mid_size * 2,
            out_features=self.num_classes,
            drop=self.dropout_prob
            )

        self.token_embedding = nn.Embedding(self.num_token, self.dim_word)

        for p in self.token_embedding.parameters():
            p.require_grad = False

        self.dropout = nn.Dropout(self.dropout_prob)

        self.text = modulars.BiGRUEncoder_2(
            dim_word=self.dim_word,
            dim_hidden=self.dim_hidden,
        )

        # self.count = Counter(objects)

        self.attention = weight_norm(BiAttention(
            v_features=self.dim_vision,
            q_features=self.dim_hidden,
            mid_features=self.mid_size,
            glimpses=glimpses,
            drop=0.5,), name='h_weight', dim=None)

        self.timereasoning = TimeReasoning(hidden_size=self.dim_vision,
                                           mid_size=self.mid_size,
                                           state_size=self.state_size,
                                           num_token=self.num_token,
                                           edge_size=self.dim_edge)

        self.apply_attention = ApplyAttention(
            v_features=self.dim_vision,
            q_features=self.dim_hidden,
            mid_features=self.mid_size,
            glimpses=glimpses,
            num_obj=objects,
            drop=0.2,
        )
                                         
    def forward(self, q, v, e_s, e_e, e_mask, e_label, edge, r_mask, q_mask, q_len):
        q = q.permute(1, 0)  #[seq_len, batch]
        w_emb = self.token_embedding(q) #[seq_len]
        hidden_q, q_out, hidden_entity, hidden_trans = self.text(w_emb, list(q_len.data), e_s, e_e)
        q_max, batch_size= q.shape
        _, num_obj, _ = v.shape

        v_mask = v.sum(-1).masked_fill(v.sum(-1) != 0., 1)
        
        q_input = hidden_entity.transpose(0, 1)
        # q_input = hidden_q.transpose(0, 1)

        q_mask = torch.ones_like(e_mask)

        atten, logits = self.attention(v, q_input, v_mask, q_mask) # batch x glimpses x v_num x q_num

        loss_time = self.timereasoning(hidden_trans, v, e_label, e_s, e_e, e_mask, edge, r_mask, logits, v_mask)

        new_q = self.apply_attention(v, q_input, v_mask, q_mask, atten, logits)
        answer = self.classifier(new_q)

        return answer, loss_time

class Classifier(nn.Module):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.lin1 = FCNet(in_features, mid_features, activate='relu')
        self.lin2 = FCNet(mid_features, out_features, drop=drop)

    def forward(self, q):
        x = self.lin1(q)
        x = self.lin2(x)
        return x

class BiAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(BiAttention, self).__init__()
        self.hidden_aug = 3
        self.glimpses = glimpses
        self.lin_v = FCNet(v_features, int(mid_features * self.hidden_aug), activate='relu', drop=drop/2.5)  # let self.lin take care of bias
        self.lin_q = FCNet(q_features, int(mid_features * self.hidden_aug), activate='relu', drop=drop/2.5)
        
        self.h_weight = nn.Parameter(torch.Tensor(1, glimpses, 1, int(mid_features * self.hidden_aug)).normal_())
        self.h_bias = nn.Parameter(torch.Tensor(1, glimpses, 1, 1).normal_())

        self.drop = nn.Dropout(drop)

    def forward(self, v, q, v_mask, q_mask):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        """
        v_num = v.size(1)
        q_num = q.size(1)

        v_ = self.lin_v(v).unsqueeze(1)  # batch, 1, v_num, dim
        q_ = self.lin_q(q).unsqueeze(1)  # batch, 1, q_num, dim
        v_ = self.drop(v_)

        h_ = v_ * self.h_weight # broadcast:  batch x glimpses x v_num x dim
        logits = torch.matmul(h_, q_.transpose(2,3)) # batch x glimpses x v_num x q_num
        logits = logits + self.h_bias

        logits_out = logits

        # apply v_mask, q_mask
        logits = logits.masked_fill(v_mask.unsqueeze(1).unsqueeze(3).expand(logits.shape) == 0, -float('inf'))
        logits = logits.masked_fill(q_mask.unsqueeze(1).unsqueeze(2).expand(logits.shape) == 0, -float('inf'))

        atten = F.softmax(logits.view(-1, self.glimpses, v_num * q_num), 2)
        return atten.view(-1, self.glimpses, v_num, q_num), logits_out


class ApplyAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, num_obj, drop=0.0):
        super(ApplyAttention, self).__init__()
        self.glimpses = glimpses
        layers = []
        for g in range(self.glimpses):
            layers.append(ApplySingleAttention(v_features, q_features, mid_features, num_obj, drop))
        self.glimpse_layers = nn.ModuleList(layers)
    
    def forward(self, v, q, v_mask, q_mask, atten, logits):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        atten:  batch x glimpses x v_num x q_num
        logits:  batch x glimpses x v_num x q_num
        """
        for g in range(self.glimpses):
            atten_h = self.glimpse_layers[g](v, q, v_mask, q_mask, atten[:,g,:,:], logits[:,g,:,:])
            # residual (in original paper)
            q = q + atten_h 
        #q = q * q_mask.unsqueeze(2)
        return q.sum(1)

class ApplySingleAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, num_obj, drop=0.0):
        super(ApplySingleAttention, self).__init__()
        self.lin_v = FCNet(v_features, mid_features, activate='relu', drop=drop)  # let self.lin take care of bias
        self.lin_q = FCNet(q_features, mid_features, activate='relu', drop=drop)
        self.lin_atten = FCNet(mid_features, mid_features, drop=drop)

    def forward(self, v, q, v_mask, q_mask, atten, logits):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        atten:  batch x v_num x q_num
        logits:  batch x v_num x q_num
        """

        # apply single glimpse attention
        v_ = self.lin_v(v).transpose(1,2).unsqueeze(2) # batch, dim, 1, num_obj
        q_ = self.lin_q(q).transpose(1,2).unsqueeze(3) # batch, dim, que_len, 1
        v_ = torch.matmul(v_, atten.unsqueeze(1)) # batch, dim, 1, que_len
        h_ = torch.matmul(v_, q_) # batch, dim, 1, 1
        h_ = h_.squeeze(3).squeeze(2) # batch, dim
        
        atten_h = self.lin_atten(h_.unsqueeze(1))


        return atten_h

class TimeReasoning(nn.Module):
    def __init__(self, hidden_size, mid_size, state_size, num_token, edge_size):
        super(TimeReasoning, self).__init__()
        # blocks = []
        self.entity_cell = TimeReasoningCell(hidden_size=hidden_size, mid_size=mid_size, num_token=num_token)
        self.trans_cell = TransCell(edge_size, mid_size)
        self.mid_size = mid_size
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.decoder = Decoder(hidden_size, num_token, num_layers=0)

    def _kl_divergence_recon(self, prior, posterior, post_pre, post_back, obs, v, v_mask):
        z = posterior.sample()
        l = posterior.log_prob(z) - prior.log_prob(z)

        z_p = post_back.sample()
        l2 = post_back.log_prob(z_p) - post_pre.log_prob(z_p)
        # act = z.exp()
        act = F.softmax(posterior.logits.masked_fill(v_mask == 0, -float('inf')) , dim=1)
        h_act = torch.matmul(v.transpose(-1, -2), act.unsqueeze(-1)).squeeze(-1)
        recons_1 = self.decoder(h_act)

        # recon_erro_1 = F.cross_entropy(recons_1, obs, reduction='none')
        recon_erro_1 = F.binary_cross_entropy_with_logits(recons_1, obs, reduction='none').mean(-1)

        return recon_erro_1, l, l2 

    def forward(self, q_embeddings, object_list, e_label, e_s, e_e, e_mask, edge, r_mask, logits, v_mask):
        batch_size, max_len = e_mask.shape
        logits0 = 10. * torch.zeros(batch_size, self.state_size).cuda().detach()
        z0 = modulars.GumbelCategorical(logits0)
        state = {
            'hidden_pre': object_list,
            'z' : z0
        }
        v_input = object_list

        z_ = z0
        Loss_time = []
        Loss_recon = []
        for i in range(max_len-1):

            trans_emb = q_embeddings[i]

            state_pre = state['z']

            logit_input = logits[:,i,:,:]
            
            # q_input = q[entity_end, torch.arange(batch_size)]
            q_input = e_label[:, i, :]

            state = self.entity_cell(logit_input, v_mask)
            recon, l_forward, l_back = self._kl_divergence_recon(z_, state['z'], state_pre, state['z_back'], q_input, object_list, v_mask)

            z_ = self.trans_cell(trans_emb, edge, r_mask, state['z'], v_mask)

            if i == 0:
                l_t = recon
            else:
                l_t = recon + l_forward #+ l_back

            Loss_time.append(l_t.unsqueeze(-1))
            Loss_recon.append(recon.unsqueeze(-1))
  
        L_t = torch.cat(Loss_time, dim=1) * e_mask[:, :-1] #/ (e_mask[:, :-1].sum(1) + 1.).unsqueeze(-1)
        L_recon = torch.cat(Loss_recon, dim=1) * e_mask[:, :-1]
        loss_time = torch.sum(L_t, dim=1).mean(0)
        loss_r= torch.sum(L_recon, dim=1).mean(0)

        return loss_time

class TransCell(nn.Module):
    def __init__(self, edge_size, hidden_size):
        super(TransCell, self).__init__()
        self.linear_r = FCNet(edge_size, 256)
        self.linear_q = FCNet(hidden_size, 256)
        self.linear_out = FCNet(256, 1)
        
    def forward(self, q, trans_mat, r_mask, z, v_mask):
        trans_mat = self.linear_r(trans_mat)
        batch_size, num_obj, num_obj = r_mask.shape
        q = self.linear_q(q)
        q_ = q.unsqueeze(1).unsqueeze(1).expand(trans_mat.shape)
        relation = self.linear_out(trans_mat * q_).squeeze(-1)
        
        relation_mat = F.softmax(relation.masked_fill(r_mask == 0., -float('inf')), dim=1)

        logits = torch.matmul(relation_mat, z.logits.unsqueeze(-1)).squeeze(-1)
        z_ = modulars.GumbelCategorical(logits)

        return z_

class TimeReasoningCell(nn.Module):
    def __init__(self, hidden_size, mid_size, num_token):
        super(TimeReasoningCell, self).__init__()

        self.encoder_posterior = Encoder(6, 1, num_layers=0)
        self.encoder_back = Encoder(6, 1, num_layers=0)

    def _ditribution(self, input, encoder, v_mask):
        # logits = encoder(input)
        logits = input.sum(-1)
        batch_size = logits.shape[0] 
        logits = logits.view(batch_size, -1)
        logits.data.masked_fill_(v_mask==0, -1e5)
        dist = modulars.GumbelCategorical(logits)
        return dist

    def forward(self, logits, v_mask):
        logits.data.masked_fill_(logits == -float('inf'), 0)
        posterior = self._ditribution(logits, self.encoder_posterior, v_mask)
        posterior_back = self._ditribution(logits, self.encoder_back, v_mask)
        
        return {
            'z': posterior,
            'z_back': posterior_back,
            # 'z_prior': prior
            }



