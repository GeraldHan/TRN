import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import modulars
from modulars import FCNet, Encoder, Decoder
from counting import Counter

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

        self.num_classes = len(self.vocab['answer_token_to_idx'])
        

        self.num_token = len(self.vocab['question_token_to_idx'])

        self.classifier = Classifier(
            in_features=(self.glimpses * self.dim_vision, self.mid_size),
            mid_features=self.dim_hidden,
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

        self.vision_to_v = FCNet(self.dim_vision, self.dim_hidden, drop=0.3, bias=False)
        self.map_two_v_to_edge = FCNet(self.dim_hidden*2, self.dim_edge, bias=False)

        self.timereasoning = TimeReasoning(hidden_size=self.dim_vision,
                                           mid_size=self.mid_size,
                                           state_size=self.state_size,
                                           num_token=self.num_token,
                                           edge_size=self.dim_edge)

                                           
    def forward(self, q, vision, e_s, e_e, e_mask, e_label, q_mask, r_mask, q_len):
        v = vision
        b = vision[:, : ,-4:]
        q = q.permute(1, 0)  #[seq_len, batch]
        w_emb = self.token_embedding(q) #[seq_len]
        hidden_q, q_out, hidden_entity, hidden_trans = self.text(w_emb, list(q_len.data), e_s, e_e)
        q_max, batch_size= q.shape
        _, num_obj, _ = v.shape
        v_mask = v.sum(-1).masked_fill(v.sum(-1) > 0, 1)

        vision_feat = v / (v.norm(p=2, dim=1, keepdim=True) + 1e-12)
        feat_inputs = self.vision_to_v(vision_feat)
        feat_inputs_expand_0 = feat_inputs.unsqueeze(1).expand(batch_size, num_obj, num_obj, self.dim_hidden)
        feat_inputs_expand_1 = feat_inputs.unsqueeze(2).expand(batch_size, num_obj, num_obj, self.dim_hidden)
        feat_edge = torch.cat([feat_inputs_expand_0, feat_inputs_expand_1], dim=3) # (bs, num_feat, num_feat, 2*dim_v)
        feat_edge = self.map_two_v_to_edge(feat_edge)
        

        loss_time, h_out = self.timereasoning(hidden_entity, hidden_trans, v, e_s, e_e, e_mask, e_label, feat_edge, r_mask)

        answer = self.classifier(h_out, q_out)

        return answer, loss_time

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.lin11 = nn.Linear(in_features[0], mid_features)
        self.lin12 = nn.Linear(in_features[1], mid_features)
        self.lin2 = FCNet(mid_features, mid_features, activate='relu')
        self.lin3 = FCNet(mid_features, out_features)

    def forward(self, x, y):
        x = self.lin11(x) * self.lin12(y)
        x = self.lin2(x)
        x = self.lin3(self.drop(x))
        return x


class TimeReasoning(nn.Module):
    def __init__(self, hidden_size, mid_size, state_size, num_token, edge_size):
        super(TimeReasoning, self).__init__()
        blocks = []
        for i in range(2):
            blocks.append(TimeReasoningCell(hidden_size=hidden_size, mid_size=mid_size, num_token=num_token))
        # self.entity_cell = TimeReasoningCell(hidden_size=hidden_size, mid_size=mid_size, num_token=num_token)
        self.cell = nn.ModuleList(blocks)
        self.trans_cell = TransCell(edge_size, mid_size)
        self.mid_size = mid_size
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.decoder = Decoder(hidden_size, num_token, num_layers=0)
        


    def _kl_divergence_recon(self, prior, posterior, post_pre, post_back, obs, v):
        z = posterior.sample()
        l = posterior.log_prob(z) - prior.log_prob(z)

        z_p = post_back.sample()
        l2 = post_back.log_prob(z_p) - post_pre.log_prob(z_p)
        act = z.exp()
        h_act = torch.matmul(v.transpose(-1, -2), act.unsqueeze(-1)).squeeze(-1)
        recons_1 = self.decoder(h_act)

      
        recon_erro_1 = F.binary_cross_entropy_with_logits(recons_1, obs, reduction='none').mean(-1)

        return recon_erro_1, 0.1 * l, l2 

    def forward(self, hidden_entity, hidden_trans, object_list, e_s, e_e, e_mask, e_label, edge, r_mask):
        batch_size, max_len = e_mask.shape
        hidden_h = object_list.mean(1)
        logits0 = 10. * torch.zeros(batch_size, self.state_size).cuda().detach()
        z0 = modulars.GumbelCategorical(logits0)
        state = {
            'hidden_pre': object_list,
            'hidden_h': hidden_h,
            'z' : z0
        }
        v_input = object_list

        z_ = z0
        Loss_time = []
        for i in range(max_len-1):

            entity_emb = hidden_entity[i]
            trans_emb = hidden_trans[i]

            state_pre = state['z']
            
            q_input = e_label[:, i, :]

            state = self.cell[0](entity_emb, v_input)
            recon, l_forward, l_back = self._kl_divergence_recon(z_, state['z'], state_pre, state['z_back'], q_input, object_list)

            z_ = self.trans_cell(trans_emb, edge, r_mask, state_pre)

            if i == 0:
                l_t = recon
            else:
                l_t = recon + l_forward #+ l_back

            v_input = state['hidden_pre']
        

            Loss_time.append(l_t.unsqueeze(-1))
  
        L_t = torch.cat(Loss_time, dim=1) * e_mask[:, :-1] #/ (e_mask[:, :-1].sum(1) + 1.).unsqueeze(-1)

        state = self.cell[-1](hidden_entity[-1], v_input)
        recon, l_forward, l_back = self._kl_divergence_recon(z_, state['z'], state_pre, state['z_back'], q_input, object_list)
        
        loss_time = torch.sum(L_t, dim=1).mean(0) + l_forward.mean(0)

        node = F.softmax(state['z'].logits, dim=1)
        h_out = torch.matmul(state['hidden_pre'].transpose(-1, -2), node.unsqueeze(-1)).squeeze(-1)


        return loss_time, h_out

class TransCell(nn.Module):
    def __init__(self, edge_size, hidden_size):
        super(TransCell, self).__init__()
        # self.linear_r = FCNet(edge_size, edge_size//2)
        self.linear_q = FCNet(hidden_size, edge_size)
        self.linear_out = FCNet(edge_size, 1, drop=0.1)
        
    def forward(self, q, trans_mat, r_mask, z):
        # trans_mat = self.linear_r(trans_mat)
        q = self.linear_q(q)
        q_ = q.unsqueeze(1).unsqueeze(1).expand(trans_mat.shape)
        relation = self.linear_out(trans_mat * q_).squeeze(-1)

        relation_mat = F.softmax(relation.masked_fill(r_mask == 0., -1e12), dim=1)
        logits = torch.matmul(relation_mat, z.logits.unsqueeze(-1)).squeeze(-1)
        z_ = modulars.GumbelCategorical(logits)

        return z_

class TimeReasoningCell(nn.Module):
    def __init__(self, hidden_size, mid_size, num_token):
        super(TimeReasoningCell, self).__init__()
        self.linear_h = FCNet(hidden_size, mid_size, activate='relu')
        self.linear_q = FCNet(mid_size, mid_size, activate='relu')

        self.hidden_size = hidden_size
        # self.bilinear = modulars.BiAttention(mid_size, hidden_size, hidden_size, glimpse=1)

        self.encoder_prior = Encoder(hidden_size, 1, num_layers=0)
        self.encoder_posterior = Encoder(mid_size, 1, num_layers=0)
        self.encoder_back = Encoder(mid_size, 1, num_layers=0)
        
        # self.b_net = modulars.BCNet(mid_size, hidden_size, hidden_size, None, k=1)

    def _ditribution(self, input, encoder):

        logits = encoder(input)
        batch_size = logits.shape[0] 
        logits = logits.view(batch_size, -1)
        dist = modulars.GumbelCategorical(logits)
        return dist

    def forward(self, q, v):
        batch, num_obj, _ = v.shape
        _, q_dim = q.shape

        hidden_prior = v

        h_to_q = self.linear_h(v)
        q = self.linear_q(q)

        q_to_h = q.unsqueeze(1).expand(batch, num_obj, q_dim)

        hidden_now = h_to_q * q_to_h

        posterior = self._ditribution(hidden_now, self.encoder_posterior)

        posterior_back = self._ditribution(hidden_now, self.encoder_back)
        
        return {
            'hidden_pre':v,
            'z': posterior,
            'z_back': posterior_back,
            }




