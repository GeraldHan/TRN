import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import modulars
from modulars import FCNet, Encoder, Decoder
from torch.nn.utils import weight_norm

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

        self.text = modulars.BiGRUEncoder(
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

        self.apply_attention = ApplyAttention(
            v_features=self.dim_vision,
            q_features=self.dim_hidden,
            mid_features=self.mid_size,
            glimpses=glimpses,
            num_obj=objects,
            drop=0.2,
        )
                                         
    def forward(self, q, v, e_s, e_e, e_mask, edge, r_mask, q_mask, q_len):
        q = q.permute(1, 0)  #[seq_len, batch]
        w_emb = self.token_embedding(q) #[seq_len]
        hidden_q, q_out = self.text(w_emb, list(q_len.data))
        q_max, batch_size= q.shape
        _, num_obj, _ = v.shape

        v_mask = v.sum(-1).masked_fill(v.sum(-1) > 0, 1)
        
        q_input = hidden_q.transpose(0, 1)

        atten, logits = self.attention(v, q_input, v_mask, q_mask) # batch x glimpses x v_num x q_num

        new_q = self.apply_attention(v, q_input, v_mask, q_mask, atten, logits)
        answer = self.classifier(new_q)

        loss_time = torch.zeros_like(v_mask)

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

        # apply v_mask, q_mask
        logits.data.masked_fill_(v_mask.unsqueeze(1).unsqueeze(3).expand(logits.shape) == 0, -float('inf'))
        #logits.masked_fill_(q_mask.unsqueeze(1).unsqueeze(2).expand(logits.shape) == 0, -float('inf'))

        atten = F.softmax(logits.view(-1, self.glimpses, v_num * q_num), 2)
        return atten.view(-1, self.glimpses, v_num, q_num), logits


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





