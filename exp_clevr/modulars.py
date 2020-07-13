'''
question words as time series
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from itertools import chain
from misc import reverse_padded_sequence
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence

class FCNet(nn.Module):
    def __init__(self, in_size, out_size, activate=None, drop=0.0, bias=True):
        super(FCNet, self).__init__()
        self.lin = weight_norm(nn.Linear(in_size, out_size, bias=bias), dim=None)
        # self.ln = nn.LayerNorm(out_size)
        self.drop_value = drop
        self.drop = nn.Dropout(drop)

        self.activate = activate.lower() if (activate is not None) else None
        if activate == 'relu':
            self.ac_fn = nn.ReLU()
        elif activate == 'sigmoid':
            self.ac_fn = nn.Sigmoid()
        elif activate == 'tanh':
            self.ac_fn = nn.Tanh()

    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)


        x = self.lin(x)

        if self.activate is not None:
            # x = self.ln(x)
            x = self.ac_fn(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_size, state_size, num_layers):
        super(Encoder, self).__init__()
        self.lin1 = FCNet(input_size, 512, activate='relu')
        self.layers = nn.ModuleList([FCNet(512, 512, activate='relu') for _ in range(num_layers)])
        self.lin2 = FCNet(512, state_size, drop=0.2)
    def forward(self, input):
        x = self.lin1(input)
        # for layer in self.layers:
        #     x = layer(x) + x
        x = self.lin2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, state_size, obs_size, num_layers):
        super(Decoder, self).__init__()
        # self.lin = FCNet(state_size, 2048, activate='relu')
        # self.layers = nn.ModuleList([FCNet(2048, 2048, activate='relu') for _ in range(num_layers)])
        self.lin_o = FCNet(state_size, obs_size, drop=0.2)
    def forward(self, input):
        # x = self.lin(input)
        # for layer in self.layers:
            # x = layer(x) + x
        return self.lin_o(input)

class GumbelCategorical():
    def __init__(self, logits, temperature=2.):
        self.k = torch.tensor([logits.shape[1]], dtype=torch.float32).cuda()
        self.logits = logits
        self.temperature = torch.tensor([temperature]).cuda()

    def log_prob(self, x):
        log_Z = (torch.lgamma(self.k) + (self.k - 1) * self.temperature.log())
        log_prob_unnormalized = (F.log_softmax(self.logits - self.temperature * x, dim=1)).sum(1)

        return log_prob_unnormalized + log_Z

    def sample(self):
        gumbel = -(-(torch.rand_like(self.logits)+1e-30).log()).log()
        # sample = F.softmax((self.logits + gumbel) / self.temperature, dim=1)
        relax = (self.logits + gumbel) / self.temperature
        sample = relax - torch.logsumexp(relax, dim=1, keepdim=True)

        return sample

class BiGRUEncoder_2(nn.Module):
    def __init__(self, dim_word, dim_hidden):
        super().__init__()
        self.forward_gru = nn.GRU(dim_word, dim_hidden//2)
        self.backward_gru = nn.GRU(dim_word, dim_hidden//2)
        for name, param in chain(self.forward_gru.named_parameters(), self.backward_gru.named_parameters()):
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, input_embedded, input_seq_lens, e_s, e_e):
        """
            Input:
                input_seqs: [seq_max_len, batch_size]
                input_seq_lens: [batch_size]
        """
        embedded = input_embedded # [seq_max_len, batch_size, word_dim]
        self.forward_gru.flatten_parameters()
        forward_outputs = self.forward_gru(embedded)[0] # [seq_max_len, batch_size, dim_hidden/2]
        backward_embedded = reverse_padded_sequence(embedded, input_seq_lens)
        self.backward_gru.flatten_parameters()
        backward_outputs = self.backward_gru(backward_embedded)[0]
        backward_outputs = reverse_padded_sequence(backward_outputs, input_seq_lens)
        outputs = torch.cat([forward_outputs, backward_outputs], dim=2) # [seq_max_len, batch_size, dim_hidden]
        # indexing outputs via input_seq_lens
        hidden = []
        hidden_entiy = []
        hidden_trans = []
        batch_size, max_len = e_e.shape
        for i, l in enumerate(input_seq_lens):
            hidden.append(
                torch.cat([forward_outputs[l-1, i], backward_outputs[0, i]], dim=0)
            )
        hidden = torch.stack(hidden) # (batch_size, dim)
        for k in range(max_len - 1):
            hidden_entiy.append(
                torch.cat([forward_outputs[e_e[:,k]-1, torch.arange(batch_size)], backward_outputs[e_s[:,k], torch.arange(batch_size)]], dim=1)
                )
            hidden_trans.append(
                torch.cat([forward_outputs[e_e[:,k + 1]-1, torch.arange(batch_size)], backward_outputs[e_s[:,k], torch.arange(batch_size)]], dim=1)
                )
        hidden_entiy.append(hidden)
        hidden_entiy = torch.stack(hidden_entiy)
        hidden_trans = torch.stack(hidden_trans)
        return outputs, hidden, hidden_entiy, hidden_trans

class BiGRUEncoder(nn.Module):
    def __init__(self, dim_word, dim_hidden):
        super().__init__()
        self.forward_gru = nn.GRU(dim_word, dim_hidden//2)
        self.backward_gru = nn.GRU(dim_word, dim_hidden//2)
        for name, param in chain(self.forward_gru.named_parameters(), self.backward_gru.named_parameters()):
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, input_embedded, input_seq_lens):
        """
            Input:
                input_seqs: [seq_max_len, batch_size]
                input_seq_lens: [batch_size]
        """
        embedded = input_embedded # [seq_max_len, batch_size, word_dim]
        self.forward_gru.flatten_parameters()
        forward_outputs = self.forward_gru(embedded)[0] # [seq_max_len, batch_size, dim_hidden/2]
        backward_embedded = reverse_padded_sequence(embedded, input_seq_lens)
        self.backward_gru.flatten_parameters()
        backward_outputs = self.backward_gru(backward_embedded)[0]
        backward_outputs = reverse_padded_sequence(backward_outputs, input_seq_lens)
        outputs = torch.cat([forward_outputs, backward_outputs], dim=2) # [seq_max_len, batch_size, dim_hidden]
        # indexing outputs via input_seq_lens
        hidden = []
        for i, l in enumerate(input_seq_lens):
            hidden.append(
                torch.cat([forward_outputs[l-1, i], backward_outputs[0, i]], dim=0)
                )
        hidden = torch.stack(hidden) # (batch_size, dim)
        return outputs, hidden 

# class WordEmbedding(nn.Module):
#     def __init__(self, classes, embedding_features):
#         super(WordEmbedding, self).__init__()
#         classes = list(classes)

#         self.embed = nn.Embedding(len(classes) + 1, embedding_features, padding_idx=len(classes))
#         assert weight_init.shape == (len(classes), embedding_features)
#         self.embed.weight.data[:len(classes)] = weight_init


#     def forward(self, q, q_len):
#         embedded = self.embed(q)

#         return embedded

class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.lin_v = FCNet(v_features, mid_features, activate='relu')  # let self.lin take care of bias
        self.lin_q = FCNet(q_features, mid_features, activate='relu')
        self.lin = FCNet(mid_features, glimpses, drop=drop)

    def forward(self, v, q):
        """
        v = batch, num_obj, dim
        q = batch, dim
        """
        v_mid = self.lin_v(v)
        q = self.lin_q(q)
        batch, num_obj, _ = v.shape
        _, q_dim = q.shape
        q = q.unsqueeze(1).expand(batch, num_obj, q_dim)

        x = v_mid * q
        x = self.lin(x)  # batch, num_obj, glimps

        x = F.softmax(x, dim=1)
        v_apply = apply_attention(v.transpose(-1, -2),x)
        return v_apply, x


def apply_attention(input, attention):
    """
    input = batch, dim, num_obj
    attention = batch, num_obj, glimps
    """
    batch, dim, _ = input.shape
    _, _, glimps = attention.shape
    x = input @ attention  # batch, dim, glimps
    assert (x.shape[1] == dim)
    assert (x.shape[2] == glimps)
    return x.view(batch, -1)

class GRUEncoder(nn.Module):
    def __init__(self, dim_word, dim_hidden):
        super(GRUEncoder, self).__init__()

        self.lstm = nn.GRU(input_size=dim_word,
                        hidden_size=dim_hidden,
                        num_layers=1,
                        bidirectional=False)

    def forward(self, embedded, q_len):
        self.lstm.flatten_parameters()
        # packed = pack_padded_sequence(embedded,  q_len, batch_first=False)
        hid = self.lstm(embedded)[0]
        hidden = []
        for i, l in enumerate(q_len):
            hidden.append(
                hid[l-1, i]
                )
        hidden_out = torch.stack(hidden) # (batch_size, dim)
        return hid, hidden_out


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    """
    def __init__(self, v_dim, q_dim, h_dim, h_out, dropout=[.2,.5], k=3):
        super(BCNet, self).__init__()
        
        self.c = 32
        self.k = k
        self.v_dim = v_dim; self.q_dim = q_dim
        self.h_dim = h_dim; self.h_out = h_out

        self.v_net = FCNet(v_dim, h_dim * self.k, activate='relu', drop=dropout[0])
        self.q_net = FCNet(q_dim, h_dim * self.k, activate='relu', drop=dropout[0])
        self.dropout = nn.Dropout(dropout[1]) # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)
        
        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            logits = torch.einsum('bvk,bqk->bvqk', (v_, q_))
            return logits

        # low-rank bilinear pooling using einsum
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            return logits # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        else: 
            v_ = self.dropout(self.v_net(v)).transpose(1,2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1,2).unsqueeze(2)
            d_ = torch.matmul(v_, q_) # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1,2).transpose(2,3)) # b x v x q x h_out
            return logits.transpose(2,3).transpose(1,2) # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v) # b x v x d
        q_ = self.q_net(q) # b x q x d
        logits = torch.einsum('bvk,bvq,bqk->bqk', (v_, w, q_))
        if 1 < self.k:
            logits = logits.unsqueeze(1) # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k # sum-pooling
        return logits

class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3), \
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True, logit=False, mask_with=-float('inf')):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v,q) # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, mask_with)

        if not logit:
            p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
            return p.view(-1, v_num, q_num), logits.squeeze(1)

        return logits