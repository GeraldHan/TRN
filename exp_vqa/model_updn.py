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

        self.text = modulars.GRUEncoder(
            dim_word=self.dim_word,
            dim_hidden=self.dim_hidden,
        )

        self.attention = modulars.Attention(
            v_features=self.dim_vision,
            q_features=self.dim_hidden,
            mid_features=self.mid_size,
            glimpses=self.glimpses,
            drop=0.2
        )
                                           
    def forward(self, q, vision, e_s, e_e, e_mask, e_label, q_mask, r_mask, q_len):
        v = vision[:,:,:-4]
        boxes = vision[:, : ,-4:]
        q = q.permute(1, 0)  #[seq_len, batch]
        w_emb = self.token_embedding(q) #[seq_len]
        hidden_q, q_out = self.text(w_emb, list(q_len.data))
        q_max, batch_size= q.shape
        _, num_obj, _ = v.shape


        h_out, att = self.attention(v, q_out)
        answer = self.classifier(h_out, q_out)
        loss_time = torch.zeros_like(v).mean()

        return answer, att

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



