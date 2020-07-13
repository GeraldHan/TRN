import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import modulars
from modulars import FCNet, Encoder, Decoder
from torch.nn.utils import weight_norm


APPLY_MASK = True
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
        
        self.num_inter_head = 8
        self.num_intra_head = 8
        self.num_block = 6

        self.num_classes = len(self.vocab['answer_token_to_idx'])
        

        self.num_token = len(self.vocab['question_token_to_idx'])

        self.token_embedding = nn.Embedding(self.num_token, self.dim_word)

        for p in self.token_embedding.parameters():
            p.require_grad = False

        self.dropout = nn.Dropout(self.dropout_prob)

        self.text = modulars.BiGRUEncoder_2(
            dim_word=self.dim_word,
            dim_hidden=self.dim_hidden,
        )

        # self.vision_to_v = FCNet(self.dim_vision, self.dim_hidden, drop=0.3, bias=False)
        # self.map_two_v_to_edge = FCNet(self.dim_hidden*2, self.dim_edge, bias=False)

        # self.timereasoning = TimeReasoning(hidden_size=self.dim_vision,
        #                                    mid_size=self.mid_size,
        #                                    state_size=self.state_size,
        #                                    num_token=self.num_token,
        #                                    edge_size=self.dim_edge)

        self.interIntraBlocks = MultiBlock(
            num_block=self.num_block,
            v_size=self.dim_vision,
            q_size=self.dim_hidden,
            output_size=self.mid_size,
            num_inter_head=self.num_inter_head,
            num_intra_head=self.num_intra_head,
            drop=0.1,
        )

        self.classifier = Classifier(
            in_features=self.mid_size,
            mid_features=2048,
            out_features=self.num_classes,
            drop=0.5,)
                                         
    def forward(self, q, v, e_s, e_e, e_mask, e_label, edge, r_mask, q_mask, q_len):
        q = q.permute(1, 0)  #[seq_len, batch]
        w_emb = self.token_embedding(q) #[seq_len]
        hidden_q, q_out, hidden_entity, hidden_trans = self.text(w_emb, list(q_len.data), e_s, e_e)
        q_max, batch_size= q.shape
        _, num_obj, _ = v.shape

        v_mask = v.sum(-1).masked_fill(v.sum(-1) > 0, 1)


        hidden_input = hidden_entity.transpose(0,1)

        q_mask = torch.ones_like(e_mask)
        v, q = self.interIntraBlocks(v, hidden_input, v_mask, q_mask)

        answer = self.classifier(v, q, v_mask, q_mask)
        loss_time = torch.zeros_like(v_mask).mean()

        return answer, loss_time

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.lin1 = FCNet(in_features, mid_features, activate='relu', drop=drop/2.5)
        self.lin2 = FCNet(mid_features, out_features, drop=drop)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, 512]
        q: question            [batch, max_len, 512]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        num_obj = v_mask.shape[1]
        max_len = q_mask.shape[1]

        if APPLY_MASK:
            v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
            q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)
        else:
            v_mean = v.sum(1) / num_obj
            q_mean = q.sum(1) / max_len
        out = self.lin1(v_mean * q_mean)
        out = self.lin2(out)
        return out

class SingleBlock(nn.Module):
    """
    Single Block Inter-/Intra-modality stack multiple times
    """
    def __init__(self, num_block, v_size, q_size, output_size, num_inter_head, num_intra_head, drop=0.0):
        super(SingleBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_inter_head = num_inter_head
        self.num_intra_head = num_intra_head
        self.num_block = num_block

        self.v_lin = FCNet(v_size, output_size, drop=drop)
        self.q_lin = FCNet(q_size, output_size, drop=drop)

        self.v2q_interBlock = OneSideInterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop)
        self.q2v_interBlock = OneSideInterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop)
        self.intraBlock = DyIntraModalityUpdate(output_size, output_size, output_size, num_intra_head, drop)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        # transfor features
        v = self.v_lin(v)
        q = self.q_lin(q)
        v_container = [v]
        q_container = [q]
        result_v = [v]
        result_q = [q]
        for i in range(self.num_block):
            q1 = self.v2q_interBlock(v_container[-1], q_container[-1], v_mask, q_mask)
            q_container.append(q1)
            v1 = self.q2v_interBlock(q_container[-1] + q_container[-2], v_container[-1], q_mask, v_mask)
            v_container.append(v1)
            v2, q2 = intraBlock(v_container[-1] + v_container[-2], q_container[-1] + q_container[-2], v_mask, q_mask)
            v_container.append(v2)
            q_container.append(q2)
            result_v.append(v1)
            result_v.append(v2)
            result_q.append(q1)
            result_q.append(q2)
            v_container.append(v_container[-1] + v_container[-2] + v_container[-3])
            q_container.append(q_container[-1] + q_container[-2] + q_container[-3])
        return sum(result_v), sum(result_q)

class MultiBlock(nn.Module):
    """
    Multi Block (different parameters) Inter-/Intra-modality
    """
    def __init__(self, num_block, v_size, q_size, output_size, num_inter_head, num_intra_head, drop=0.0):
        super(MultiBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_inter_head = num_inter_head
        self.num_intra_head = num_intra_head
        self.num_block = num_block

        self.v_lin = FCNet(v_size, output_size, drop=drop)
        self.q_lin = FCNet(q_size, output_size, drop=drop)

        blocks = []
        for i in range(num_block):
            blocks.append(OneSideInterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop))
            blocks.append(OneSideInterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop))
            blocks.append(DyIntraModalityUpdate(output_size, output_size, output_size, num_intra_head, drop))
        self.multi_blocks = nn.ModuleList(blocks)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        v = self.v_lin(v)
        q = self.q_lin(q)
        v_container = [v]
        q_container = [q]
        result_v = [v]
        result_q = [q]
        # use dense residual 
        for i in range(self.num_block):
            q1 = self.multi_blocks[i*3+0](v_container[-1], q_container[-1], v_mask, q_mask)
            q_container.append(q1)
            v1 = self.multi_blocks[i*3+1](q_container[-1] + q_container[-2], v_container[-1], q_mask, v_mask)
            v_container.append(v1)
            v2, q2 = self.multi_blocks[i*3+2](v_container[-1] + v_container[-2], q_container[-1] + q_container[-2], v_mask, q_mask)
            v_container.append(v2)
            q_container.append(q2)
            result_v.append(v1)
            result_v.append(v2)
            result_q.append(q1)
            result_q.append(q2)
            v_container.append(v_container[-1] + v_container[-2] + v_container[-3])
            q_container.append(q_container[-1] + q_container[-2] + q_container[-3])
            
        return sum(result_v), sum(result_q)

class InterModalityUpdate(nn.Module):
    """
    Inter-modality Attention Flow
    """
    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
        super(InterModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head

        self.v_lin = FCNet(v_size, output_size * 3, drop=drop)
        self.q_lin = FCNet(q_size, output_size * 3, drop=drop)

        self.v_output = FCNet(output_size + v_size, output_size, drop=drop)
        self.q_output = FCNet(output_size + q_size, output_size, drop=drop)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        batch_size, num_obj = v_mask.shape
        _         , max_len = q_mask.shape
        # transfor features
        v_trans = self.v_lin(v)
        q_trans = self.q_lin(q)
        # mask all padding object/word features
        if APPLY_MASK:
            v_trans = v_trans * v_mask.unsqueeze(2)
            q_trans = q_trans * q_mask.unsqueeze(2)
        # split for different use of purpose
        v_key, v_qry, v_val = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_key, q_qry, q_val = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
        # apply multi-head
        v_key_set = torch.split(v_key, v_key.size(2) // self.num_head, dim=2)
        v_qry_set = torch.split(v_qry, v_qry.size(2) // self.num_head, dim=2)
        v_val_set = torch.split(v_val, v_val.size(2) // self.num_head, dim=2)
        q_key_set = torch.split(q_key, q_key.size(2) // self.num_head, dim=2)
        q_qry_set = torch.split(q_qry, q_qry.size(2) // self.num_head, dim=2)
        q_val_set = torch.split(q_val, q_val.size(2) // self.num_head, dim=2)
        # multi-head
        for i in range(self.num_head):
            v_key_slice, v_qry_slice, v_val_slice = v_key_set[i], v_qry_set[i], v_val_set[i]  #[batch, num_obj, feat_size]
            q_key_slice, q_qry_slice, q_val_slice = q_key_set[i], q_qry_set[i], q_val_set[i]  #[batch, max_len, feat_size]
            # inner product & set padding object/word attention to negative infinity & normalized by square root of hidden dimension
            q2v = (v_qry_slice @ q_key_slice.transpose(1,2)) / ((self.output_size // self.num_head) ** 0.5)  #[batch, num_obj, max_len]
            v2q = (q_qry_slice @ v_key_slice.transpose(1,2)) / ((self.output_size // self.num_head) ** 0.5)  #[batch, max_len, num_obj]
            if APPLY_MASK:
                q2v.masked_fill_(q_mask.unsqueeze(1).expand([batch_size, num_obj, max_len]) == 0, -float('inf')) 
                v2q.masked_fill_(v_mask.unsqueeze(1).expand([batch_size, max_len, num_obj]) == 0, -float('inf')) 
            # softmax attention
            interMAF_q2v = F.softmax(q2v, dim=2).unsqueeze(3) #[batch, num_obj, max_len, 1]
            interMAF_v2q = F.softmax(v2q, dim=2).unsqueeze(3) #[batch, max_len, num_obj, 1]
            # calculate update input (each head of multi-head is calculated independently and concatenate together)
            v_update = (interMAF_q2v * q_val_slice.unsqueeze(1)).sum(2) if (i==0) else torch.cat((v_update, (interMAF_q2v * q_val_slice.unsqueeze(1)).sum(2)), dim=2)
            q_update = (interMAF_v2q * v_val_slice.unsqueeze(1)).sum(2) if (i==0) else torch.cat((q_update, (interMAF_v2q * v_val_slice.unsqueeze(1)).sum(2)), dim=2)
        # update new feature
        cat_v = torch.cat((v, v_update), dim=2)
        cat_q = torch.cat((q, q_update), dim=2)
        updated_v = self.v_output(cat_v)
        updated_q = self.q_output(cat_q)
        return updated_v, updated_q


class OneSideInterModalityUpdate(nn.Module):
    """
    One-Side Inter-modality Attention Flow
    
    According to original paper, instead of parallel V->Q & Q->V, we first to V->Q and then Q->V
    """
    def __init__(self, src_size, tgt_size, output_size, num_head, drop=0.0):
        super(OneSideInterModalityUpdate, self).__init__()
        self.src_size = src_size
        self.tgt_size = tgt_size
        self.output_size = output_size
        self.num_head = num_head

        self.src_lin = FCNet(src_size, output_size * 2, drop=drop)
        self.tgt_lin = FCNet(tgt_size, output_size, drop=drop)

        self.tgt_output = FCNet(output_size + tgt_size, output_size, drop=drop)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        src: src feature      [batch, num_src, feat_size]
        tgt: tgt feautre      [batch, num_tgt, feat_size]
        src_mask              [batch, num_src]
        tgt_mask              [batch, num_tgt]
        """
        batch_size, num_src = src_mask.shape
        _         , num_tgt = tgt_mask.shape
        
        src_trans = self.src_lin(src)
        tgt_trans = self.tgt_lin(tgt)
        
        if APPLY_MASK:
            src_trans = src_trans * src_mask.unsqueeze(2)
            tgt_trans = tgt_trans * tgt_mask.unsqueeze(2)
        
        src_key, src_val = torch.split(src_trans, src_trans.size(2) // 2, dim=2)
        tgt_qry = tgt_trans

        src_key_set = torch.split(src_key, src_key.size(2) // self.num_head, dim=2)
        src_val_set = torch.split(src_val, src_val.size(2) // self.num_head, dim=2)
        tgt_qry_set = torch.split(tgt_qry, tgt_qry.size(2) // self.num_head, dim=2)
        for i in range(self.num_head):
            src_key_slice, tgt_qry_slice, src_val_slice = src_key_set[i], tgt_qry_set[i], src_val_set[i]
            src2tgt = (tgt_qry_slice @ src_key_slice.transpose(1,2)) / ((self.output_size // self.num_head) ** 0.5)  #[batch, tgt_num, src_num]
            if APPLY_MASK:
                src2tgt.masked_fill_(src_mask.unsqueeze(1).expand([batch_size, num_tgt, num_src]) == 0, -float('inf'))
            interMAF_src2tgt = F.softmax(src2tgt, dim=2).unsqueeze(3)
            tgt_update = (interMAF_src2tgt * src_val_slice.unsqueeze(1)).sum(2) if (i==0) else torch.cat((tgt_update, (interMAF_src2tgt * src_val_slice.unsqueeze(1)).sum(2)), dim=2)
        cat_tgt = torch.cat((tgt, tgt_update), dim=2)
        update_tgt = self.tgt_output(cat_tgt)
        return update_tgt


class DyIntraModalityUpdate(nn.Module):
    """
    Dynamic Intra-modality Attention Flow
    """
    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
        super(DyIntraModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head

        self.v4q_gate_lin = FCNet(v_size, output_size, drop=drop)
        self.q4v_gate_lin = FCNet(q_size, output_size, drop=drop)

        self.v_lin = FCNet(v_size, output_size * 3, drop=drop)
        self.q_lin = FCNet(q_size, output_size * 3, drop=drop)

        self.v_output = FCNet(output_size, output_size, drop=drop)
        self.q_output = FCNet(output_size, output_size, drop=drop)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        batch_size, num_obj = v_mask.shape
        _         , max_len = q_mask.shape
        # conditioned gating vector
        if APPLY_MASK:
            v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
            q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)
        else:
            v_mean = v.sum(1) / num_obj
            q_mean = q.sum(1) / max_len

        v4q_gate = self.sigmoid(self.v4q_gate_lin(v_mean)).unsqueeze(1) #[batch, 1, feat_size]
        q4v_gate = self.sigmoid(self.q4v_gate_lin(q_mean)).unsqueeze(1) #[batch, 1, feat_size]

        # key, query, value
        v_trans = self.v_lin(v)
        q_trans = self.q_lin(q)
        # mask all padding object/word features
        if APPLY_MASK:
            v_trans = v_trans * v_mask.unsqueeze(2)
            q_trans = q_trans * q_mask.unsqueeze(2)
        # split for different use of purpose
        v_key, v_qry, v_val = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_key, q_qry, q_val = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
        # apply conditioned gate
        gated_v_qry = (1 + q4v_gate) * v_qry
        gated_v_key = (1 + q4v_gate) * v_key
        gated_v_val = (1 + q4v_gate) * v_val
        gated_q_qry = (1 + v4q_gate) * q_qry
        gated_q_key = (1 + v4q_gate) * q_key
        gated_q_val = (1 + v4q_gate) * q_val

        # apply multi-head
        v_key_set = torch.split(gated_v_key, gated_v_key.size(2) // self.num_head, dim=2)
        v_qry_set = torch.split(gated_v_qry, gated_v_qry.size(2) // self.num_head, dim=2)
        v_val_set = torch.split(gated_v_val, gated_v_val.size(2) // self.num_head, dim=2)
        q_key_set = torch.split(gated_q_key, gated_q_key.size(2) // self.num_head, dim=2)
        q_qry_set = torch.split(gated_q_qry, gated_q_qry.size(2) // self.num_head, dim=2)
        q_val_set = torch.split(gated_q_val, gated_q_val.size(2) // self.num_head, dim=2)
        # multi-head
        for i in range(self.num_head):
            v_key_slice, v_qry_slice, v_val_slice = v_key_set[i], v_qry_set[i], v_val_set[i]  #[batch, num_obj, feat_size]
            q_key_slice, q_qry_slice, q_val_slice = q_key_set[i], q_qry_set[i], q_val_set[i]  #[batch, max_len, feat_size]
            # calculate attention
            v2v = (v_qry_slice @ v_key_slice.transpose(1,2)) / ((self.output_size // self.num_head) ** 0.5)
            q2q = (q_qry_slice @ q_key_slice.transpose(1,2)) / ((self.output_size // self.num_head) ** 0.5)

            if APPLY_MASK:
                v2v.masked_fill_(v_mask.unsqueeze(1).expand([batch_size, num_obj, num_obj]) == 0, -float('inf')) 
                q2q.masked_fill_(q_mask.unsqueeze(1).expand([batch_size, max_len, max_len]) == 0, -float('inf')) 
            dyIntraMAF_v2v = F.softmax(v2v, dim=2).unsqueeze(3) #[batch, num_obj, num_obj, 1]
            dyIntraMAF_q2q = F.softmax(q2q, dim=2).unsqueeze(3) #[batch, max_len, max_len, 1]
            # calculate update input
            v_update = (dyIntraMAF_v2v * v_val_slice.unsqueeze(1)).sum(2) if (i==0) else torch.cat((v_update, (dyIntraMAF_v2v * v_val_slice.unsqueeze(1)).sum(2)), dim=2)
            q_update = (dyIntraMAF_q2q * q_val_slice.unsqueeze(1)).sum(2) if (i==0) else torch.cat((q_update, (dyIntraMAF_q2q * q_val_slice.unsqueeze(1)).sum(2)), dim=2)
        # update
        updated_v = self.v_output(v + v_update)
        updated_q = self.q_output(q + q_update)
        return updated_v, updated_q

