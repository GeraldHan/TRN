
"""
This code is based on XNM-Net
"""

import numpy as np
import json
import pickle
import torch
import math
import h5py
from torch.utils.data import Dataset, DataLoader, dataloader
# from torch.utils.data.dataloader import default_collate
from IPython import embed


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        # vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
    return vocab

def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return dataloader.default_collate(batch)

class VQADataset(Dataset):

    def __init__(self, answers, questions, questions_len, q_image_indices, questions_mask, questions_id,
                    entity_starts, entity_ends, entity_masks,
                    feature_h5, feat_coco_id_to_index, num_answer, num_token, use_spatial):
        # convert data to tensor
        self.all_answers = answers
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))
        self.all_q_image_idxs = torch.LongTensor(np.asarray(q_image_indices))
        self.all_questions_mask = torch.FloatTensor(np.asarray(questions_mask))
        self.all_entity_starts = torch.LongTensor(np.asarray(entity_starts))
        self.all_entity_ends = torch.LongTensor(np.asarray(entity_ends))
        self.all_entity_masks = torch.LongTensor(np.asarray(entity_masks))
        self.all_questions_id = torch.LongTensor(np.asarray(questions_id))

        self.feature_h5 = feature_h5
        self.feat_coco_id_to_index = feat_coco_id_to_index
        self.num_answer = num_answer
        self.use_spatial = use_spatial
        self.num_token = num_token


    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        if answer is not None:
            _answer = torch.zeros(self.num_answer)
            for i in answer:
                _answer[i] += 1
            answer = _answer
        question = self.all_questions[index]
        questions_id = self.all_questions_id[index]
        question_len = self.all_questions_len[index]
        question_mask = self.all_questions_mask[index]

        entity_start = self.all_entity_starts[index]
        entity_end = self.all_entity_ends[index]
        entity_mask = self.all_entity_masks[index]

        image_idx = self.all_q_image_idxs[index].item() # coco_id
        # fetch vision features
        index = self.feat_coco_id_to_index[image_idx]
        with h5py.File(self.feature_h5, 'r') as f:
            vision_feat = f['features'][index]
            boxes = f['boxes'][index]
            w = f['widths'][index]
            h = f['heights'][index]
        
        spatial_feat = np.zeros((4, len(boxes[0])))
        spatial_feat[0, :] = boxes[0, :] * 2 / w - 1 # x1
        spatial_feat[1, :] = boxes[1, :] * 2 / h - 1 # y1
        spatial_feat[2, :] = boxes[2, :] * 2 / w - 1 # x2
        spatial_feat[3, :] = boxes[3, :] * 2 / h - 1 # y2
        # spatial_feat[4, :] = (spatial_feat[2, :]-spatial_feat[0, :]) * (spatial_feat[3, :]-spatial_feat[1, :])

        if self.use_spatial:
            vision_feat = np.concatenate((vision_feat, spatial_feat), axis=0)
        vision_feat = torch.from_numpy(vision_feat).float().transpose(-1,-2)
        #########
        num_feat = boxes.shape[1]
        relation_mask = np.zeros((num_feat,num_feat))
        for i in range(num_feat):
            for j in range(i, num_feat):
                # if there is no overlap between two bounding box
                if boxes[0,i]>boxes[2,j] or boxes[0,j]>boxes[2,i] or boxes[1,i]>boxes[3,j] or boxes[1,j]>boxes[3,i]:
                    pass
                else:
                    relation_mask[i,j] = relation_mask[j,i] = 1
        relation_mask = torch.from_numpy(relation_mask).byte()

        q_onehot = torch.zeros(question_len, self.num_token).scatter_(1, question[:question_len].unsqueeze(-1), 1)
        entity_label = []
        for i in range(4):
            ee = q_onehot[entity_start[i] : entity_end[i]].sum(0)
            entity_label.append(ee)
        
        entity_label = torch.stack(entity_label)

        return (image_idx, questions_id, answer, question, vision_feat, entity_start, entity_end, entity_mask, entity_label, question_mask, relation_mask, question_len)

    def __len__(self):
        return len(self.all_questions)



class VQADataLoader(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            questions_len = obj['questions_len']
            q_image_indices = obj['image_idxs']
            answers = obj['answers']
            glove_matrix = obj['glove']
            questions_mask = obj['questions_mask']
            entity_starts = obj['e_starts']
            entity_ends = obj['e_ends']
            entity_masks = obj['e_masks']
            questions_id = obj['questions_id']
        
        use_spatial = kwargs.pop('spatial')
        with h5py.File(kwargs['feature_h5'], 'r') as features_file:
            coco_ids = features_file['ids'][()]
        feat_coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        self.feature_h5 = kwargs.pop('feature_h5')
        self.dataset = VQADataset(answers, questions, questions_len, q_image_indices, questions_mask, questions_id,
                entity_starts, entity_ends, entity_masks,
                self.feature_h5, feat_coco_id_to_index, len(vocab['answer_token_to_idx']), len(vocab['question_token_to_idx']), use_spatial)
        
        self.vocab = vocab
        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix

        kwargs['collate_fn'] = collate_fn
        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

