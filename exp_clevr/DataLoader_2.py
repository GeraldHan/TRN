
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

class CLEVRDataset(Dataset):

    def __init__(self, answers, questions, questions_len, image_indices, questions_mask,
                    entity_starts, entity_ends, entity_masks, orig_indices,
                    feature_h5, num_answer, num_token):
        # convert data to tensor
        self.all_answers = answers
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))
        self.all_image_idxs = torch.LongTensor(np.asarray(image_indices))
        self.all_orig_idxs = torch.LongTensor(np.asarray(orig_indices))
        self.all_questions_mask = torch.FloatTensor(np.asarray(questions_mask))
        self.all_entity_starts = torch.LongTensor(np.asarray(entity_starts))
        self.all_entity_ends = torch.LongTensor(np.asarray(entity_ends))
        self.all_entity_masks = torch.LongTensor(np.asarray(entity_masks))

        self.feature_h5 = feature_h5
        self.num_answer = num_answer
        self.num_token = num_token

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        question = self.all_questions[index]
        question_len = self.all_questions_len[index]
        question_mask = self.all_questions_mask[index]

        entity_start = self.all_entity_starts[index]
        entity_end = self.all_entity_ends[index]
        entity_mask = self.all_entity_masks[index]

        image_idx = self.all_image_idxs[index].item() #image_id
        orig_idx = self.all_orig_idxs[index].item()

        with h5py.File(self.feature_h5, 'r') as f:
            vision_feat = f['features'][image_idx]

        # if self.use_spatial:
        #     vision_feat = np.concatenate((vision_feat, spatial_feat), axis=0)
        
        #########
        num_obj, vec_size = vision_feat.shape
        edge_vec = np.zeros((num_obj, num_obj, vec_size))
        relation_mask = np.eye(num_obj)
        for i in range(num_obj):
            for j in range(num_obj):
                if sum(vision_feat[i]) != 0 and sum(vision_feat[j]) != 0:
                    relation_mask[i, j] = 1
                    edge_vec[i,j] = vision_feat[i] - vision_feat[j]
        edge_vec = torch.from_numpy(edge_vec).float()
        relation_mask = torch.from_numpy(relation_mask).byte()
        vision_feat = torch.from_numpy(vision_feat).float()
        
        q_onehot = torch.zeros(question_len, self.num_token).scatter_(1, question[:question_len].unsqueeze(-1), 1)
        entity_label = []
        for i in range(5):
            ee = q_onehot[entity_start[i] : entity_end[i]].sum(0)
            entity_label.append(ee)
        
        entity_label = torch.stack(entity_label)

        return (orig_idx, image_idx, answer, question, vision_feat, entity_start, entity_end, entity_mask, entity_label,
         edge_vec, relation_mask, question_mask, question_len)

    def __len__(self):
        return len(self.all_questions)



class CLEVRDataLoader(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)

        feature_h5 = kwargs.pop('feature_h5')
        print('loading features from %s' % (feature_h5))

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            questions_len = obj['questions_len']
            image_indices = obj['image_idxs']
            answers = obj['answers']
            questions_mask = obj['questions_mask']
            entity_starts = obj['e_starts']
            entity_ends = obj['e_ends']
            entity_masks = obj['e_masks']
            orig_indices = obj['orig_idxs']
        
        # with h5py.File(kwargs['feature_h5'], 'r') as features_file:
        #     coco_ids = features_file['ids'][()]
        # feat_coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}

  ##feature: 0:3 shape, 3:5 size, 5:7 materials, 7:15 colors, 15:18 position
        
        self.dataset = CLEVRDataset(answers, questions, questions_len, image_indices, questions_mask,
                entity_starts, entity_ends, entity_masks, orig_indices,
                feature_h5, len(vocab['answer_token_to_idx']), len(vocab['question_token_to_idx']))
        
        self.vocab = vocab
        self.batch_size = kwargs['batch_size']

        kwargs['collate_fn'] = collate_fn
        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

