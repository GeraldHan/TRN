import os,sys
# os.environ['CUDA_VISIBLE_DEVICES'] = "7"
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # to import shared utils
import torch
from tqdm import tqdm
import argparse
import numpy as np
import os
import json
from IPython import embed
from collections import defaultdict

from DataLoader_2 import VQADataLoader
from model_* import Net
from misc import todevice


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    _, predicted_index = predicted.max(dim=1, keepdim=True)
    agreeing = true.gather(dim=1, index=predicted_index)
    '''
    Acc needs to be averaged over all 10 choose 9 subsets of human answers.
    While we could just use a loop, surely this can be done more efficiently (and indeed, it can).
    There are two cases for the 1 chosen answer to be discarded:
    (1) the discarded answer is not the predicted answer => acc stays the same
    (2) the discarded answer is the predicted answer => we have to subtract 1 from the number of agreeing answers
    
    There are (10 - num_agreeing_answers) of case 1 and num_agreeing_answers of case 2, thus
    acc = ((10 - agreeing) * min( agreeing      / 3, 1)
           +     agreeing  * min((agreeing - 1) / 3, 1)) / 10
    
    Let's do some more simplification:
    if num_agreeing_answers == 0:
        acc = 0  since the case 1 min term becomes 0 and case 2 weighting term is 0
    if num_agreeing_answers >= 4:
        acc = 1  since the min term in both cases is always 1
    The only cases left are for 1, 2, and 3 agreeing answers.
    In all of those cases, (agreeing - 1) / 3  <  agreeing / 3  <=  1, so we can get rid of all the mins.
    By moving num_agreeing_answers from both cases outside the sum we get:
        acc = agreeing * ((10 - agreeing) + (agreeing - 1)) / 3 / 10
    which we can simplify to:
        acc = agreeing * 0.3
    Finally, we can combine all cases together with:
        min(agreeing * 0.3, 1)
    '''
    return (agreeing * 0.3).clamp(max=1)


def validate(model, data, device):
    count, correct = 0, 0
    model.eval()
    print('validate...')
    total_acc, count = 0, 0
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            coco_ids, q_ids, answers, *batch_input = [todevice(x, device) for x in batch]
            batch_input = [x.detach() for x in batch_input]
            logits, loss_time = model(*batch_input)
            acc = batch_accuracy(logits, answers)
            total_acc += acc.sum().data.item()
            count += answers.size(0)
    acc = total_acc / count
    return acc

def test(model, data, device):
    model.eval()
    results = []
    question_ids = []
    for batch in tqdm(data, total=len(data)):
        coco_ids, q_ids, answers, *batch_input = [todevice(x, device) for x in batch]
        logits, loss_time = model(*batch_input)
        predicts = torch.max(logits, dim=1)[1]
        for predict in predicts:
            results.append(data.vocab['answer_idx_to_token'][predict.item()])
        for q_id in q_ids:
            question_ids.append(q_id.item())
    return results, question_ids

def val_with_acc(model, data, device):
    model.eval()
    question_ids = []
    accs = []
    total_acc, count = 0, 0
    for batch in tqdm(data, total=len(data)):
        coco_ids, q_ids, answers, *batch_input = [todevice(x, device) for x in batch]
        logits, loss_time = model(*batch_input)
        batch_acc = batch_accuracy(logits, answers)
        predicts = torch.max(logits, dim=1)[1]
        for q_id in q_ids:
            question_ids.append(q_id.item())
        for acc in batch_acc:
            accs.append(acc.item())
    return accs, question_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', choices=['val', 'test'])
    # input
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--input_dir', default='process', required=True)
    parser.add_argument('--val_question_pt', default='new_2/val_questions.pt')
    parser.add_argument('--test_question_pt', default='new_2/test_std_questions.pt')
    parser.add_argument('--vocab_json', default='new_2/vocab.json')
    parser.add_argument('--val_feature_h5', default='genome-trainval_36.h5')
    parser.add_argument('--test_feature_h5', default='genome-test.h5')
    parser.add_argument('--output_file', help='used only in test mode')
    parser.add_argument('--pair_question_json', default='data/v2_mscoco_val2014_complementary_pairs.json')
    parser.add_argument('--val_ann_json', default='data/v2_mscoco_val2014_annotations.json')
    parser.add_argument('--val_question_json', default='data/v2_OpenEnded_mscoco_val2014_questions.json')
    parser.add_argument('--test_question_json', default='data/v2_OpenEnded_mscoco_test-dev2015_questions.json', help='path to v2_OpenEnded_mscoco_test2015_questions.json, used only in test mode')
    args = parser.parse_args()

    args.vocab_json = os.path.join(args.input_dir, args.vocab_json)
    args.val_question_pt = os.path.join(args.input_dir, args.val_question_pt)
    args.test_question_pt = os.path.join(args.input_dir, args.test_question_pt)
    args.val_feature_h5 = os.path.join(args.input_dir, args.val_feature_h5)
    args.test_feature_h5 = os.path.join(args.input_dir, args.test_feature_h5)
    
    device = 'cuda'
    loaded = torch.load(args.ckpt, map_location={'cuda:0': 'cpu'})
    model_kwargs = loaded['model_kwargs']
    if args.mode == 'val':
        val_loader_kwargs = {
            'question_pt': args.val_question_pt,
            'vocab_json': args.vocab_json,
            'feature_h5': args.val_feature_h5,
            'batch_size': 64,
            'spatial': model_kwargs['spatial'],
            'num_workers': 2,
            'shuffle': False
        }

        with open(args.pair_question_json, 'r') as fd:
            pairs = json.load(fd)
        with open(args.val_question_json, 'r') as fd:
            q_json = json.load(fd)
        with open(args.val_ann_json, 'r') as fd:
            a_json = json.load(fd)

        question_list = q_json['questions']
        questions = [q['question'] for q in question_list]
        answer_list = a_json['annotations']
        # categories = [a['answer_type'] for a in answer_list]  # {'yes/no', 'other', 'number'}

        val_loader = VQADataLoader(**val_loader_kwargs)
        model_kwargs.update({'vocab': val_loader.vocab, 'device': device})
        model = Net(**model_kwargs).to(device)
        model.load_state_dict(loaded['state_dict'])
        accs, question_ids = val_with_acc(model, val_loader, device)
        # accs = list(torch.cat(accs, dim=0))
        assert len(accs) == len(questions)

        
        accept_condition = {
            'number': (lambda x: id_to_cat[x] == 'number'),
            'yes/no': (lambda x: id_to_cat[x] == 'yes/no'),
            'other': (lambda x: id_to_cat[x] == 'other'),
            'count': (lambda x: id_to_question[x].lower().startswith('how many')),
            'all': (lambda x: True),
        }
        statistics = defaultdict(list)
        accs = map(lambda x: x, accs)
        id_to_acc = dict(zip(question_ids, accs))
        # id_to_cat = dict(zip(question_ids, categories))
        id_to_cat= {a['question_id']:a['answer_type'] for a in answer_list}
        id_to_question = {q['question_id']: q['question'] for q in question_list }
        
        for name, f in accept_condition.items():
            for on_pairs in [False, True]:
                acc = []
                if on_pairs:
                    for a, b in pairs:
                        if not (f(a) and f(b)):
                            continue
                        if id_to_acc[a] == id_to_acc[b] == 1:
                            acc.append(1)
                        else:
                            acc.append(0)
                else:
                    for x in question_ids:
                        if not f(x):
                            continue
                        acc.append(id_to_acc[x])
                acc = np.mean(acc)
                statistics[name, 'pair' if on_pairs else 'single'].append(acc)

        for (name, pairness), accs in statistics.items():
            mean = np.mean(accs)
            std = np.std(accs, ddof=1)
            print('{} ({})\t: {:.2f}% +- {}'.format(name, pairness, 100 * mean, 100 * std))


    elif args.mode == 'test':
        assert args.output_file and os.path.exists(args.test_question_json)
        test_loader_kwargs = {
            'question_pt': args.test_question_pt,
            'vocab_json': args.vocab_json,
            'feature_h5': args.test_feature_h5,
            'batch_size': 64,
            'spatial': model_kwargs['spatial'],
            'num_workers': 2,
            'shuffle': False
        }
        test_loader = VQADataLoader(**test_loader_kwargs)
        model_kwargs.update({'vocab': test_loader.vocab, 'device': device})
        model = Net(**model_kwargs).to(device)
        model.load_state_dict(loaded['state_dict'])
        results, question_ids = test(model, test_loader, device)
        questions = json.load(open(args.test_question_json))['questions']
        print(len(results))
        print(len(questions))
        assert len(results) == len(questions)
        assert len(results) == len(question_ids)

        question_id_to_answer = {q_id: r for r, q_id in zip(results, question_ids)}
        for i,a in question_id_to_answer.items():
            question_id_to_answer[i] = a
        
        results = [{'answer': question_id_to_answer[q['question_id']], "question_id":q['question_id']} for q in questions]


        with open(args.output_file, 'w') as f:
            json.dump(results, f)
        print('write into %s' % args.output_file)
