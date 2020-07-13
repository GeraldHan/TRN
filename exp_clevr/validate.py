import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # to import shared utils

import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import numpy as np
import json
from DataLoader import CLEVRDataLoader
# from utils.generate_programs import generate_single_program, load_program_generator
# from utils.misc import convert_david_program_to_mine, invert_dict, todevice
from misc import todevice
from model import Net

map_program_to_cat = {
        'count': 'count',
        'equal': 'compare attribute',
        'equal_integer': 'compare number',
        'exist': 'exist',
        'greater_than': 'compare number',
        'less_than': 'compare number',
        'query': 'query',
        }


def validate(model, data, device, detail=False):
    count, correct = 0, 0
    beta=1.
    model.eval()
    print('validate...')
    for batch in tqdm(data, total=len(data)):
        orig_idx, image_idx, answers, *batch_input = [todevice(x, device) for x in batch]
        logits, loss_t = model(*batch_input)
        predicts = logits.max(1)[1]
        correct += torch.eq(predicts, answers).long().sum().item()
        count += answers.size(0)

    acc = correct / count
    return acc





# def test_with_david_generated_program(model, data, device, pretrained_dir):
#     program_generator = load_program_generator(os.path.join(pretrained_dir, 'program_generator.pt')).to(device)
#     david_vocab = json.load(open(os.path.join(pretrained_dir, 'david_vocab.json')))
#     david_vocab['program_idx_to_token'] = invert_dict(david_vocab['program_token_to_idx'])
#     results = []
#     model.eval()
#     for batch in tqdm(data, total=len(data)):
#         _, questions, gt_programs, gt_program_inputs, features, edge_vectors = [todevice(x, device) for x in batch]
#         programs, program_inputs = [], []
#         # generate program using david model for each question
#         for i in range(questions.size(0)):
#             question_str = []
#             for j in range(questions.size(1)):
#                 word = data.vocab['question_idx_to_token'][questions[i,j].item()]
#                 if word == '<START>': continue
#                 if word == '<END>': break
#                 question_str.append(word)
#             question_str = ' '.join(question_str) # question string
#             david_program = generate_single_program(question_str, program_generator, david_vocab, device)
#             david_program = [david_vocab['program_idx_to_token'][i.item()] for i in david_program.squeeze()]
#             # convert david program to ours. return two index lists
#             program, program_input = convert_david_program_to_mine(david_program, data.vocab)
#             programs.append(program)
#             program_inputs.append(program_input)
#         # padding
#         max_len = max(len(p) for p in programs)
#         for i in range(len(programs)):
#             while len(programs[i]) < max_len:
#                 programs[i].append(vocab['program_token_to_idx']['<NULL>'])
#                 program_inputs[i].append(vocab['question_token_to_idx']['<NULL>'])
#         # to tensor
#         programs = torch.LongTensor(programs).to(device)
#         program_inputs = torch.LongTensor(program_inputs).to(device)

#         logits = model(programs, program_inputs, features, edge_vectors)
#         predicts = logits.max(1)[1]
#         for predict in predicts: # note questions must not shuffle!
#             results.append(data.vocab['answer_idx_to_token'][predict.item()])
#     return results


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ckpt', required=True)
#     parser.add_argument('--input_dir', required=True)
#     parser.add_argument('--val_question_pt', default='val_questions.pt')
#     parser.add_argument('--val_feature_pt', default='val_features.pt')
#     parser.add_argument('--test_question_pt', default='test_questions.pt')
#     parser.add_argument('--test_feature_pt', default='test_features.pt')
#     parser.add_argument('--vocab_json', default='vocab.json')
#     parser.add_argument('--pretrained_dir', default='../pretrained', help='shoud contain pretrained program generator and david vocab')
#     # control parameters
#     parser.add_argument('--mode', default='val', choices=['val', 'test'])
#     parser.add_argument('--program', default='gt', choices=['gt', 'david'])
#     parser.add_argument('--output_file', help='used in test mode')
#     args = parser.parse_args()

#     args.vocab_json = os.path.join(args.input_dir, args.vocab_json)
#     args.val_question_pt = os.path.join(args.input_dir, args.val_question_pt)
#     args.val_feature_pt = os.path.join(args.input_dir, args.val_feature_pt)
#     args.test_question_pt = os.path.join(args.input_dir, args.test_question_pt)
#     args.test_feature_pt = os.path.join(args.input_dir, args.test_feature_pt)
    
#     device = 'cuda'
#     loaded = torch.load(args.ckpt, map_location={'cuda:0': 'cpu'})
#     model_kwargs = loaded['model_kwargs']

#     if args.mode == 'val':
#         val_loader_kwargs = {
#             'question_pt': args.val_question_pt,
#             'feature_pt': args.val_feature_pt,
#             'vocab_json': args.vocab_json,
#             'batch_size': 256,
#             'shuffle': False,
#         }
#         val_loader = CLEVRDataLoader(**val_loader_kwargs)
#         model_kwargs.update({'vocab': val_loader.vocab})
#     elif args.mode == 'test':
#         test_loader_kwargs = {
#             'question_pt': args.test_question_pt,
#             'feature_pt': args.test_feature_pt,
#             'vocab_json': args.vocab_json,
#             'batch_size': 256,
#             'shuffle': False,
#         }
#         test_loader = ClevrDataLoader(**test_loader_kwargs)
#         model_kwargs.update({'vocab': test_loader.vocab})

#     model = Net(**model_kwargs).to(device)
#     model.load_state_dict(loaded['state_dict'])
#     num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(' ~~~~~~~~~~ num parameters: %d ~~~~~~~~~~~~~' % num_parameters)

#     if args.mode == 'val':
#         if args.program == 'gt':
#             print('validate with **ground truth** program')
#             val_acc, val_details = validate(model, val_loader, device, detail=True)
#         elif args.program == 'david':
#             print('validate with **david predicted** program')
#             val_acc, val_details = validate_with_david_generated_program(model, val_loader, device, args.pretrained_dir)
#         print("Validate acc: %.4f" % val_acc)
#         print(json.dumps(val_details, indent=2))
#     elif args.mode == 'test':
#         assert args.output_file, 'output_file must be given in test mode'
#         print('test with david predicted program')
#         results = test_with_david_generated_program(model, test_loader, device, args.pretrained_dir)
#         with open(args.output_file, 'w') as f:
#             for res in results:
#                 f.write(res+'\n') # one answer per line, in the same order as the questions file

