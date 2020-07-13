# The scrpit is modified based on https://github.com/facebookresearch/clevr-iep/blob/master/scripts/preprocess_questions.py

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse

import json
import os

import numpy as np
import pickle

import programs
from utils import tokenize, encode, build_vocab

import spacy

"""
Preprocessing script for CLEVR question files.
"""


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='prefix',
                    choices=['chain', 'prefix', 'postfix'])
parser.add_argument('--input_vocab_json', default='')
parser.add_argument('--expand_vocab', default=0, type=int)
parser.add_argument('--unk_threshold', default=1, type=int)
parser.add_argument('--encode_unk', default=0, type=int)

parser.add_argument('--input_questions_json', required=True)
parser.add_argument('--output_pt_file', required=True)
parser.add_argument('--output_vocab_json', default='')

def find_noun_chunks(s):
  starts = []
  ends = []
  for nounc in s.noun_chunks:
    tokens = [token.lemma_ for token in nounc]
    if 'cylinder' in tokens or 'object' in tokens or 'cube' in tokens or 'sphere' in tokens or 'thing' in tokens:
      starts.append(nounc.start)
      ends.append(nounc.end)

  return starts, ends

  ####################################################################################


def main(args):

  nlp = spacy.load('en')
  print('Loading data')
  with open(args.input_questions_json, 'r') as f:
    questions = json.load(f)['questions']


  # Either create the vocab or load it from disk
  if args.input_vocab_json == '' or args.expand_vocab == 1:
    print('Building vocab')
    if 'answer' in questions[0]:
      answer_token_to_idx = build_vocab(
        (q['answer'] for q in questions)
      )
    question_token_to_idx = build_vocab(
      (q['question'] for q in questions),
      min_token_count=args.unk_threshold,
      punct_to_keep=[';', ','], punct_to_remove=['?', '.'],
      add_special=True
    )
    # all_program_strs = []
    # for q in questions:
    #   if 'program' not in q: continue
    #   program_str = program_to_strs(q['program'], args.mode)[0]
    #   if program_str is not None:
    #     all_program_strs.append(program_str)
    # program_token_to_idx = build_vocab(all_program_strs, add_special=True)
    vocab = {
      'question_token_to_idx': question_token_to_idx,
      # 'program_token_to_idx': program_token_to_idx,
      'answer_token_to_idx': answer_token_to_idx, # no special tokens
    }

  if args.input_vocab_json != '':
    print('Loading vocab')
    if args.expand_vocab == 1:
      new_vocab = vocab
    with open(args.input_vocab_json, 'r') as f:
      vocab = json.load(f)
    if args.expand_vocab == 1:
      num_new_words = 0
      for word in new_vocab['question_token_to_idx']:
        if word not in vocab['question_token_to_idx']:
          print('Found new word %s' % word)
          idx = len(vocab['question_token_to_idx'])
          vocab['question_token_to_idx'][word] = idx
          num_new_words += 1
      print('Found %d new words' % num_new_words)

  if args.output_vocab_json != '':
    with open(args.output_vocab_json, 'w') as f:
      json.dump(vocab, f, indent=4)

  # Encode all questions and entities
  print('Encoding data')
  questions_encoded = []
  orig_idxs = []
  image_idxs = []
  answers = []
  questions_len = []
  questions_mask = []
  noun_chunk_starts = []
  noun_chunk_ends = []
  entity_masks = []
  max_entity_length = 5

  for orig_idx, q in enumerate(questions):
    question = q['question'].replace('?', '').replace('.', '').replace(';', ' ;').replace(',', ' ,')

    doc = nlp(question)
    start, end = find_noun_chunks(doc)
    noun_chunk_starts.append(start[:max_entity_length])
    noun_chunk_ends.append(end[:max_entity_length])

    orig_idxs.append(orig_idx)
    image_idxs.append(q['image_index'])
    question_tokens = tokenize(question)

    question_encoded = encode(question_tokens,
                         vocab['question_token_to_idx'],
                         allow_unk=args.encode_unk == 1)

    questions_encoded.append(question_encoded)
    questions_len.append(len(question_encoded))

    if 'answer' in q:
      answers.append(vocab['answer_token_to_idx'][q['answer']])
    else:
      answers.append(-1)

  # Pad encoded questions and entities
  max_question_length = max(len(x) for x in questions_encoded)


  for st, ed, qe in zip(noun_chunk_starts, noun_chunk_ends, questions_encoded):
    entity_masks.append((np.arange(max_entity_length)<len(st)).astype(int))
    if len(st) < max_entity_length:
      # qe.append(vocab['question_token_to_idx']['<NULL>'])
      padding = [len(qe)-1] * (max_entity_length - len(st))
      st += padding

    if len(ed) < max_entity_length:
      # qe.append(vocab['question_token_to_idx']['<NULL>'])
      padding = [len(qe)] * (max_entity_length - len(ed))
      ed += padding

    questions_mask.append((np.arange(max_question_length)<len(qe)).astype(int))
    if len(qe) < max_question_length:
      # qe.append(vocab['question_token_to_idx']['<NULL>'])
      padding = [vocab['question_token_to_idx']['<NULL>']] * (max_question_length - len(qe))
      qe += padding
  questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
  questions_len = np.asarray(questions_len, dtype=np.int32)
  print(questions_encoded.shape)
 
  entity_starts = np.asarray(noun_chunk_starts, dtype=np.int32)
  entity_ends = np.asarray(noun_chunk_ends, dtype=np.int32)
  print(entity_starts.shape)

  print('Writing')
  obj = {
          'questions': questions_encoded,
          'image_idxs': np.asarray(image_idxs),
          'orig_idxs': np.asarray(orig_idxs),
          # 'programs': programs_encoded,
          # 'program_inputs': program_inputs_encoded,
          'answers': answers,
          'questions_len': questions_len,
          'questions_mask': questions_mask,
          'e_starts': entity_starts,
          'e_ends': entity_ends,
          'e_masks': entity_masks
          }
  with open(args.output_pt_file, 'wb') as f:
    pickle.dump(obj, f)
 



if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
