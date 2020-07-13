#!/usr/bin/env python3
import re
import os
import argparse
import json
import numpy as np
import pickle
from utils import encode_question
from collections import Counter
import spacy

"""
Preprocessing script for VQA question files.
"""

# according to https://github.com/Cyanogenoid/vqa-counting/blob/master/vqa-v2/data.py
_special_chars = re.compile('[^a-z0-9 ]*')
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))


contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']

def process_punctuation(s):
    if _punctuation.search(s) is None:
        return s
    s = _punctuation_with_a_space.sub('', s)
    if re.search(_comma_strip, s) is not None:
        s = s.replace(',', '')
    s = _punctuation.sub(' ', s)
    s = _period_strip.sub('', s)
    return s.strip()

def find_noun_chunks(s):
    starts = []
    ends = []
    for nounc in s.noun_chunks:
        # starts.append(nounc.start)
        # ends.append(nounc.end)
        tokens = [token.lemma_ for token in nounc]
        if 'what' not in tokens and 'how' not in tokens and 'where' not in tokens and 'there' not in tokens:
            starts.append(nounc.start)
            ends.append(nounc.end)
    
    
    return starts, ends

def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def main(args):
    nlp = spacy.load('en')
    print('Loading data')
    annotations, questions = [], []
    if args.input_annotations_json is not None:
        for f in args.input_annotations_json.split(':'):
            annotations += json.load(open(f, 'r'))['annotations']
    for f in args.input_questions_json.split(':'):
        questions += json.load(open(f, 'r'))['questions']
    print('number of questions: %s' % len(questions))
    question_id_to_str = {q['question_id']: q['question'] for q in questions }

    if args.mode != 'test':
        assert len(annotations) > 0

    # Either create the vocab or load it from disk
    if args.mode == 'train':
        print('Building vocab')
        answer_cnt = {}
        for ann in annotations:
            answers = [_['answer'] for _ in ann['answers']]
            for i,answer in enumerate(answers):
                answer = process_digit_article(process_punctuation(answer))
                answer_cnt[answer] = answer_cnt.get(answer, 0) + 1
                answers[i] = answer
            ann['answers'] = answers # update
        answer_token_to_idx = {}
        for token, cnt in Counter(answer_cnt).most_common(args.answer_top):
            answer_token_to_idx[token] = len(answer_token_to_idx)
        print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

        print("Load glove from %s" % args.glove_pt)
        word2idx, idx2word = pickle.load(open(args.glove_pt, 'rb'))
        question_token_to_idx = word2idx
        # question_token_to_idx['<UNK>'] = len(question_token_to_idx)
        question_token_to_idx['<NULL>'] = len(question_token_to_idx)
        print('Get question_token_to_idx')
        print(len(question_token_to_idx))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
        }
        
        print('Write into %s' % args.vocab_json)
        with open(args.vocab_json, 'w') as f:
            json.dump(vocab, f, indent=4)
    else:
        print('Loading vocab')
        with open(args.vocab_json, 'r') as f:
            vocab = json.load(f)
        for ann in annotations:
            answers = [_['answer'] for _ in ann['answers']]
            for i,answer in enumerate(answers):
                answer = process_punctuation(answer)
                answers[i] = answer
            ann['answers'] = answers # update

    for i,q in question_id_to_str.items():
        question = q.lower()
        question = question.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        question_id_to_str[i] = question


    # Encode all questions
    print('Encoding data')
    questions_encoded = []
    questions_len = []
    image_idxs = []
    answers = []
    questions_mask = []
    noun_chunk_starts = []
    noun_chunk_ends = []
    entity_masks = []
    # max_question_length = 15
    max_entity_length = 4

    if args.mode in {'train', 'val'}:
        for a in annotations:
            question = question_id_to_str[a['question_id']]
            
            doc = nlp(question)
            start, end = find_noun_chunks(doc)
            noun_chunk_starts.append(start[:max_entity_length])
            noun_chunk_ends.append(end[:max_entity_length])
            
            question_tokens = question.split(' ')
            # question_encoded = encode(question_tokens, vocab['question_token_to_idx'], allow_unk=False)
            question_encoded= encode_question(question_tokens, vocab['question_token_to_idx'])
            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))
            image_idxs.append(a['image_id'])

            answer = [] 
            for per_ans in a['answers']:
                if per_ans in vocab['answer_token_to_idx']:
                    i = vocab['answer_token_to_idx'][per_ans]
                    answer.append(i)
            answers.append(answer)

    elif args.mode == 'test':
        for q in questions: # remain the original order to match the question_id
            question = question_id_to_str[q['question_id']] # processed question

            doc = nlp(question)
            start, end = find_noun_chunks(doc)
            noun_chunk_starts.append(start[:max_entity_length])
            noun_chunk_ends.append(end[:max_entity_length])
            
            question_tokens = question.split(' ')
            question_encoded = encode_question(question_tokens, vocab['question_token_to_idx'])
            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))
            image_idxs.append(q['image_id'])
            answers.append([0])

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

    # for ed in noun_chunk_ends:
    #     if len(ed) < max_entity_length:
    #         # qe.append(vocab['question_token_to_idx']['<NULL>'])
    #         padding = [len(qe)] * (max_entity_length - len(ed))
    #         ed += padding

    # for qe in questions_encoded:
    #     questions_mask.append((np.arange(max_question_length)<len(qe)).astype(int))
    #     if len(qe) < max_question_length:
    #         # qe.append(vocab['question_token_to_idx']['<NULL>'])
    #         padding = [vocab['question_token_to_idx']['<NULL>']] * (max_question_length - len(qe))
    #         qe += padding
        
    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    entity_starts = np.asarray(noun_chunk_starts, dtype=np.int32)
    entity_ends = np.asarray(noun_chunk_ends, dtype=np.int32)
    print(entity_starts.shape)
    
    glove_matrix = np.zeros((len(vocab['question_token_to_idx']), 300), dtype=np.float32)
    weight_matrix = np.load(args.glove_array)
    glove_matrix[:-1] = weight_matrix
    
    print(glove_matrix.shape)
    

    print('Writing') 
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'questions_mask': questions_mask,
        'image_idxs': np.asarray(image_idxs),
        'answers': answers,
        'glove': glove_matrix,
        'e_starts': entity_starts,
        'e_ends': entity_ends,
        'e_masks': entity_masks
        }
    with open(args.output_pt, 'wb') as f:
        pickle.dump(obj, f)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_top', default=3120, type=int)
    parser.add_argument('--glove_pt', default='../data/dictionary.pkl', help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--input_questions_json', required=True)
    parser.add_argument('--input_annotations_json', help='not need for test mode')
    parser.add_argument('--output_pt', required=True)
    parser.add_argument('--vocab_json', required=True)
    parser.add_argument('--glove_array', default='../data/glove6b_init_300d.npy')
    parser.add_argument('--mode', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    main(args)
    print('Finish')