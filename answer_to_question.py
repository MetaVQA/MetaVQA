import os
import io
import numpy as np
import cv2
import torch
import json
import spacy
import string
import re

class Handler(object):
    def __init__(self, nlp):
        self.nlp = nlp
        self.type_set = [
            ['how'], # how_handle
            ['why is the', 'why'], # why_handle
            ['which'], # do_handle
            ['who'], # obj_handle
            ['what'], # head_handle
            ['what is on the', 'what is in the', 
             'who is', 'what is', 'what animal is'], # is_handle
            ['where are the', 'where is the'], # where_handle
            ['what is the dog', 'what is the cat', 
             'what is the woman', 'what is the little', 
             'what is the man', 'what are they', 
             'what is the boy', 'what is the girl',
             'what is the baby', 'what are the', 
             'what is the', 'what is the lady'], # what_handle
            ['what color are the', 'what color is the',
             'what color is', 'what color'], # color_handle
            ['what kind of', 'what type of'] # type_handle
        ]
        self.build_dict()

    def build_dict(self):
        self.func_dict = dict()
        func_list = [
            self.how_handle, self.why_handle, self.do_handle,
            self.obj_handle, self.head_handle, self.is_handle,
            self.where_handle, self.what_handle, self.color_handle,
            self.type_handle
        ]
        for i, tp in enumerate(self.type_set):
            temp = {k: func_list[i] for k in tp}
            self.func_dict.update(temp)
        self.type_list = []
        for ls in self.type_set:
            self.type_list += ls

    def handle(self, text, q_type, A):
        return self.func_dict[q_type](text, q_type, A)

    def remove(self, text, s_list):
        text = text.lower()
        for s in s_list:
            if s in text:
                text = text.replace(s, '')
        return text

    def type_handle(self, text, q_type, A):
        text = self.remove(text, [q_type + ' ', '?'])
        cand_list = ['is', 'are']
        for cand in cand_list:
            if (' %s ' % cand) in text:
                text = text.replace((' %s ' % cand), ' ')
                return ('%s %s %s?' % (cand.capitalize(), text, A), 
                        '%s %s not %s?' % (cand.capitalize(), text, A))
        return None

    def color_handle(self, text, q_type, A):
        text = self.remove(text, ['what color ', '?'])
        if q_type in ['what color']:
            return ('%s %s?' % (A.capitalize(), text), 
                    'Non %s %s?' % (A.capitalize(), text))
        else:
            return ('%s %s?' % (text.capitalize(), A), 
                    '%s not %s?' % (text.capitalize(), A))

    def what_handle(self, text, q_type, A):
        if ('ing' in A) and (' doing' in text):
            text = text.replace('doing', A)
            text = self.remove(text, [q_type + ' ', '?'])
            prefix = ' '.join(q_type.split(' ')[1:])
            return ('%s %s?' % (prefix.capitalize(), text),
                    '%s not %s?' % (prefix.capitalize(), text))
        else:
            text = self.remove(text, [q_type + ' ', '?'])
            prefix = ' '.join(q_type.split(' ')[1:])
            return ('%s %s %s?' % (prefix.capitalize(), text, A),
                    '%s %s not %s?' % (prefix.capitalize(), text, A))

    def where_handle(self, text, q_type, A):
        text = self.remove(text, ['where ', '?'])
        if len(A.split(' ')) == 1:
            A = 'on ' + A
        return ('%s %s?' % (text.capitalize(), A),
                '%s not %s?' % (text.capitalize(), A))

    def is_handle(self, text, q_type, A):
        text = self.remove(text, [q_type + ' ', '?'])
        words = q_type.split(' ')
        idx = words.index('is')
        prefix = ' '.join(words[idx+1:])
        if len(prefix):
            return ('Is %s %s %s?' % (A, prefix, text),
                    'Is %s not %s %s?' % (A, prefix, text))
        else:
            return ('Is %s %s?' % (A, text),
                    'Is %s not %s?' % (A, text))

    def head_handle(self, text, q_type, A):
        text = self.remove(text, [q_type + ' ', '?'])
        return ('%s %s?' % (A.capitalize(), text),
                '%s not %s?' % (A.capitalize(), text))

    def obj_handle(self, text, q_type, A):
        doc = self.nlp(text)
        if doc[1].tag_ == 'NN':
            pad = 'the'
        else:
            pad = 'that'
        text = self.remove(text, [q_type + ' ', '?'])
        return ('Is %s %s %s?' % (A, pad, text),
                'Is %s not %s %s?' % (A, pad, text))

    def do_handle(self, text, q_type, A):
        text = self.remove(text, [q_type + ' ', '?'])
        cand_list = ['does', 'is', 'are']
        text = text.split(' ')
        idx = None
        for cand in cand_list:
            if cand in text:
                idx = text.index(cand)
                return ('%s %s %s %s?' % (A.capitalize(), text[idx], ' '.join(text[:idx]), ' '.join(text[idx+1:])),
                        '%s %s not %s %s?' % (A.capitalize(), text[idx], ' '.join(text[:idx]), ' '.join(text[idx+1:])))
        return None

    def why_handle(self, text, q_type, A):
        text = self.remove(text, ['why ', '?'])
        text = text.split(' ')
        if text[0] in ['is', 'are', 'do', 'did', 'does', 'would', 'might']:
            text = text[1:]
        elif 'n\'t' in text[0]:
            text = ['not'] + text[1:]
        return ('Is %s the reason why %s?' % (A, ' '.join(text)),
                'Is %s not the reason why %s?' % (A, ' '.join(text)))

    def how_handle(self, text, q_type, A):
        text = self.remove(text, [q_type + ' ', '?'])
        text = text.split(' ')
        if text[0] in ['do', 'did', 'does', 'is', 'are', 'will', 'would']:
            return ('%s %s?' % (' '.join(text).capitalize(), A),
                    '%s not %s?' % (' '.join(text).capitalize(), A))
        else:
            return ('%s %s %s?' % (' '.join(text[1:]).capitalize(), A, text[0]),
                    '%s not %s %s?' % (' '.join(text[1:]).capitalize(), A, text[0]))


class Answer2Question(object):
    def __init__(self, Q_path, A_path):
        with open(Q_path, 'r') as f:
            content = json.load(f)
            self.Q = content['questions']
        with open(A_path, 'r') as f:
            content = json.load(f)
            self.A = content['annotations']
        
        assert len(self.Q) == len(self.A)
 
        self.nlp = spacy.load('en_core_web_sm')
        self.table = str.maketrans(
            dict.fromkeys(
            string.punctuation.replace(':', '')
            )
        )
        self.handler = Handler(nlp=self.nlp)
        self.build_pair()
        self.build_pool()

        #self.just_for_test()

    def just_for_test(self):
        pair = []
        for p in self.pair:
            if int(p['image_id']) >= 20000 and int(p['image_id']) <= 20049:
                pair.append(p)
        self.pair = pair

    def build_pair(self):
        self.pair = []
        for i in range(len(self.Q)):
            q = self.Q[i]
            a = self.A[i]
            dic = {
                'Q': ' '.join(re.split(r'[\s]', q['question'])),
                'A': a['multiple_choice_answer'],
                'question_type': a['question_type'],
                'answer_type': a['answer_type'],
                'image_id': a['image_id']
            }
            self.pair.append(dic)

    def build_pool(self):
        print('Preparing `OR` Pool...')
        self.OR_pool = dict()
        for p in self.pair:
            if self.is_or(p):
                (A, B) = self.split(p['Q'], p['question_type'])
                if len(A) and len(B):
                    self.OR_pool = self.pool_add(self.OR_pool, p['question_type'], [A, B])
            elif self.is_yn(p):
                A = self.preprocess(p['Q'], p['question_type'])
                if len(A):
                    self.OR_pool = self.pool_add(self.OR_pool, p['question_type'], [A])
        self.OR_pool = self.pool_dedup(self.OR_pool)
        
        print('Preparing `OTHER` Pool...')
        self.OTHER_pool = dict()
        for p in self.pair:
            if self.is_other(p):
                self.OTHER_pool = self.pool_add(self.OTHER_pool, p['question_type'], [p['A']])
        self.OTHER_pool = self.pool_dedup(self.OTHER_pool)

    def pool_add(self, pool, k:str, v:list):
        if k in pool.keys():
            pool[k] += v
        else:
            pool[k] = v
        return pool

    def pool_dedup(self, pool):
        for k in pool.keys():
            pool[k] = list(set(pool[k]))
        return pool

    def is_yn(self, p):
        return (p['question_type'] != 'none of the above') and (p['answer_type'] == 'yes/no')

    def is_or(self, p):
        return (p['question_type'] != 'none of the above') and (' or ' in p['Q']) \
               and (p['question_type'].split(' ')[0] in ['is', 'are', 'do', 'did', 'does', 'will'])

    def is_other(self, p):
        return (p['question_type'] != 'none of the above') and (p['question_type'] in self.handler.type_list)
    
    def preprocess(self, text, q_type):
        # FROM `q_type A?` / `q_type A or B?`
        text = text.translate(self.table)
        text = text[0].lower() + text[1:]
        text  = text.replace(q_type + ' ', '')
        # TO `A` / `A or B`
        return text

    def pad(self, A, B):
        doc_A = self.nlp(A)
        doc_B = self.nlp(B)
        if len(doc_A) == len(doc_B):
            TAG = True
            for i in range(len(doc_A)):
                if doc_A[i].tag_ != doc_B[i].tag_:
                    TAG = False
                    break
            if TAG:
                return (A, B)

        A = A.split(' ')
        B = B.split(' ')
        s_A, e_A = len(doc_A) - 1, len(doc_A)
        s_B, e_B = 0, 1

        while (s_A >= 0) and (e_B <= len(doc_B)):
            length = e_A - s_A
            TAG = True
            for i in range(length):
                if doc_A[s_A + i].tag_ != doc_B[s_B + i].tag_:
                    TAG = False
                    break
            if TAG:
                break
            else:
                s_A -= 1
                e_B += 1
        if TAG:
            final_A, final_B = ' '.join(A), ' '.join(B)
            if s_A > 0:
                final_B = ' '.join(A[0:s_A]) + ' ' + final_B
            if e_B < len(B):
                final_A = final_A + ' ' + ' '.join(B[e_B:])
            return (final_A, final_B)
        return '', ''

    def split(self, text, q_type):
        text = self.preprocess(text, q_type)
        if len(text.split(' or ')) == 2:
            [A, B] = text.split(' or ')
            (A, B) = self.pad(A, B)
            return (A, B)
        return '', ''

    def fromOR(self, text, q_type):
        (A, B) = self.split(text, q_type)
        if len(A) and len(B):
            q_type = q_type[0].upper() + q_type[1:]
            return (
                ('%s %s or %s?' % (q_type, A, B)),
                (('%s %s?') % (q_type, A)),
                (('%s %s?') % (q_type, B))
            )
        return None

    def toOR(self, text, q_type):
        A = self.preprocess(text, q_type)
        B_idx = np.random.randint(len(self.OR_pool[q_type]))
        B = self.OR_pool[q_type][B_idx]
        while B == A:
            B_idx = np.random.randint(len(self.OR_pool[q_type]))
            B = self.OR_pool[q_type][B_idx]
        q_type = q_type[0].upper() + q_type[1:]
        return (
            ('%s %s or %s?' % (q_type, A, B)), 
            (('%s %s?') % (q_type, A)),
            (('%s %s?') % (q_type, B))
        )

    def fromAnswer(self, Q, q_type, topk_A:list):
        # LABLE should be (True, False, False, True)
        real_A = topk_A[0]
        fake_A_idx = np.random.randint(len(self.OTHER_pool[q_type]))
        fake_A = self.OTHER_pool[q_type][fake_A_idx]
        while fake_A in topk_A:
            fake_A_idx = np.random.randint(len(self.OTHER_pool[q_type]))
            fake_A = self.OTHER_pool[q_type][fake_A_idx]
        real = self.handler.handle(Q, q_type, real_A)
        fake = self.handler.handle(Q, q_type, fake_A)
        if real and fake:
            return real + fake
        return None
            

if __name__ == '__main__':
    Q_path = '/root/data/VQA/questions/abstract/OpenEnded_abstract_v002_val2015_questions.json'
    A_path = '/root/data/VQA/questions/abstract/abstract_v002_val2015_annotations.json'
    a2q = Answer2Question(Q_path, A_path)
    print(a2q.OTHER_pool['what kind of'])
    for _ in range(10):
        idx = np.random.randint(len(a2q.pair))
        p = a2q.pair[idx]
        print(_)
        print(p['Q'])
        print(p['A'])
        if a2q.is_or(p):
            print(a2q.fromOR(p['Q'], p['question_type']))
        elif a2q.is_yn(p):
            print(a2q.toOR(p['Q'], p['question_type']))
        elif a2q.is_other(p):
            print(a2q.fromAnswer(p['Q'], p['question_type'], [p['A']]))