import os
import io
import cv2
import json
import time
from PIL import Image
import numpy as np
import progressbar

import torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from answer_to_question import Answer2Question

'''
Utils
'''

class Record(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, correct, total):
        self.correct += correct
        self.total += total

    def mean(self):
        if self.total:
            return self.correct / self.total
        return -1

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

'''
Dataset
'''

class MTDatasetDis(Dataset):
    def __init__(self, Q_root, F_root, ID, Q_type='add', total_len=8, F_type=None, is_diff=None):
        super(MTDatasetDis, self).__init__()
        self.Q_root = Q_root + ('' if Q_root[-1] == '/' else '/')
        self.Q_type = Q_type
        self.F_root = F_root + ('' if F_root[-1] == '/' else '/')
        self.F_type = F_type
        self.is_diff = is_diff
        Q_file_names = sorted(os.listdir(self.Q_root))
        #F_file_names = sorted(os.listdir(self.F_root))
        length = len(Q_file_names)
        unit_len = length // total_len
        if ID == total_len - 1:
            Q_file_names = Q_file_names[ID*unit_len:]
        else:
            Q_file_names = Q_file_names[ID*unit_len:(ID+1)*unit_len]
        
        # Q_file_names = ['COCO_val2014_000000001554.json']
        # F_file_names = ['COCO_val2014_000000001554.npz']

        self.Q_list = []
        self.Q_name_list = []
        for Q_name in Q_file_names:
            with open(self.Q_root + Q_name, 'r') as f:
                json_list = json.load(f)[Q_type]
                for js in json_list:
                    self.Q_list.append(js)
                    self.Q_name_list.append(Q_name)

    def __len__(self):
        return len(self.Q_name_list)

    def __getitem__(self, index):
        Q_tuple = self.Q_list[index]['Q']
        C = self.Q_list[index]['C']
        N = self.Q_list[index]['N']
        Q_name = self.Q_name_list[index]
        prefix = '.'.join(Q_name.split('.')[:-1])
        suffix = '.npz'
        F = np.load(self.F_root + prefix + suffix)['arr_0']
        return (Q_tuple, F, C, N)

class MTDatasetText(Dataset):
    def __init__(self, Q_root, F_root, Q_type='add', F_type=None, is_diff=None):
        super(MTDatasetText, self).__init__()
        self.Q_root = Q_root + ('' if Q_root[-1] == '/' else '/')
        self.Q_type = Q_type
        self.F_root = F_root + ('' if F_root[-1] == '/' else '/')
        self.F_type = F_type
        self.is_diff = is_diff
        Q_file_names = sorted(os.listdir(self.Q_root))
        #F_file_names = sorted(os.listdir(self.F_root))
        length = len(Q_file_names)
        Q_file_names = Q_file_names[:int(length/2)]
        
        # Q_file_names = ['COCO_val2014_000000001554.json']
        # F_file_names = ['COCO_val2014_000000001554.npz']

        self.Q_list = []
        self.Q_name_list = []
        for Q_name in Q_file_names:
            with open(self.Q_root + Q_name, 'r') as f:
                json_list = json.load(f)[Q_type]
                for js in json_list:
                    self.Q_list.append(js)
                    self.Q_name_list.append(Q_name)

    def __len__(self):
        return len(self.Q_name_list)

    def __getitem__(self, index):
        Q_tuple = self.Q_list[index]['Q']
        Q_name = self.Q_name_list[index]
        prefix = '.'.join(Q_name.split('.')[:-1])
        suffix = '.npz'
        F = np.load(self.F_root + prefix + suffix)['arr_0']
        return (Q_tuple, F)

class MTDatasetCut(Dataset):
    def __init__(self, Q_root, F_root, Q_type='add', F_type=None, is_diff=None):
        super(MTDatasetCut, self).__init__()
        self.Q_root = Q_root + ('' if Q_root[-1] == '/' else '/')
        self.Q_type = Q_type
        self.F_root = F_root + ('' if F_root[-1] == '/' else '/')
        self.F_type = F_type
        self.is_diff = is_diff
        Q_file_names = sorted(os.listdir(self.Q_root))
        F_file_names = sorted(os.listdir(self.F_root))
        #print(sorted(os.listdir(self.Q_root))[300:800])
        #Q_file_names = ['COCO_val2014_000000009274.json']

        self.Q_list = []
        self.Q_name_list = []
        for Q_name in Q_file_names:
            with open(self.Q_root + Q_name, 'r') as f:
                json_list = json.load(f)[Q_type]
                for js in json_list:
                    self.Q_list.append(js)
                    self.Q_name_list.append(Q_name)

    def __len__(self):
        return len(self.Q_name_list)

    def __getitem__(self, index):
        Q_tuple = self.Q_list[index]['Q']
        Q_name = self.Q_name_list[index]
        prefix = '.'.join(Q_name.split('.')[:-1])
        suffix = '.npz'
        npz = np.load(self.F_root + prefix + suffix, allow_pickle=True)
        original = npz['origin']
        X_list = npz['X']
        Y_list = npz['Y']
        return (Q_tuple, (original, X_list, Y_list))

class MTDatasetImage(Dataset):
    def __init__(self, Q_root, F_root, Q_type='add', F_type='masked', is_diff=True):
        super(MTDatasetImage, self).__init__()
        self.Q_root = Q_root + ('' if Q_root[-1] == '/' else '/')
        self.Q_type = Q_type
        self.F_root = F_root + ('' if F_root[-1] == '/' else '/')
        self.F_type = F_type
        self.is_diff = is_diff
        self.Q_file_names = sorted(os.listdir(self.Q_root))
        self.F_file_names = sorted(os.listdir(self.F_root))
        #self.Q_file_names = ['COCO_val2014_000000006484.json']
        #self.F_file_names = ['COCO_val2014_000000006484.npz']

    def __len__(self):
        return len(self.Q_file_names)

    def __getitem__(self, index):
        Q_path = self.Q_root + self.Q_file_names[index]
        F_path = self.F_root + self.F_file_names[index]
        npz = np.load(F_path, allow_pickle=True)
        with open(Q_path, 'r') as f:
            json_list = json.load(f)
        original = npz['origin']
        tested_F_list = npz[self.F_type]
        diff = 'diff' if self.is_diff else 'same'
        tested_Q_list = []
        for i in range(len(tested_F_list)):
            tested_Q = [js['Q'] for js in json_list[i][diff][self.Q_type]]
            tested_Q_list.append(tested_Q)
        return (original, tested_Q_list, tested_F_list)

'''
Tester
'''

class DefaultArgs(object):
    def __init__(self):
        self.Q_root = '/root/data/VQA/questions/abstract/OpenEnded_abstract_v002_val2015_questions.json'
        self.F_root = '/root/data/Oscar-features/abstract/'
        self.Q_type = 'add'
        self.F_type = 'insert'
        self.is_diff = True

class Tester(object):
    def __init__(self, name, args, model, inference):
        if args is None:
            args = DefaultArgs()
        self.args = args
        self.model = model
        self.inference = inference
        self.file = open('/root/code/MT-VQA/logs/%s.txt' % name, 'a+')

        self.print_log('Q_root: %s' % self.args.Q_root)
        self.print_log('F_root: %s' % self.args.F_root)
        self.print_log('Q_type: %s' % self.args.Q_type)
        self.print_log('F_type: %s' % self.args.F_type)
        self.print_log('is_diff: %d' % self.args.is_diff)

    def print_log(self, text):
        print(text)
        print(text, file=self.file)

    def transfer(self, ans):
        if ans.isdigit():
            return int(ans)
        dic = {
            'yes': 1, 'no': 0,
            'true': 1, 'false': 0
        }
        if ans.lower() in dic.keys():
            return dic[ans.lower()]
        if ans.lower() in ['many', 'much', 'lot', 'a lot']:
            return float('-inf')
        else:
            return 0x1FFF

    def text_correct(self, answer_tuple):
        if self.args.Q_type == 'add':
            return int(answer_tuple[0] == answer_tuple[1] + answer_tuple[2]), 1
        if self.args.Q_type == 'not':
            return int(answer_tuple[0] != answer_tuple[1]), 1
        if self.args.Q_type == 'equ':
            return int(answer_tuple[0] == answer_tuple[1]), 1

    def cut_correct(self, original, answer_list):
        if self.args.Q_type == 'add':
            sum_tuple = [0, 0, 0]
            for answer_tuple in answer_list:
                for i in range(3):
                    sum_tuple[i] += answer_tuple[i]
            results = [
                int(original[0] == sum_tuple[0]), 
                int(original[1] == sum_tuple[1]),
                int(original[2] == sum_tuple[2])
            ]
        if self.args.Q_type == 'not':
            sum_tuple = [0, 0]
            for answer_tuple in answer_list:
                sum_tuple[0] |= answer_tuple[0]
                sum_tuple[1] &= answer_tuple[1]
            results = [
                int(original[0] == sum_tuple[0]), 
                int(original[1] == sum_tuple[1])
            ]
        return sum(results), len(results)

    def image_correct(self, original_list, tested_list):
        length = len(tested_list)
        results = []
        for i in range(length):
            original_tuple = original_list[i]
            tested_tuple = tested_list[i]
            for j in range(len(tested_tuple)):
                results.append(
                    original_tuple[j] == tested_tuple[j] if not self.args.is_diff \
                    else (original_tuple[j] == tested_tuple[j] - 1 \
                        if self.args.F_type == 'insert' else \
                        original_tuple[j] == tested_tuple[j] + 1
                    )
                )
        return sum(results), len(results)

    def a2q_correct(self, Q_list, extended_answers, answer_type):
        if answer_type == 'other':
            return int(extended_answers == ['yes', 'no', 'no', 'yes']), 1
        else:
            if extended_answers[1] != extended_answers[2]:
                if extended_answers[1]:
                    return int(extended_answers[0] in Q_list[1]), 1
                else:
                    return int(extended_answers[0] in Q_list[2]), 1
            return 0, 0

    def tuple_answer(self, Q_tuple, F, allow_transfer=True):
        answer_tuple = []
        for Q in Q_tuple:
            answer = self.inference(self.model, Q, F)
            if allow_transfer:
                answer = self.transfer(answer)
            answer_tuple.append(answer)
        return answer_tuple

    def test_a2q(self, F_path, Q_path=None, A_path=None, answer_type='other', topk=5):
        self.print_log('test_a2q')
        if Q_path == None:
            Q_path = '/root/data/VQA/questions/abstract/OpenEnded_abstract_v002_val2015_questions.json'
        if A_path == None:
            A_path = '/root/data/VQA/questions/abstract/abstract_v002_val2015_annotations.json'
        
        self.print_log('Q_path: %s' % Q_path)
        self.print_log('A_path: %s' % A_path)

        a2q = Answer2Question(Q_path, A_path)
        F_names = sorted(os.listdir(F_path))
        F_dict = dict()
        for name in F_names:
            image_id = int(name.split('.')[0].split('_')[-1])
            F_dict[image_id] = np.load(F_path + name)['arr_0']
        record = Record()
        record_OR = Record()
        t_start = time.time()
        progress = progressbar.ProgressBar(maxval=len(a2q.pair)).start()
        for i, p in enumerate(a2q.pair):
            progress.update(i + 1)
            F = F_dict[p['image_id']]
            answer_list = self.inference(self.model, p['Q'], F, topk)
            
            Q_list = None
            if answer_type == 'other' and a2q.is_other(p):
                # print(p['Q'])
                # print(answer_list)
                Q_list = a2q.fromAnswer(p['Q'], p['question_type'], list(answer_list))
                if Q_list:
                    extended_answers = []
                    for Q in Q_list:
                        answer = self.inference(self.model, Q, F)
                        extended_answers.append(answer)
                    is_correct, total = self.a2q_correct(Q_list, extended_answers, 'other')
                    # print(p['Q'])
                    # print(Q_list)
                    # print(extended_answers)
                    # print(is_correct, '\t', total)
                    record.add(is_correct, total)
            else:
                if a2q.is_or(p):
                    Q_list = a2q.fromOR(p['Q'], p['question_type'])
                elif a2q.is_yn(p):
                    Q_list = a2q.toOR(p['Q'], p['question_type'])
                if Q_list:
                    extended_answers = []
                    for Q in Q_list:
                        answer = self.inference(self.model, Q, F)
                        extended_answers.append(answer)
                    is_correct, total = self.a2q_correct(Q_list, extended_answers, 'or')
                    record_OR.add(is_correct, total)
            if i % 1000 == 0:
                print('Acc: %f' % record.mean())
                print('Acc OR: %f' % record_OR.mean())
        progress.finish()
        t_end = time.time()
        self.print_log('Cost Time: %f' % (t_end - t_start))
        self.print_log('Accuracy: %f' % record.mean())
        self.print_log('Correct: %d' % record.correct)
        self.print_log('Total: %d' % record.total)
        self.print_log('Accuracy OR: %f' % record_OR.mean())
        self.print_log('Correct OR: %d' % record_OR.correct)
        self.print_log('Total OR: %d' % record_OR.total)

    def test_dis(self, ID):
        self.print_log('test_dis')
        dataset = MTDatasetDis(
            Q_root=self.args.Q_root, 
            F_root=self.args.F_root, 
            Q_type=self.args.Q_type, 
            F_type=None, 
            is_diff=None,
            ID=ID
        )
        record_list = [
                [Record(), Record(), Record()],
                [Record(), Record(), Record()]
                ]
        t_start = time.time()
        def _print_acc(record_list):
            for c in [0, 1]:
                for n in [0, 1, 2]:
                    print('C=%d,N=%d,Acc is:%f' % (c, n, record_list[c][n].mean()))
        progress = progressbar.ProgressBar(maxval=len(dataset)).start()
        for i, (Q_tuple, F, C, N) in enumerate(dataset):
            progress.update(i + 1)
            answer_tuple = self.tuple_answer(Q_tuple, F)
            # print(Q_tuple)
            # print(answer_tuple)
            is_correct, total = self.text_correct(answer_tuple)
            record_list[int(C)][int(N)].add(is_correct, total)
            if i % 10000 == 0:
                _print_acc(record_list)
        progress.finish()
        t_end = time.time()
        self.print_log('Cost Time: %f' % (t_end - t_start))
        for c in [0, 1]:
            for n in [0, 1, 2]:
                self.print_log('C=%d, N=%d, Accuracy: %f' % (c, n, record_list[c][n].mean()))
                self.print_log('Correct: %d' % (record_list[c][n].correct))
                self.print_log('Total: %d' % (record_list[c][n].total))

    def test_text(self):
        self.print_log('test_text')
        dataset = MTDatasetText(
            Q_root=self.args.Q_root, 
            F_root=self.args.F_root, 
            Q_type=self.args.Q_type, 
            F_type=None, 
            is_diff=None
        )
        record = Record()
        t_start = time.time()
        progress = progressbar.ProgressBar(maxval=len(dataset)).start()
        for i, (Q_tuple, F) in enumerate(dataset):
            progress.update(i + 1)
            answer_tuple = self.tuple_answer(Q_tuple, F)
            # print(Q_tuple)
            # print(answer_tuple)
            is_correct, total = self.text_correct(answer_tuple)
            record.add(is_correct, total)
            if i % 10000 == 0:
                print('Acc: %f' % record.mean())
        progress.finish()
        t_end = time.time()
        self.print_log('Cost Time: %f' % (t_end - t_start))
        self.print_log('Accuracy: %f' % record.mean())
        self.print_log('Correct: %d' % record.correct)
        self.print_log('Total: %d' % record.total)

    def test_cut(self):
        self.print_log('test_cut')
        dataset = MTDatasetCut(
            Q_root=self.args.Q_root, 
            F_root=self.args.F_root, 
            Q_type=self.args.Q_type, 
            F_type=None, 
            is_diff=None
        )
        record_X = Record()
        record_Y = Record()
        t_start = time.time()
        progress = progressbar.ProgressBar(maxval=len(dataset)).start()
        for i, (Q_tuple, (original, X_list, Y_list)) in enumerate(dataset):
            progress.update(i + 1)
            print('Q: ', Q_tuple)
            original_tuple = self.tuple_answer(Q_tuple, original)
            print('original: ', original_tuple)
            if len(X_list):
                answer_list = []
                for X in X_list:
                    X_tuple = self.tuple_answer(Q_tuple, X)
                    answer_list.append(X_tuple)
                is_correct, total = self.cut_correct(original_tuple, answer_list)
                record_X.add(is_correct, total)
                print('X: ', answer_list)
            if len(Y_list):
                answer_list = []
                for Y in Y_list:
                    Y_tuple = self.tuple_answer(Q_tuple, Y)
                    answer_list.append(Y_tuple)
                is_correct, total = self.cut_correct(original_tuple, answer_list)
                record_Y.add(is_correct, total)
                print('Y: ', answer_list)
            if i % 1000 == 0:
                print('Acc X: %f' % record_X.mean())
                print('Acc Y: %f' % record_Y.mean())
        progress.finish()
        t_end = time.time()
        self.print_log('Cost Time: %f' % (t_end - t_start))
        self.print_log('Accuracy of X: %f' % record_X.mean())
        self.print_log('Correct X: %d' % record_X.correct)
        self.print_log('Total X: %d' % record_X.total)
        self.print_log('Accuracy of Y: %f' % record_Y.mean())
        self.print_log('Correct Y: %d' % record_Y.correct)
        self.print_log('Total Y: %d' % record_Y.total)

    def test_image(self):
        self.print_log('test_image')
        dataset = MTDatasetImage(
            Q_root=self.args.Q_root, 
            F_root=self.args.F_root, 
            Q_type=self.args.Q_type, 
            F_type=self.args.F_type, 
            is_diff=self.args.is_diff
        )
        record = Record()
        t_start = time.time()
        progress = progressbar.ProgressBar(maxval=len(dataset)).start()
        for i, (original, tested_Q_list, tested_F_list) in enumerate(dataset):
            progress.update(i + 1)
            if len(tested_Q_list):
                original_list = []
                tested_list = []
                for j in range(len(tested_Q_list)):
                    print('INDEX: ', j)
                    for k in range(len(tested_Q_list[j])):
                        original_tuple = self.tuple_answer(tested_Q_list[j][k], original)
                        tested_tuple = self.tuple_answer(tested_Q_list[j][k], tested_F_list[j])
                        if original_tuple != tested_tuple:
                            print('Q: ', tested_Q_list[j][k])
                            print('original: ', original_tuple)
                            print('tested: ', tested_tuple)
                        original_list.append(original_tuple)
                        tested_list.append(tested_tuple)
                is_correct, total = self.image_correct(original_list, tested_list)
                record.add(is_correct, total)
            if i % 100 == 0:
                print('Acc: %f' % record.mean())
        progress.finish()
        t_end = time.time()
        self.print_log('Cost Time: %f' % (t_end - t_start))
        self.print_log('Accuracy: %f' % record.mean())
        self.print_log('Correct: %d' % record.correct)
        self.print_log('Total: %d' % record.total)
