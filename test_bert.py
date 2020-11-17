import yaml
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys
import os


import torchvision.models as models
import torchvision.transforms as transforms

# from PIL import Image
# from IPython.display import display, HTML, clear_output
# from ipywidgets import widgets, Layout
# from io import BytesIO
from argparse import Namespace

from mmf.datasets.processors.processors import VocabProcessor, VQAAnswerProcessor, TransformerBboxProcessor
from mmf.datasets.processors.bert_processors import BertTokenizer
from mmf.models.visual_bert import VisualBERT
from mmf.models.vilbert import ViLBERT
from mmf.common.registry import registry
from mmf.common.sample import Sample, SampleList
from mmf.utils.env import setup_imports
from mmf.utils.configuration import Configuration

setup_imports()

import argparse

parser = argparse.ArgumentParser(description='BERT Args')

parser.add_argument('--Q_root', type=str, default='', help='Q_root')
parser.add_argument('--F_root', type=str, default='', help='F_root')
parser.add_argument('--R_root', type=str, default='', help='R_root')
parser.add_argument('--Q_type', type=str, default='', help='Q_type')
parser.add_argument('--F_type', type=str, default='', help='F_type')
parser.add_argument('--is_diff', type=int, default=1, help='is_diff')

args = parser.parse_args()

os.environ['MMF_CACHE_DIR'] = '/root/collection/MMF_CACHE_DIR/'
os.environ['PYTORCH_TRANSFORMERS_CACHE'] = '/root/collection/PYTORCH_TRANSFORMERS_CACHE'

class TestedVisualBert:

    def __init__(self, trained_on='train'):
        self.trained_on = trained_on
        self.root = '/root/collection/pretrained-models'
        self._init_processors()
        self.bert_model = self._build_bert_model()

    def _init_processors(self):
        args = Namespace()
        args.opts = [
            "config=projects/visual_bert/configs/vqa2/defaults.yaml",
            "datasets=vqa2",
            "model=visual_bert",
            "evaluation.predict=True"
        ]
        args.config_override = None

        configuration = Configuration(args=args)

        config = self.config = configuration.config
        vqa_config = config.dataset_config.vqa2
        text_processor_config = vqa_config.processors.text_processor
        answer_processor_config = vqa_config.processors.answer_processor

        text_processor_config.params.vocab.vocab_file = self.root + "/content/model_data/vocabulary_100k.txt"
        answer_processor_config.params.vocab_file = self.root + "/content/model_data/answers_vqa.txt"
        # Add preprocessor as that will needed when we are getting questions from user
        self.text_processor = BertTokenizer(text_processor_config.params)
        self.answer_processor = VQAAnswerProcessor(answer_processor_config.params)

        registry.register("vqa2_text_processor", self.text_processor)
        registry.register("vqa2_answer_processor", self.answer_processor)
        registry.register("vqa2_num_final_outputs", 
                          self.answer_processor.get_vocab_size())

    def _build_bert_model(self):
        if self.trained_on == 'train':
            state_dict = torch.load(self.root + '/content/model_data/visual_bert.finetuned.vqa2.from_coco_train/model.pth')
        elif self.trained_on == 'train+val':
            state_dict = torch.load(self.root + '/content/model_data/visual_bert.finetuned.vqa2.from_coco_train_val/model.pth')
        else:
            print('trained_on error')
        model_config = self.config.model_config.visual_bert
        model_config.model_data_dir = self.root + "/content/"
        model = VisualBERT(model_config)
        model.build()
        model.init_losses()

        if list(state_dict.keys())[0].startswith('module') and \
            not hasattr(model, 'module'):
            state_dict = self._multi_gpu_state_to_single(state_dict)

        print(state_dict)
        model.load_state_dict(state_dict, strict=False)
        model.to("cuda")
        model.eval()

        return model
        
    def _multi_gpu_state_to_single(self, state_dict):
        new_sd = {}
        for k, v in state_dict.items():
            if not k.startswith('module.'):
                raise TypeError("Not a multiple GPU state of dict")
            k1 = k[7:]
            new_sd[k1] = v
        return new_sd
  
    def predict(self, Q, F, topk):
        with torch.no_grad():
            detectron_features = torch.from_numpy(F)
            #resnet_features = torch.from_numpy(R)
            
            processed_text = self.text_processor({"text": Q})
            sample = Sample(processed_text)
            #sample.text = processed_text["text"]
            sample.text_len = len(processed_text["tokens"])

            sample.image_feature_0 = detectron_features
            sample.image_info_0 = Sample({
              "max_features": torch.tensor(100, dtype=torch.long)
            })

            #sample.image_feature_1 = resnet_features
            #print('res: ', resnet_features.shape)
            sample_list = SampleList([sample])
            #print(type(sample_list))
            sample_list = sample_list.to("cuda")

            scores = self.bert_model(sample_list)["scores"]
            scores = torch.nn.functional.softmax(scores, dim=1)
            actual, indices = scores.topk(topk, dim=1)

            top_indices = indices[:topk]
            top_scores = actual[:topk]

            probs = []
            answers = []

            for idx, score in enumerate(top_scores[0]):
                probs.append(score.item())
                answers.append(
                    self.answer_processor.idx2word(top_indices[0][idx].item())
                )
    
        gc.collect()
        torch.cuda.empty_cache()

        return probs, answers


model = TestedVisualBert(trained_on='train+val')
def inference(model, Q, F, topk=None):
    if topk is None:
        _, answers = model.predict(Q=Q, F=F, topk=1)
        print(Q)
        print(answers)
        return answers[0]
    else:
        _, answers = model.predict(Q=Q, F=F, topk=topk)
        return answers

sys.path.insert(0, '/root/code/MT-VQA/')
from Tester import Tester

print('text')
args.Q_root = '/root/data/questions/with0/'
args.F_root = '/root/data/features/real/BUTD/'
for q_type in ['add', 'not', 'equ']:
    args.Q_type = q_type
    tester = Tester('visual_bert-text', args, model, inference)
    tester.test_text()

# print('a2q')
# tester = Tester('visual_bert-a2q', args, model, inference)
# tester.test_a2q(F_path='/root/data/features/abstract/BUTD/')

# print('cut')
# args.Q_root = '/root/data/COCO/CUT/question/'
# args.F_root = '/root/data/COCO/CUT/feature/BUTD/'
# for q_type in ['add', 'not']:
#     args.Q_type = q_type
#     tester = Tester('visual_bert-cut', args, model, inference)
#     tester.test_cut()

# print('remove')
# args.Q_root = '/root/data/COCO/REMOVE/question/'
# args.F_root = '/root/data/COCO/REMOVE/feature/masked/BUTD/'
# args.F_type = 'masked'
# for diff in [1, 0]:
#     args.is_diff = diff
#     for q_type in (['add'] if diff else ['add', 'not']):
#         args.Q_type = q_type
#         tester = Tester('visual_bert-remove', args, model, inference)
#         tester.test_image()

# print('insert')
# args.Q_root = '/root/data/COCO/INSERT/question/'
# args.F_root = '/root/data/COCO/INSERT/feature/BUTD/'
# args.F_type = 'insert'
# for diff in [1, 0]:
#     args.is_diff = diff
#     for q_type in (['add'] if diff else ['add', 'not']):
#         args.Q_type = q_type
#         tester = Tester('visual_bert-insert', args, model, inference)
#         tester.test_image()
