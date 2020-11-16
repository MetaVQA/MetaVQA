import os
import io
import numpy as np
import cv2
import torch
import json

# import some common detectron2 utilities
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image

class Image2Feat(object):

    def __init__(self, data_path, yaml_path, weights_path, num_obj=36):

        self.vg_classes = []
        with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
            for object in f.readlines():
                self.vg_classes.append(object.split(',')[0].lower().strip())
                
        self.vg_attrs = []
        with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
            for object in f.readlines():
                self.vg_attrs.append(object.split(',')[0].lower().strip())

        self.cfg = get_cfg()
        self.cfg.merge_from_file(yaml_path)
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
        self.cfg.MODEL.WEIGHTS = weights_path
        
        self.predictor = DefaultPredictor(self.cfg)

        self.NUM_OBJECTS = num_obj

        self.people_list = [
            'people', 'person', 'child', 'man', 'woman', 
            'boy', 'girl'
        ]
        self.color_list = [
            'black', 'blue', 'brown', 'dark', 'gold',
            'gray', 'green', 'orange', 'pink', 'purple',
            'red', 'white', 'yellow'
        ]
        self.action_list = [
            'throwing', 'blowing', 'cooking', 'flying', 'skiing', 'surfing', 'sleeping', 'jumping',
            'swimming', 'crouching', 'waving', 'rolling', 'grazing', 'walking', 'smiling', 'skateboarding',
            'laying', 'batting', 'sitting', 'running', 'moving', 'eating', 'skating', 'playing',
            'standing', 'driving', 'watching', 'parking', 'bending', 'hanging', 'squatting', 'riding', 
            'landing', 'resting', 'looking', 'holding', 'racing', 'kneeling', 'sliding', 'lying',
            'serving', 'hitting', 'pointing', 'posing', 'wearing', 'swinging', 'laughing','talking'
        ]

        self.add = ' and '
        self.end = ' ?'
        self.template1 = 'What is the total number of '
        self.template2 = 'Is there any '
        self.template3 = 'Is there no '

    def plural(self, word):
        dic = {
            'child': 'children', 'foot': 'feet',
            'tooth': 'teeth', 'mouse': 'mice', 
            'man': 'men', 'woman': 'women'
        }
        if word in dic.keys():
            return dic[word]
        elif word in dic.values():
            return word
        elif word.endswith('y') and (word[-2] not in 'aeiou'):
            return word[:-1] + 'ies'
        elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
            return word + 'es'
        elif word.endswith('an'):
            return word[:-2] + 'en'
        else:
            return word + 's'

    def doit(self, raw_image):
        with torch.no_grad():
            raw_height, raw_width = raw_image.shape[:2]
            #print("Original image size: ", (raw_height, raw_width))
            
            # Preprocessing
            image = self.predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
            #print("Transformed image size: ", image.shape[:2])
            image_height, image_width = image.shape[:2]
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": raw_height, "width": raw_width}]
            images = self.predictor.model.preprocess_image(inputs)
            
            # Run Backbone Res1-Res4
            features = self.predictor.model.backbone(images.tensor)
            
            # Generate proposals with RPN
            proposals, _ = self.predictor.model.proposal_generator(images, features, None)
            proposal = proposals[0]
            #print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)
            
            # Run RoI head for each proposal (RoI Pooling + Res5)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in self.predictor.model.roi_heads.in_features]
            box_features = self.predictor.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
            #print('Pooled features size:', feature_pooled.shape)
            
            # Predict classes and boxes for each proposal.
            pred_class_logits, pred_attr_logits, pred_proposal_deltas = self.predictor.model.roi_heads.box_predictor(feature_pooled)
            outputs = FastRCNNOutputs(
                self.predictor.model.roi_heads.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.predictor.model.roi_heads.smooth_l1_beta,
            )
            probs = outputs.predict_probs()[0]
            boxes = outputs.predict_boxes()[0]
            
            box_width = boxes[:, 2] - boxes[:, 0]
            box_height = boxes[:, 3] - boxes[:, 1]
            scaled_width = box_width / image_width
            scaled_height = box_height / image_height
            scaled_x = boxes[:, 0] / image_width
            scaled_y = boxes[:, 1] / image_height
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]
            spatial_features = torch.cat( (scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height), 1)
            oscar_features = torch.cat((feature_pooled, spatial_features), 1)
            
            attr_prob = pred_attr_logits[..., :-1].softmax(-1)
            max_attr_prob, max_attr_label = attr_prob.max(-1)
            
            # Note: BUTD uses raw RoI predictions,
            #       we use the predicted boxes instead.
            # boxes = proposal_boxes[0].tensor    
            
            # NMS
            for nms_thresh in np.arange(0.5, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes, probs, image.shape[1:], 
                    score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=self.NUM_OBJECTS
                )
                if len(ids) >= self.NUM_OBJECTS:
                    break
            instances = detector_postprocess(instances, raw_height, raw_width)
            roi_features = feature_pooled[ids].detach()
            oscar_features = oscar_features[ids].detach()
            max_attr_prob = max_attr_prob[ids].detach()
            max_attr_label = max_attr_label[ids].detach()
            instances.attr_scores = max_attr_prob
            instances.attr_classes = max_attr_label
            
            return instances, roi_features, oscar_features

    def add_Q1_Q3(self, Qs, obj1, obj2, C):
        obj1 = obj1.split(' ')
        obj1[-1] = self.plural(obj1[-1])
        obj1 = ' '.join(obj1)
        obj2 = obj2.split(' ')
        obj2[-1] = self.plural(obj2[-1])
        obj2 = ' '.join(obj2)
        Qs['add'].append(
                {'Q': (self.template1 + obj1 + self.add + obj2 + self.end,
                       self.template1 + obj1 + self.end,
                       self.template1 + obj2 + self.end),
                 'C': C}
            )
        Qs['equ'].append(
                {'Q': (self.template1 + obj1 + self.add + obj2 + self.end,
                       self.template1 + obj2 + self.add + obj1 + self.end),
                 'C': C}
            )
        return Qs

    def add_Q2(self, Qs, obj, C):
        obj = obj.split(' ')
        obj[-1] = self.plural(obj[-1])
        obj = ' '.join(obj)
        Qs['not'].append(
                {'Q': (self.template2 + obj + self.end,
                       self.template3 + obj + self.end),
                 'C': C}
            )
        return Qs

    def get_class_name(self, idx):
        return self.vg_classes[idx]

    def get_attr_name(self, idx):
        return self.vg_attrs[idx]

    def generate_Q(self, object_list, Question, 
                   obj_th1=0.9, obj_th2=0.7, obj_th3=0, 
                   attr_th1=0.9, attr_th2=0.7, attr_th3=0, 
                   rand_q1=5, rand_q2=5):
        class_name = self.vg_classes
        objects = object_list
        # Q1 and Q3
        # without attributes
        len_objects = len(objects.keys())
        key_list = list(objects.keys())
        for idx1 in range(len_objects):
            for idx2 in range(idx1+1, len_objects):
                obj1 = key_list[idx1]
                obj2 = key_list[idx2]
                
                prob = np.random.uniform()
                if prob > obj_th1: #0.9
                    # 10%: 0 + 0 == (0 + 0)
                    _idx1 = int(np.random.uniform(high=len(class_name)))
                    _obj1 = class_name[_idx1]
                    while _obj1 in key_list:
                        _idx1 = int(np.random.uniform(high=len(class_name)))
                        _obj1 = class_name[_idx1]

                    _idx2 = int(np.random.uniform(high=len(class_name)))
                    while _idx1 == _idx2:
                        _idx2 = int(np.random.uniform(high=len(class_name)))
                    _obj2 = class_name[_idx2]

                    Question = self.add_Q1_Q3(Question, _obj1, _obj2, False)
                elif prob > obj_th2: #0.7
                    # 20%: x1 + 0 == (x1 + 0) / 0 + x1 == (0 + x1)
                    _idx1 = int(np.random.uniform(high=len(class_name)))
                    _obj1 = class_name[_idx1]
                    while _obj1 in key_list:
                        _idx1 = int(np.random.uniform(high=len(class_name)))
                        _obj1 = class_name[_idx1]
                    
                    if np.random.uniform() > 0.5:
                        Question = self.add_Q1_Q3(Question, _obj1, obj2, False)
                    else:
                        Question = self.add_Q1_Q3(Question, obj2, _obj1, False)
                elif prob >= obj_th3:
                    # 70%: x1 + x2 == (x1 + x2)
                    if np.random.uniform() > 0.5:
                        Question = self.add_Q1_Q3(Question, obj1, obj2, False)
                    else:
                        Question = self.add_Q1_Q3(Question, obj2, obj1, False)
        # with attributes
        for obj in objects.keys():
            len_attr = len(objects[obj]['attr'])
            val_list = objects[obj]['attr']

            if obj in self.people_list:
                target_attr = self.action_list
            else:
                target_attr = self.color_list
            
            if len_attr == 0:
                if np.random.uniform() >= attr_th3:
                    for _ in range(rand_q1):
                        _idx1 = int(np.random.uniform(high=len(target_attr)))
                        _attr1 = target_attr[_idx1]

                        _idx2 = int(np.random.uniform(high=len(target_attr))) 
                        while _idx2 == _idx1:
                            _idx2 = int(np.random.uniform(high=len(target_attr)))
                        _attr2 = target_attr[_idx2]
                        Question = self.add_Q1_Q3(Question, (_attr1+' '+obj), (_attr2+' '+obj), True)
            else:
                for idx1 in range(len_attr):
                    for idx2 in range(idx1+1, len_attr):
                        attr1 = val_list[idx1]
                        attr2 = val_list[idx2]
                        
                        prob = np.random.uniform()
                        if prob > attr_th1:#0.9
                            _idx1 = int(np.random.uniform(high=len(target_attr)))
                            _attr1 = target_attr[_idx1]
                            while _attr1 in val_list:
                                _idx1 = int(np.random.uniform(high=len(target_attr)))
                                _attr1 = target_attr[_idx1]

                            _idx2 = int(np.random.uniform(high=len(target_attr))) 
                            while _idx2 == _idx1:
                                _idx2 = int(np.random.uniform(high=len(target_attr)))
                            _attr2 = target_attr[_idx2]

                            Question = self.add_Q1_Q3(Question, (_attr1+' '+obj), (_attr2+' '+obj), True)
                        elif prob > attr_th2:#0.7
                            _idx1 = int(np.random.uniform(high=len(target_attr)))
                            _attr1 = target_attr[_idx1]
                            while _attr1 in val_list:
                                _idx1 = int(np.random.uniform(high=len(target_attr)))
                                _attr1 = target_attr[_idx1]

                            if np.random.uniform() > 0.5:
                                Question = self.add_Q1_Q3(Question, (_attr1+' '+obj), (attr2+' '+obj), True)
                            else:
                                Question = self.add_Q1_Q3(Question, (attr2+' '+obj), (_attr1+' '+obj), True)

                        elif prob >= attr_th3: #0
                            if np.random.uniform() > 0.5:
                                Question = self.add_Q1_Q3(Question, (attr1+' '+obj), (attr2+' '+obj), True)
                            else:
                                Question = self.add_Q1_Q3(Question, (attr2+' '+obj), (attr1+' '+obj), True)
        # Q2
        # without attributes            
        for obj in objects.keys():
            prob = np.random.uniform()
            if prob > obj_th2:
                _obj = class_name[int(np.random.uniform(high=len(class_name)))]
                while _obj in objects.keys():
                    _obj = class_name[int(np.random.uniform(high=len(class_name)))]
                Question = self.add_Q2(Question, _obj, False)
            elif prob >= obj_th3:
                Question = self.add_Q2(Question, obj, False)

        # with attributes
        for obj in objects.keys():
            val_list = objects[obj]['attr']
            if len(val_list) == 0:
                if np.random.uniform() >= attr_th3:
                    for _ in range(rand_q2):
                        if obj in self.people_list:
                            target_attr = self.action_list
                        else:
                            target_attr = self.color_list
                        _attr = target_attr[int(np.random.uniform(high=len(target_attr)))]
                        Question = self.add_Q2(Question, (_attr+' '+obj), True)
            else:    
                for attr in val_list:
                    prob = np.random.uniform()
                    if prob > attr_th2:
                        if obj in self.people_list:
                            target_attr = self.action_list
                        else:
                            target_attr = self.color_list
                        _attr = target_attr[int(np.random.uniform(high=len(target_attr)))]
                        while _attr in val_list:
                            _attr = target_attr[int(np.random.uniform(high=len(target_attr)))]
                        Question = self.add_Q2(Question, (_attr+' '+obj), True)
                    elif prob >= attr_th3:
                        Question = self.add_Q2(Question, (attr+' '+obj), True)

        return Question