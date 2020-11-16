import json
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import copy
import os
import progressbar
import PIL
from PIL import Image
import numpy as np
import random

from pycocotools.coco import COCO

from image_to_feature import Image2Feat

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_bboxs(img_id, json_data):
    bboxs = []
    c_ids = []
    annos = []
    for anno in json_data['annotations']:
        if anno['image_id'] == img_id:
            bboxs.append(anno['bbox'])
            c_ids.append(anno['category_id'])
            annos.append(anno)
    return bboxs, c_ids, annos

def possible_points(inserted_bbox, target_bbox, W, H):
    # No. edge of target_bbox
    #        1
    #     ------
    #  3 |      | 4
    #    |      |
    #     ------
    #        2
    points = []
    x_begin = int(max(0, target_bbox[0]-inserted_bbox[2]))
    x_end = int(min(target_bbox[0]+target_bbox[2], W-inserted_bbox[2]))
    y_begin = int(max(0, target_bbox[1]-inserted_bbox[3]))
    y_end = int(min(target_bbox[1]+target_bbox[3], H-inserted_bbox[3]))
    if x_begin <= x_end:
        if target_bbox[1] - inserted_bbox[3] >= 0:
            points += [(x, target_bbox[1] - inserted_bbox[3]) \
                        for x in list(
                            range(x_begin, x_end+1)
                            )]
        if target_bbox[1] + target_bbox[3] + inserted_bbox[3] <= H:
            points += [(x, target_bbox[1] + target_bbox[3]) \
                        for x in list(
                            range(x_begin, x_end+1)
                            )]
    if y_begin <= y_end:
        if target_bbox[0] - inserted_bbox[2] >= 0:
            points += [(target_bbox[0] - inserted_bbox[2], y) \
                        for y in list(
                            range(y_begin, y_end+1)
                            )]
        if target_bbox[0] + target_bbox[2] + inserted_bbox[2] <= W:
            points += [(target_bbox[0] + target_bbox[2], y) \
                        for y in list(
                            range(y_begin, y_end+1)
                            )]
    return points

def insertable_points(bbox_list, c_id_list, W, H):
    w_rec = dict()
    h_rec = dict()
    for i in range(len(bbox_list)):
        bbox = bbox_list[i]
        c_id = c_id_list[i]
        if c_id in w_rec.keys():
            w_rec[c_id].append(bbox[2])
            h_rec[c_id].append(bbox[3])
        else:
            w_rec[c_id] = [bbox[2]]
            h_rec[c_id] = [bbox[3]]
    for k in w_rec.keys():
        w_rec[k] = int(sum(w_rec[k]) / len(w_rec[k]))
        h_rec[k] = int(sum(h_rec[k]) / len(h_rec[k]))
    
    p_points = dict()
    points = dict()
    for c_id in set(c_id_list):
        p_points[c_id] = []
        inserted_bbox = [0, 0, w_rec[c_id], h_rec[c_id]]
        for bbox in bbox_list:
            p_points[c_id] += possible_points(inserted_bbox, bbox, W, H)

    for c_id in set(c_id_list):
        p_points[c_id] = list(set(p_points[c_id]))

    for c_id in set(c_id_list):
        points[c_id] = []
        cand_points = p_points[c_id]
        for cand in cand_points:
            inserted_bbox = [cand[0], cand[1], w_rec[c_id], h_rec[c_id]]
            is_ok = True
            for bbox in bbox_list:
                if isBoxOverlap(inserted_bbox, bbox):
                    is_ok = False
                    break
            if is_ok:
                points[c_id].append(cand)
    return points, w_rec, h_rec

def insert_box(img, pool, bbox_list, c_id_list, W, H, k=2):
    points, w_rec, h_rec = insertable_points(bbox_list, c_id_list, W, H)
    final_images = []
    final_c_ids = []
    import copy
    for c_id in list(set(c_id_list)) * k:
        if len(points[c_id]):
            temp_img = copy.deepcopy(img)
            p = points[c_id][np.random.randint(len(points[c_id]))]
            bbox = [int(p[0]), int(p[1]), w_rec[c_id], h_rec[c_id]]
            inserted = pool[c_id][np.random.randint(len(pool[c_id]))]
            cnt = 0
            while cnt < len(pool[c_id]) \
                and (
                    h_rec[c_id] > 2*inserted['bbox'][3] \
                    or w_rec[c_id] > 2*inserted['bbox'][2]
                ):
                inserted = pool[c_id][np.random.randint(len(pool[c_id]))]
                cnt += 1
            inserted_img = cv2.imread(inserted['image'])
            #inserted_img = cv2.cvtColor(inserted_img, cv2.COLOR_RGBA2BGR)
            inserted_img = cv2.resize(inserted_img, (w_rec[c_id], h_rec[c_id]))
            #mask = (inserted_img == 0)
            mask = cv2.imread(inserted['mask'])
            mask = cv2.resize(mask, (w_rec[c_id], h_rec[c_id]))
            temp_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = \
            temp_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] * (1 - mask) + inserted_img
            final_images.append(temp_img)
            final_c_ids.append(c_id)
    return final_images, final_c_ids

def build_pool(img_root, img_list, categories, pool_dir, mask_dir, is_loaded=False):
    global root_dir
    if is_loaded:
        with open(root_dir + 'pool.json', 'r') as f:
            pool = json.load(f)
    else:    
        pool = dict()
        cnt = 0
        progress = progressbar.ProgressBar(maxval=len(img_list)).start()
        for idx, img_name in enumerate(img_list):
            progress.update(idx + 1)
            img_id = int(img_name.split('.')[0].split('_')[-1])
            bbox_list, c_id_list, anno_list = get_bboxs(img_id, json_data)
            image = cv2.imread(img_root + img_name)
            for i in range(len(c_id_list)):
                bbox = bbox_list[i]
                c_id = c_id_list[i]
                anno = anno_list[i]
                mask = coco.annToMask(anno)
                temp_image = copy.deepcopy(image)
                for j in range(3):
                    temp_image[:, :, j] = temp_image[:, :, j] * mask + 255 * (1- mask)
                temp_image = temp_image[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
                mask = mask[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
                make_path(pool_dir + categories[c_id])
                make_path(mask_dir + categories[c_id])
                if temp_image.shape[0] and temp_image.shape[1]:
                    temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
                    Image.fromarray(temp_image).save(pool_dir + categories[c_id] + '/' + str(cnt) + '.png')
                    Image.fromarray(np.stack([mask] * 3, -1)).save(mask_dir + categories[c_id] + '/' + str(cnt) + '.png')
                    #cv2.imwrite(pool_dir + categories[c_id] + '/' + str(cnt) + '.png', temp_image)
                    if c_id in pool.keys():
                        pool[c_id].append({
                                'bbox': bbox,
                                'image': pool_dir + categories[c_id] + '/' + str(cnt) + '.png',
                                'mask': mask_dir + categories[c_id] + '/' + str(cnt) + '.png'
                            })
                    else:
                        pool[c_id] = [{
                            'bbox': bbox,
                            'image': pool_dir + categories[c_id] + '/' + str(cnt) + '.png',
                            'mask': mask_dir + categories[c_id] + '/' + str(cnt) + '.png'
                        }]
                    cnt += 1
        with open(root_dir + 'pool.json', 'w') as f:
            json.dump(pool, f)
    for k in pool.keys():
        pool[k] = sorted(pool[k], key=lambda x: x['bbox'][2] * x['bbox'][3])
        pool[k] = pool[k][int(len(pool[k])*0.8):]
        print('Class %s has %d objects' % (categories[k], len(pool[k])))
    progress.finish()
    return pool

# (W, H) = (233, 233)
# bbox_list, c_id_list = get_bboxs(img_id, json_data)
# for c_id in set(c_id_list):
#     img = inserted_bbox(c_id, bbox_list, c_id_list, W, H):
#     if img:
#         pass
#         # get question
#         # get img feature

def can_extract(idx, bbox, bbox_list):
    TAG = True
    for i in range(len(bbox_list)):
        if i != idx:
            if isBoxOverlap(bbox, bbox_list[i]):
                TAG = False
                break
    return TAG

def isEdgeOverlap(s1, e1, s2, e2):
    return not (e1 <= s2 or e2 <= s1)

def isBoxOverlap(box1, box2):
    # box: top-left (x, y, w, h)
    x_overlap = isEdgeOverlap(box1[0], box1[0]+box1[2], box2[0], box2[0]+box2[2])
    y_overlap = isEdgeOverlap(box1[1], box1[1]+box1[3], box2[1], box2[1]+box2[3])
    return x_overlap and y_overlap 

data_path = '/root/data/BUTD-config/data/genome/1600-400-20'
yaml_path = '/root/data/BUTD-config/configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml'
weights_path = '/root/collection/pretrained-models/BUTD/faster_rcnn_from_caffe_attr.pkl'

I2F = Image2Feat(
        data_path=data_path,
        yaml_path=yaml_path,
        weights_path=weights_path
    )

img_root = '/root/data/COCO/val2014/'
img_list = sorted(os.listdir(img_root))
json_path = '/root/data/COCO/annotations/instances_val2014.json'
out_dir = '/root/data/COCO/INSERT/image/'
root_dir = '/root/data/COCO/INSERT/'
feature_dir = '/root/data/COCO/INSERT/feature/BUTD/'
question_dir = '/root/data/COCO/INSERT/question/'
pool_dir = '/root/data/COCO/INSERT/pool/'
mask_dir = '/root/data/COCO/INSERT/mask/'

coco=COCO(json_path)

with open(json_path, 'r') as f:
    json_data = json.load(f)

categories = dict()
for c in json_data['categories']:
    categories[int(c['id'])] = c['name']

print('Building Pool...')
pool = build_pool(img_root, img_list, categories, pool_dir, mask_dir)
print('Finished!')

length = 2000

progress = progressbar.ProgressBar(maxval=length).start()
cur_idx = 0
for idx, img_name in enumerate(img_list):
    
    if cur_idx >= length:
        print('From %d Images' % idx)
        break
    progress.update(cur_idx + 1)

    img_path = img_root + img_name
    img_id = int(img_name.split('.')[0].split('_')[-1])
    bbox_list, c_id_list, _ = get_bboxs(img_id, json_data)
    c_name_list = [categories[c_id] for c_id in c_id_list]

    img = cv2.imread(img_path)
    (H, W, _) = img.shape
    
    instances, BUTD_features, _ = I2F.doit(img)
    pred = instances.to('cpu')

    class_list = [cls.item() for cls in pred.pred_classes]
    attr_list = [attr.item() for attr in pred.attr_classes]
    objects = dict()

    for c_name in c_name_list:
        objects[c_name] = {'attr': set()}

    # select classes in COCO
    for i in range(min(len(class_list), I2F.NUM_OBJECTS)):
        cls = str(I2F.get_class_name(class_list[i]))
        attr = str(I2F.get_attr_name(attr_list[i]))
        if cls in c_name_list:
            if cls in objects.keys():
                objects[cls]['attr'].add(attr)
            else:
                objects[cls] = {'attr': set([attr])}
    for k in objects.keys():
        objects[k]['attr'] = list(objects[k]['attr'])

    final_images, final_c_ids = insert_box(img, pool, bbox_list, c_id_list, W, H, k=2)

    if len(final_c_ids) == 0:
        continue
    cur_idx += 1

    final_Q_list = []
    inserted_feat_list = []
    for i in range(len(final_c_ids)):
        c_id = final_c_ids[i]
        final_image = final_images[i]
        c_name = categories[c_id]
        sub = str(img_name.split('.')[0])
        make_path(out_dir + sub)
        
        key = c_name
        value = objects.pop(key, None)
        # generate questions
        Question1 = {'add': [], 'not': [], 'equ': []}
        if len(objects.keys()) == 1:
            obj = list(objects.keys())[0]
            Question1['add'].append(
                {'Q': [I2F.template1 + obj + I2F.end],
                'C': False}
                )
        Question1 = I2F.generate_Q(objects, Question1, obj_th1=2, obj_th2=2)
        Question1 = I2F.add_Q2(Question1, key, False)
        
        Question2 = {'add': [], 'not': [], 'equ': []}
        Question2['add'].append(
            {'Q': [I2F.template1 + key + I2F.end],
            'C': False}
            )
        for obj in objects.keys():
            obj1 = key
            obj2 = obj
            if np.random.uniform() > 0.5:
                Question2 = I2F.add_Q1_Q3(Question2, obj1, obj2, False)
            else:
                Question2 = I2F.add_Q1_Q3(Question2, obj2, obj1, False)

        final_Q = {
            'same': Question1,
            'diff': Question2
        }

        final_Q_list.append(final_Q)
        objects[key] = value

        _, inserted_BUTD_features, _ = I2F.doit(final_image)
        inserted_feat_list.append(inserted_BUTD_features.to('cpu').numpy())
        cv2.imwrite(out_dir + sub + '/' + (sub+'-insert-%d' % i) + '.jpg', final_image)

    prefix = '.'.join(img_name.split('.')[:-1])
    suffix = '.npz'
    np.savez_compressed(feature_dir + prefix + suffix,
        origin=BUTD_features.to('cpu').numpy(),
        insert=inserted_feat_list
        )
    suffix = '.json'
    with open(question_dir + prefix + suffix, 'w') as f:
        json.dump(final_Q_list, f)
progress.finish()