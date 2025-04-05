from collections import defaultdict
import numpy as np
import os
import cv2
from PIL import Image
from torchvision.ops import masks_to_boxes
import torch
import json

TIME_TO_KILL = 20
TRACKLET_FLOOR = 50
IOU_FLOOR = 0.5
CONFIDENCE = 0.5

def multi_compare(p1, p2):
    """Comparaison de multiples histogrammes a la fois"""
    res = [cv2.compareHist(hp1, hp2) for hp1 in p1 for hp2 in p2]

    return max(res)

def calc_iou(b1, b2):
    """Calculer le IoU entre deux box (x, y, w, h)"""

    max_coord1 = (b1[0] + b1[2], b1[1] + b1[3])
    max_coord2 = (b2[0] + b2[2], b2[1] + b2[3])

    intersect = max(min(max_coord1[0], max_coord2[0]) - max(b1[0], b2[0]), 0) * max(min(max_coord1[1], max_coord2[1]) - max(b1[1], b2[1]), 0)
    union = ((b1[2] * b1[3]) + (b2[2] * b2[3])) - intersect

    return intersect / union if union else 0

def predict(tracklet):
    """Return the last box translated by delta"""
    if len(tracklet['hits']) < 2: return tracklet['hits'][-1]['bbox']

    prev2 = (tracklet['hits'][-1]['bbox'], tracklet['hits'][-2]['bbox'])

    prev2_centres = (
        (prev2[0][0] + prev2[0][2] / 2, prev2[0][1] + prev2[0][3] / 2),
        (prev2[1][0] + prev2[1][2] / 2, prev2[1][1] + prev2[1][3] / 2)
    )

    delta = (prev2_centres[0][0] - prev2_centres[1][0], prev2_centres[0][1] - prev2_centres[1][1])

    return [prev2[0][0] + delta[0], prev2[0][1] + delta[1], prev2[0][2], prev2[0][3]]

def save_drawn_boxes(tracklet, img, img_id):

    if img is None:
        return f'Skipping: Unable to read image'
    
    last = tracklet['hits'][-1]
    if last['frame'] == img_id:
        cv2.rectangle(
            img,
            (int(last['bbox'][0]), int(last['bbox'][1])),
            (int(last['bbox'][2]), int(last['bbox'][3])),
            (255, 0, 0),
            1
        )
        cv2.putText(
            img,
            f'Tracklet #{tracklet["tracklet_id"]}',
            (int(last['bbox'][0]), int(last['bbox'][1] - 3)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 0, 0),
            1
        )


if __name__ == "__main__":
    
    res, frames = [], defaultdict(list)
    active, terminated, iter = [], [], 1
    
    input_file = '2021-11-20_lunch_2_cam0.json'
    with open(input_file, 'r') as f:
        read_annotations = json.load(f)
    
    for hit in read_annotations['annotations']:
        if hit['confidence'] >= CONFIDENCE: frames[hit['image_id']].append(hit)
    
    sort = sorted(read_annotations['images'], key = lambda x: x['id'])

    for frame in sort:

        id = frame['id']
        hits = frames.get(id, [])
        
        associated = [False for hit in hits]

        count_preprocessing = len(active)

        for tracklet in active:

            best = {
            'iou' : 0,
            'idx': -1
            }

            box_predicted = predict(tracklet)
            for idx, hit in enumerate(hits):
                if not associated[idx]:
                    iou = calc_iou(box_predicted, hit['bbox'])
                    if iou > best['iou']:
                        best['iou'] = iou
                        best['idx'] = idx

            if best['iou'] >= IOU_FLOOR and best['idx'] != -1:
                hit = hits[best['idx']]
                tracklet['hits'].append({'frame': id, 'bbox': hit['bbox'], 'hit_id': hit['id']})
                tracklet['last'], tracklet['gaps'], associated[best['idx']] = id, 0, True
            else:
                tracklet['gaps'] += 1
                suffix = 'aucun hit compatible' if best['iou'] <= 0 else 'IoU en dessous du seuil'
                msg = f'{tracklet["tracklet_id"]} non associe: ' + suffix
                print(msg)
        
        prune = []
        for tracklet in active:
            if tracklet['gaps'] >= TIME_TO_KILL:
                terminated.append(tracklet)
                print(f'{tracklet["tracklet_id"]} terminated: reached time to kill')
            else:
                prune.append(tracklet)
        active = prune

        for idx, hit in enumerate(hits):
            if not associated[idx]:
                active.append({
                    'tracklet_id': iter,
                    'hits': [{
                        'frame': id,
                        'bbox': hit['bbox'],
                        'hit_id': hit['id']
                    }],
                    'gaps': 0,
                    'last': id
                })
                print(f'New tracklet generated: #{iter} for {hit["id"]}')
                iter += 1
        print(f'''!DEBUG!\n
              Post-processing tracklet count: {count_preprocessing} - {len(active)} = {count_preprocessing - len(active)} 
              ''')
        
        file_name = frame['file_name'].split('/')[-1]
        img_path = os.path.join('images', file_name)
        img = cv2.imread(img_path)

        for tracklet in active:
            if len(tracklet['hits']):
                save_drawn_boxes(tracklet, img=img, img_id=id)
        
        out = os.path.join('results', f'{id}.jpg')
        if img is None:
            print(f"Skipping write: Unable to read image at {img_path}")
        else:
            cv2.imwrite(out, img)

    
    terminated += active

    valid = []
    for tracklet in terminated:
        if len(tracklet['hits']) >= TRACKLET_FLOOR:
            retain = tracklet['hits'][0]
            valid.append({
                'img': retain['frame'],
                'tracklet_id': tracklet['tracklet_id'],
                'hit_id': retain['hit_id']
            })
    out = {"Retained tracklets: ": valid}
    with open('output.json', 'w') as f:
        json.dump(out, f)