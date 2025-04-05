from collections import defaultdict
import numpy as np
import os
import cv2
import json

# const params
TIME_TO_KILL = 20
TRACKLET_FLOOR = 50
IOU_FLOOR = 0.5
CONFIDENCE = 0.5

def calc_iou(b1, b2):
    """Calculate IoU between two boxes in (x,y,w,h) format"""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    
    # calculate intersection coords
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    # TODO doublecheck these 2 calculations

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    union_area = w1*h1 + w2*h2 - inter_area
    
    return inter_area / union_area if union_area else 0

def predict(tracklet):
    """Predict next position using delta from last two positions"""
    if len(tracklet['hits']) < 2:
        return tracklet['hits'][-1]['bbox']
    
    # calculate center mvmt
    prev = tracklet['hits'][-1]['bbox']
    prev_prev = tracklet['hits'][-2]['bbox']
    
    cx = prev[0] + prev[2]/2
    cy = prev[1] + prev[3]/2
    pcx = prev_prev[0] + prev_prev[2]/2
    pcy = prev_prev[1] + prev_prev[3]/2
    
    dx = cx - pcx
    dy = cy - pcy
    
    return [prev[0] + dx, prev[1] + dy, prev[2], prev[3]]

def save_drawn_boxes(tracklet, img, img_id):
    """Draw bounding boxes and IDs on image"""
    if img is None:
        return
    
    last = tracklet['hits'][-1]
    if last['frame'] == img_id:
        x, y, w, h = [int(v) for v in last['bbox']]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, 
                    f'ID:{tracklet["tracklet_id"]}', 
                    (x, y - 10), 
                    cv2.FONT_HERSHEY_DUPLEX, 
                    0.5, (255, 0, 0), 2)
        cv2.putText(img, 
                    f'ID:{tracklet["tracklet_id"]}', 
                    (x, y - 10), 
                    cv2.FONT_HERSHEY_DUPLEX, 
                    0.5, (255, 0, 0), 2)

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    
    active = []
    terminated = []
    tracklet_counter = 1
    
    with open('2021-11-20_lunch_2_cam0.json') as f:
        data = json.load(f)
    
    frame_detections = defaultdict(list)
    for ann in data['annotations']:
        if ann['confidence'] >= CONFIDENCE:
            # convert bbox from x1, y1, x2, y2 to x, y, w, h
            x1, y1, x2, y2 = ann['bbox']
            w = x2 - x1
            h = y2 - y1
            converted_ann = ann.copy()
            converted_ann['bbox'] = [x1, y1, w, h]
            frame_detections[converted_ann['image_id']].append(converted_ann)
    
    for frame in sorted(data['images'], key=lambda x: x['id']):
        frame_id = frame['id']
        detections = frame_detections.get(frame_id, [])
        
        associated = [False] * len(detections)
        
        # update existing tracklets on 1st pass
        for tracklet in active:
            best_match = {'iou': 0, 'idx': -1}
            predicted_box = predict(tracklet)
            
            for idx, det in enumerate(detections):
                if not associated[idx]:
                    iou = calc_iou(predicted_box, det['bbox'])
                    if iou > best_match['iou']:
                        best_match['iou'] = iou
                        best_match['idx'] = idx
            
            if best_match['iou'] >= IOU_FLOOR and best_match['idx'] != -1:
                match = detections[best_match['idx']]
                tracklet['hits'].append({
                    'frame': frame_id,
                    'bbox': match['bbox'],
                    'hit_id': match['id']
                })
                tracklet['gaps'] = 0
                associated[best_match['idx']] = True
        
        # create new tracklets on 2nd pass
        for idx, hit in enumerate(detections):
            if not associated[idx]:
                active.append({
                    'tracklet_id': tracklet_counter,
                    'hits': [{
                        'frame': frame_id,
                        'bbox': hit['bbox'],
                        'hit_id': hit['id']
                    }],
                    'gaps': 0
                })
                tracklet_counter += 1
        
        # cleanup
        active = [tracklet for tracklet in active if tracklet['gaps'] <= TIME_TO_KILL]
        
        # generate visual results
        img_path = os.path.join('images', frame['file_name'].split('/')[-1])
        img = cv2.imread(img_path)
        if img is not None:
            for tracklet in active:
                save_drawn_boxes(tracklet, img, frame_id)
            cv2.imwrite(os.path.join('results', f'{frame_id}.jpg'), img)
    
    # output
    all_tracklets = active + terminated
    output = []
    for t in all_tracklets:
        if len(t['hits']) >= TRACKLET_FLOOR:
            for hit in t['hits']:
                output.append({
                    'object_id': t['tracklet_id'],
                    'image_id': hit['frame'],
                    'hit_id': hit['hit_id']
                })
    
    with open('output.json', 'w') as f:
        json.dump(output, f)