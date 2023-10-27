import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_predbox_from_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    bboxes = []
    scores = []
    for line in lines:
        parts = line.strip().split() 
        if len(parts) > 0:
            bbox = [int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])]
            bboxes.append(bbox)
            scores.append(parts[1])
    return bboxes, scores

def read_gtbox_from_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    bboxes = []
    for line in lines:
        parts = line.strip().split() 
        bbox = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]
        bboxes.append(bbox)
    return bboxes

def calculate_iou(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def calculate_precision_recall(predict_boxes_folder, groundtruth_boxes_folder, iou_threshold=0.5, score_threshold=0.5):
    detections = []
    len_gt = 0
    list_pred = os.listdir(predict_boxes_folder)
    for file_pred in list_pred: 
        predict_boxes_path = os.path.join(predict_boxes_folder, file_pred)
        groundtruth_boxes_path = os.path.join(groundtruth_boxes_folder, file_pred)
        pred_boxes, scores = read_predbox_from_file(predict_boxes_path)
        gt_boxes = read_gtbox_from_file(groundtruth_boxes_path)
        len_gt += len(gt_boxes)
        for pred_box, score in zip(pred_boxes, scores):
            score = float(score)
            max_iou = 0.0
            for gt_box in gt_boxes:
                iou = calculate_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou

            if score >= score_threshold and max_iou >= iou_threshold:
                TP = 1
                FP = 0         
            else:
                TP = 0
                FP = 1

            detection = {
                "score": score,
                "TP": TP,
                "FP": FP
            }

            detections.append(detection)

    sorted_detections = sorted(detections, key=lambda x: x["score"], reverse=True)

    acc_TP = 0
    acc_FP = 0

    for detection in sorted_detections:
        acc_TP += detection["TP"]
        acc_FP += detection["FP"]

        detection["AccTP"] = acc_TP
        detection["AccFP"] = acc_FP
        detection['precision'] = acc_TP / (acc_TP + acc_FP)
        detection['recall'] = acc_TP / len_gt

    return sorted_detections

def calculate_AP_11_points(sorted_detections):
  precision_inter = []

  for i in range(0, 11):
      r = i / 10.0
      max_precision = None

      for detection in reversed(sorted_detections):
          recall = detection.get('recall')
          precision = detection.get('precision')
          if recall > r and (max_precision is None or precision > max_precision):
              max_precision = precision
      precision_inter.append(max_precision)

  prec = [float(element) if element is not None  else 0.0 for element in precision_inter]
  ap = np.mean(prec)
  return ap

if __name__ == '__main__':
    wider_gt = "widerface_test_1000/wider_gt/"
    wider_pred = "widerface_test_1000/wider_pred_face/"
    
    sorted_detections = calculate_precision_recall(wider_pred, wider_gt)
    ap = calculate_AP_11_points(sorted_detections)
    print(ap)
