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

def calAveragePrecision(precision, recall):
    mprecision = np.concatenate(([0.], precision, [0.]))
    mrecall = np.concatenate(([0.], recall, [1.]))
    mprecision = np.flip(np.maximum.accumulate(np.flip(mprecision)))
    print("mprecision", len(mprecision))
    print("mrecall", len(mrecall))

    # 11 points sampling interpolation Pascal VOC 2012
    idxs = np.linspace(0, 1, 11)  
    averagePrecision = np.trapz(np.interp(idxs, mrecall, mprecision), idxs) 

    return averagePrecision

def calculate_ap_11_point_interp(rec, prec, recall_vals=11):
    mrec = []
    # mrec.append(0)
    [mrec.append(e) for e in rec]
    # mrec.append(1)
    mpre = []
    # mpre.append(0)
    [mpre.append(e) for e in prec]
    # mpre.append(0)
    recallValues = np.linspace(0, 1, recall_vals)
    recallValues = list(recallValues[::-1])
    rhoInterp = []
    recallValid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recallValues:
        # Obtain all recall values higher or equal than r
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        # If there are recalls above r
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])
        recallValid.append(r)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / len(recallValues)
    # Generating values for the plot
    rvals = []
    rvals.append(recallValid[0])
    [rvals.append(e) for e in recallValid]
    rvals.append(0)
    pvals = []
    pvals.append(0)
    [pvals.append(e) for e in rhoInterp]
    pvals.append(0)
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i in range(len(rvals)):
        p = (rvals[i], pvals[i - 1])
        if p not in cc:
            cc.append(p)
        p = (rvals[i], pvals[i])
        if p not in cc:
            cc.append(p)
    recallValues = [i[0] for i in cc]
    rhoInterp = [i[1] for i in cc]
    return [ap, rhoInterp, recallValues, None]

def calculate_ap_every_point(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

def plot_precision_recall_curves(results,
                                 showAP=False,
                                 showInterpolatedPrecision=False,
                                 savePath=None,
                                 showGraphic=True):
    result = None
    print(results)
    # Each resut represents a class
    for classId, result in results.items():
        if result is None:
            raise IOError(f'Error: Class {classId} could not be found.')

        precision = result['precision']
        recall = result['recall']
        average_precision = result['AP']
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']
        method = result['method']
        plt.close()
        # if showInterpolatedPrecision:
        #     if method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
        #         plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')
        #     elif method == MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION:
        #         # Remove duplicates, getting only the highest precision of each recall value
        #         nrec = []
        #         nprec = []
        #         for idx in range(len(mrec)):
        #             r = mrec[idx]
        #             if r not in nrec:
        #                 idxEq = np.argwhere(mrec == r)
        #                 nrec.append(r)
        #                 nprec.append(max([mpre[int(id)] for id in idxEq]))
        #         plt.plot(nrec, nprec, 'or', label='11-point interpolated precision')
        plt.plot(recall, precision, label='Precision')
        plt.xlabel('recall')
        plt.ylabel('precision')
        if showAP:
            ap_str = "{0:.2f}%".format(average_precision * 100)
            # ap_str = "{0:.4f}%".format(average_precision * 100)
            plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(classId), ap_str))
        else:
            plt.title('Precision x Recall curve \nClass: %s' % str(classId))
        plt.legend(shadow=True)
        plt.grid()

        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        if savePath is not None:
            plt.savefig(os.path.join(savePath, classId + '.png'))
        if showGraphic is True:
            plt.show()
            # plt.waitforbuttonpress()
            plt.pause(0.05)
    return results



wider_gt = "widerface_test_1000/wider_gt/"
wider_pred = "widerface_test_1000/wider_pred_face/"

list_wider_gt = os.listdir(wider_gt)
list_wider_pred = os.listdir(wider_pred)

threshold = 0.5
score_threshold = 0.5
detections = []
len_gt = 0
len_gt2 = 0

for file_gt in list_wider_gt:
    gt_boxes = read_gtbox_from_file(os.path.join(wider_gt, file_gt))
    len_gt += len(gt_boxes)
print(len_gt)

for file_pred in list_wider_pred:  
    file_pred_path = os.path.join(wider_pred, file_pred)
    file_gt_path = os.path.join(wider_gt, file_pred)
    # print(file_pred_path)
    pred_boxes, scores = read_predbox_from_file(file_pred_path)
    gt_boxes = read_gtbox_from_file(file_gt_path)

    # print(len_gt2)
 
    
    for pred_box, score in zip(pred_boxes, scores):
        score = float(score)
        max_iou = 0.0
        for gt_box in gt_boxes:
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou

        if score >= score_threshold and max_iou >= threshold:
            TP = 1
            FP = 0
            # status = "TP"
            
        else:
            TP = 0
            FP = 1
  

        detection = {
            "score": score,
            # "status": status,
            "TP": TP,
            "FP": FP
            
        }

        detections.append(detection)




sorted_detections = sorted(detections, key=lambda x: x["score"], reverse=True)
# print(sorted_detections)

acc_TP = 0
acc_FP = 0


for detection in sorted_detections:
    acc_TP += detection["TP"]
    acc_FP += detection["FP"]

    # if detection["status"] == "TP":
    #     AccTP += 1
    # else:
    #     AccFP += 1
    detection["AccTP"] = acc_TP
    detection["AccFP"] = acc_FP
    detection['precision'] = acc_TP / (acc_TP + acc_FP)
    detection['recall'] = acc_TP / len_gt


# for detection in sorted_detections:
#     print(detection)

df = pd.DataFrame(sorted_detections)
print(df)
# print(df[['score', 'TP', 'FP', 'AccTP', 'AccFP', 'precision', 'recall']])
    

precision_inter = []
# 11-point Interpolation
for i in range(0, 11):
    r = i / 10.0
    max_precision = None

    for detection in reversed(sorted_detections):
        recall = detection.get('recall')
        precision = detection.get('precision')
        if recall > r and (max_precision is None or precision > max_precision):
            max_precision = precision
    precision_inter.append(max_precision)

print(precision_inter)

float_array = [float(element) if element is not None  else 0.0 for element in precision_inter]
print("ap", np.mean(float_array))





precision = [dictionary["precision"] for dictionary in sorted_detections if "precision" in dictionary]
recall = [dictionary["recall"] for dictionary in sorted_detections if "recall" in dictionary]


# function np.trapz
ap = calAveragePrecision(precision, recall)
print(ap)


def plot_pr_curve(precisions, recalls):
    # plots the precision recall values for each threshold
    # and save the graph to disk
    plt.plot(recalls, precisions, linewidth=4, color="red")
    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    plt.show()

plot_pr_curve(precision, recall)



# reference

result = calculate_ap_11_point_interp(recall, precision, recall_vals=11)

print(result)

result_all = calculate_ap_every_point(recall, precision)

print(result_all[0])

