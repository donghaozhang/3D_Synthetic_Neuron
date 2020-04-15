import torch
import os
import numpy as np
import matplotlib.pyplot as plt

def loadtiff3d(filepath):
    """Load a tiff file into 3D numpy array"""

    import tifffile as tiff
    a = tiff.imread(filepath)

    stack = []
    for sample in a:
        stack.append(np.rot90(np.fliplr(np.flipud(sample))))
    out = np.dstack(stack)

    return out


def compute_confusion(gt, pred, threshold=40):
    gt = (gt > 0).astype('int')
    pred = (pred > threshold).astype('int')
    TP = np.sum(pred[gt == 1])
    FP = np.sum(pred[gt == 0])
    FN = np.sum(gt[pred == 0])
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    if np.isnan(precision):
        precision = 0
        f1 = 0
    if np.isnan(recall):
        recall = 0
        f1 = 0
    return precision, recall, f1, TP, FN, FP


def compute_interval_confusion(threshold_array_input, gt_im_input, pred_im_input):
    eva_metric_array = []
    for counter, thres in enumerate(threshold_array_input):
        eva = np.asarray(compute_confusion(gt=gt_im_input,
                                           pred=pred_im_input,
                                           threshold=thres))
        eva = np.expand_dims(eva, axis=1)
        if counter == 0:
            eva_metric_array = eva
        else:
            eva_metric_array = np.concatenate((eva_metric_array, eva), axis=1)
    return eva_metric_array


def post_process_confusion(eva_metric_all_input):
    precision_metric_interval = eva_metric_all_input[0, :]
    # convert last element (precision, reccall) into (1, 0)
    precision_metric_interval[-1] = 1
    # insert element value 0 at index 0 for precision array
    precision_metric_interval = np.insert(precision_metric_interval, obj=0, values=0, axis=0)
    recall_metric_interval = eva_metric_all[1, :]
    # insert element value 0 at index 0 for recall array
    recall_metric_interval = np.insert(recall_metric_interval, obj=0, values=1, axis=0)
    f1_metric_interval = eva_metric_array[1, :]
    return precision_metric_interval, recall_metric_interval, f1_metric_interval


def from_predpath_to_gtpath(predpath, prefix_gt_input):
    predpath = predpath.split('/')[-1]
    im_num = predpath.split('_')[0]
    gt_file_name = im_num + '_gt.tif'
    gt_filepath = os.path.join(prefix_gt_input, gt_file_name)
    return gt_filepath


def from_metric_to_testcurve(eva_metric_all_input):
    curve_im_path = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/checkpoints/3d_unet_pixel/test_curve/test_curve'
    precision_metric_interval, recall_metric_interval, f1_metric_interval = post_process_confusion(
        eva_metric_all_input=eva_metric_all_input)
    plt.xlabel('Threshold Interval/Precision')
    plt.ylabel('F1 Score/Recall')
    # plt.scatter(threshold_array, threshold_array)
    plt.plot(threshold_array, f1_metric_interval, "r", label='f1 score')
    # plt.plot(threshold_array, eva_metric_all[1, :], "g", label='recall')
    plt.plot(precision_metric_interval, recall_metric_interval, "g", label='precision recall curve')
    # plt.plot(threshold_array, eva_metric_all[0, :], "b", label='precision')
    plt.legend(loc="best")
    plt.savefig(curve_im_path)

prefix = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/checkpoints/3d_unet_pixel/test_full'
prefix_gt = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/datasets/datasets/fly/test_crop'
allfiles = os.listdir(prefix)
threshold_array = np.linspace(start=0, stop=1, num=10)
for i, file in enumerate(allfiles):
    filepath = os.path.join(prefix, file)
    pred_im = loadtiff3d(filepath) / 255
    gt_filepath = from_predpath_to_gtpath(predpath=filepath, prefix_gt_input=prefix_gt)
    print(gt_filepath)
    gt_im = loadtiff3d(gt_filepath) / 255
    eva_metric_array = compute_interval_confusion(threshold_array_input=threshold_array,
                                                  gt_im_input=gt_im,
                                                  pred_im_input=pred_im)
    if i == 0:
        eva_metric_all = eva_metric_array
    else:
        eva_metric_all = eva_metric_all + eva_metric_array
eva_metric_all = eva_metric_all / (i + 1)
from_metric_to_testcurve(eva_metric_all_input=eva_metric_all)