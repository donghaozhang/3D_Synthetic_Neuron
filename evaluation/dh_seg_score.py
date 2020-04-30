import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import csv


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
    if threshold == 0:
        print(precision, recall, f1, TP, FN, FP)
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
    f1_metric_interval = eva_metric_array[2, :]
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
    fig = plt.figure()
    plt.xlabel('Threshold Interval/Precision')
    plt.ylabel('F1 Score/Recall')
    # plt.scatter(threshold_array, threshold_array)
    plt.plot(threshold_array, f1_metric_interval, "r", label='f1 score')
    # plt.plot(threshold_array, eva_metric_all[1, :], "g", label='recall')
    plt.plot(precision_metric_interval, recall_metric_interval, "g", label='precision recall curve')
    # plt.plot(threshold_array, eva_metric_all[0, :], "b", label='precision')
    plt.legend(loc="best")
    plt.savefig(curve_im_path)

def cal_std(input_list):
    arr = np.asarray(input_list)
    return np.std(arr)

curve_test_flag = False
# plot f1 score curve and precision-recall curve of test dataset
if curve_test_flag:
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

threshold_from_training_flag = False
# find the optimal threshold using training images
if threshold_from_training_flag:
    # find the best threshold from training files
    prefix = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/checkpoints/3d_unet_pixel/train_full'
    prefix_gt = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/datasets/datasets/fly/test_crop'
    allfiles = os.listdir(prefix)
    threshold_array = np.linspace(start=0, stop=1, num=10)
    for i, file in enumerate(allfiles):
        print(file, 'is being processed')
        filepath = os.path.join(prefix, file)
        pred_im = loadtiff3d(filepath) / 255
        gt_filepath = from_predpath_to_gtpath(predpath=filepath, prefix_gt_input=prefix_gt)
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

save_csv_flag = True
if save_csv_flag:
    # save best f1 score, precision and recall of test dataset
    prefix = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/checkpoints/3d_unet_pixel/test_full'
    prefix_gt = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/datasets/datasets/fly/test_crop'
    csvfile = '/home/donghao/Desktop/donghao_v2/3D_Synthetic_Neuron/checkpoints/3d_unet_pixel/test_curve'
    allfiles = os.listdir(prefix)
    manual_threshold = 0.15
    fields = ['filename', 'precision', 'recall', 'f1', 'TP', 'FN', 'FP']
    csvfilename = os.path.join(csvfile, 'evaluation_score.csv')
    print('csv file is saved at', csvfilename)
    outfile = open(csvfilename, "w")
    write_outfile = csv.writer(outfile)
    write_outfile.writerow(fields)
    # with open(csvfilename, 'w') as csvfile:
    #     csvfile.writerow(fields)
    precarr = []
    recallarr = []
    f1arr = []
    TParr = []
    FNarr = []
    FParr = []
    for i, file in enumerate(allfiles):
        filepath = os.path.join(prefix, file)
        pred_im = loadtiff3d(filepath) / 255
        gt_filepath = from_predpath_to_gtpath(predpath=filepath, prefix_gt_input=prefix_gt)
        print(gt_filepath)
        gt_im = loadtiff3d(gt_filepath) / 255
        precision, recall, f1, TP, FN, FP = compute_confusion(gt=gt_im,
                                                              pred=pred_im,
                                                              threshold=manual_threshold)
        row_content = [file, precision, recall, f1, TP, FN, FP]
        write_outfile.writerow(row_content)
        precarr.append(precision)
        recallarr.append(recall)
        f1arr.append(f1)
        TParr.append(TP)
        FNarr.append(FN)
        FParr.append(FP)
    write_outfile.writerow(['avg', sum(precarr)/(i+1), sum(recallarr)/(i+1),
                            sum(f1arr)/(i+1), sum(TParr)/(i+1), sum(FNarr)/(i+1), sum(FParr)/(i+1)])
    write_outfile.writerow(['std', cal_std(precarr), cal_std(recallarr),
                            cal_std(f1arr), cal_std(TParr), cal_std(FNarr), cal_std(FParr)])

