from glob import glob
import os.path as osp
import os
import pickle
import json
import numpy as np
downsample = 3
min_change_duration = 0.3

with open('../data/export/k400_mr345_val_min_change_duration'+str(min_change_duration)+'.pkl', 'rb') as f:
    gt_dict = pickle.load(f)

exp_path = '../data/exp_k400/'
output_seg_dir = 'detect_seg'
OUTPUT_BDY_PATH = exp_path + output_seg_dir + '/{}.pkl'
list_rec = []
list_prec = []
list_f1 = []

for d in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: # threshold vary from percentage of video duration

    tp_all = 0
    num_pos_all = 0
    num_det_all = 0
    
    for vid_id in list(gt_dict.keys()):
        
        # filter by avg_f1 score
        if gt_dict[vid_id]['f1_consis_avg']<0.3:
            continue

        output_bdy_path = OUTPUT_BDY_PATH.format(vid_id)

        if not os.path.exists(output_bdy_path):
            continue
        
        with open(output_bdy_path, 'rb') as f:
            bdy_idx_save = pickle.load(f, encoding='latin1') 

        bdy_idx_list_smt = np.array(bdy_idx_save['bdy_idx_list_smt'])*downsample # already offset, index starts from 0

        myfps = gt_dict[vid_id]['fps']
        ins_start = 0
        ins_end = gt_dict[vid_id]['num_frames']-1 #number of frames
        
        # remove detected boundary outside the action instance
        tmp = []
        for det in bdy_idx_list_smt:
            tmpdet = det + ins_start
            if tmpdet >= (ins_start) and tmpdet <= (ins_end):
                tmp.append(tmpdet)
        bdy_idx_list_smt = tmp
        if bdy_idx_list_smt == []:
            num_pos_all += len(gt_dict[vid_id]['substages_myframeidx'][0])
            continue
        num_det = len(bdy_idx_list_smt)
        num_det_all += num_det
            
        # compare bdy_idx_list_smt vs. each rater's annotation, pick the one leading the best f1 score
        bdy_idx_list_gt_allraters = gt_dict[vid_id]['substages_myframeidx']
        f1_tmplist = np.zeros(len(bdy_idx_list_gt_allraters))
        tp_tmplist = np.zeros(len(bdy_idx_list_gt_allraters))
        num_pos_tmplist = np.zeros(len(bdy_idx_list_gt_allraters))
        
        for ann_idx in range(len(bdy_idx_list_gt_allraters)):
            bdy_idx_list_gt = bdy_idx_list_gt_allraters[ann_idx]
            num_pos = len(bdy_idx_list_gt)
            tp = 0
            offset_arr = np.zeros((len(bdy_idx_list_gt), len(bdy_idx_list_smt))) 
            for ann1_idx in range(len(bdy_idx_list_gt)):
                for ann2_idx in range(len(bdy_idx_list_smt)):
                    offset_arr[ann1_idx, ann2_idx] = abs(bdy_idx_list_gt[ann1_idx]-bdy_idx_list_smt[ann2_idx])
            for ann1_idx in range(len(bdy_idx_list_gt)):
                if offset_arr.shape[1] == 0:
                    break
                min_idx = np.argmin(offset_arr[ann1_idx, :])
                if offset_arr[ann1_idx, min_idx] <= d*(ins_end-ins_start+1):
                    tp += 1
                    offset_arr = np.delete(offset_arr, min_idx, 1)   
            
            num_pos_tmplist[ann_idx] = num_pos
            fn = num_pos - tp
            fp = num_det - tp
            if num_pos == 0: 
                rec = 1
            else:
                rec = tp/(tp+fn)
            if (tp+fp) == 0: 
                prec = 0
            else: 
                prec = tp/(tp+fp)
            if (rec+prec) == 0:
                f1 = 0
            else:
                f1 = 2*rec*prec/(rec+prec)            
            tp_tmplist[ann_idx] = tp
            f1_tmplist[ann_idx] = f1
            
        ann_best = np.argmax(f1_tmplist)
        tp_all += tp_tmplist[ann_best]
        num_pos_all += num_pos_tmplist[ann_best]
        
    fn_all = num_pos_all - tp_all
    fp_all = num_det_all - tp_all
    if num_pos_all == 0:
        rec = 1
    else:
        rec = tp_all/(tp_all+fn_all)
    if (tp_all+fp_all) == 0:
        prec = 0
    else:
        prec = tp_all/(tp_all+fp_all)
    if (rec+prec) == 0:
        f1 = 0
    else:
        f1 = 2*rec*prec/(rec+prec)
    list_rec.append(rec); list_prec.append(prec); list_f1.append(f1)

print("rec: " + str(np.mean(list_rec))) 
print("prec: " + str(np.mean(list_prec))) 
print("F1: " + str(np.mean(list_f1))) 

print("rec: " + str(list_rec))
print("prec: " + str(list_prec))
print("F1: " + str(list_f1)) 
    
np.save(exp_path + output_seg_dir + '.eval.mindur'+str(min_change_duration)+'.npy', [list_rec, list_prec, list_f1]) 
