"""I compare DINOv3 and IoU matching between two video frames."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from src.data import load_feature_cache
from src.association import compute_iou_matrix


def analyze_pair(a,b,alpha=0.5):
    ids_a=a['gt_ids'];ids_b=b['gt_ids']
    appearance=np.clip(a['features']@b['features'].T,-1,1)
    iou=compute_iou_matrix(a['boxes'],b['boxes'])
    common=set(ids_a)&set(ids_b)
    records=[]
    same=ids_a[:,None]==ids_b[None,:]
    if not len(ids_a) or not len(ids_b):
        return pd.DataFrame(),{}
    top1=np.argmax(appearance,axis=1)
    eligible=np.isin(ids_a,ids_b)
    top1_acc=float(np.mean(ids_b[top1][eligible]==ids_a[eligible])) if eligible.any() else np.nan
    for method,score in [('appearance',appearance),('iou',iou),('fusion',alpha*((appearance+1)/2)+(1-alpha)*iou)]:
        rows,cols=linear_sum_assignment(-score)
        assignments={int(ids_a[i]):int(ids_b[j]) for i,j in zip(rows,cols)}
        correct=sum(assignments.get(gid)==gid for gid in common)
        records.append({'Method':method,'Correct':correct,'Total_Common':len(common),
                        'Pair_Association_Accuracy':correct/len(common) if common else np.nan})
    stats={'top1_eligible_accuracy':top1_acc,
           'same_id_mean_similarity':float(appearance[same].mean()) if same.any() else np.nan,
           'different_id_mean_similarity':float(appearance[~same].mean()) if (~same).any() else np.nan}
    return pd.DataFrame(records),stats


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',required=True)
    p.add_argument('--reference',type=int,default=1)
    p.add_argument('--targets',type=int,nargs='+',default=[10,30,60,100])
    p.add_argument('--output',default='outputs/appearance_pair.csv')
    args=p.parse_args()
    data=load_feature_cache(args.cache);all_records=[]
    for target in args.targets:
        if target not in data or args.reference not in data:continue
        rows,stats=analyze_pair(data[args.reference],data[target])
        rows['Reference_Frame']=args.reference;rows['Target_Frame']=target
        all_records.append(rows)
        print('Frame',args.reference,'→',target,stats)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    if all_records:pd.concat(all_records).to_csv(path,index=False)

if __name__=='__main__':main()
