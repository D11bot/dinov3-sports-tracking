"""I load SoccerNet sequences and select frames from the training ZIP."""
from pathlib import Path
from collections import defaultdict
from zipfile import ZipFile
from PIL import Image
import numpy as np
import pandas as pd

def index_zip(zip_path):
    groups=defaultdict(lambda:{'images':[], 'gt':None})
    with ZipFile(zip_path) as z:
        for name in z.namelist():
            parts=Path(name).parts
            if 'img1' in parts and Path(name).suffix.lower() in ('.jpg','.jpeg','.png'):
                index=parts.index('img1');groups['/'.join(parts[:index])]['images'].append(name)
            if Path(name).name=='gt.txt' and 'gt' in parts:
                index=parts.index('gt');groups['/'.join(parts[:index])]['gt']=name
    return {k:v for k,v in groups.items() if v['images'] and v['gt']}

def extract_zip_sequence(zip_path, sequence, dest, num_frames=120):
    sequences=index_zip(zip_path)
    if sequence not in sequences:
        raise KeyError(f'{sequence} unavailable; examples: {list(sequences)[:5]}')
    dest=Path(dest);(dest/'img1').mkdir(parents=True,exist_ok=True)
    (dest/'gt').mkdir(parents=True,exist_ok=True)
    seq=sequences[sequence]
    with ZipFile(zip_path) as z:
        for path in sorted(seq['images'])[:num_frames]:
            (dest/'img1'/Path(path).name).write_bytes(z.read(path))
        (dest/'gt'/'gt.txt').write_bytes(z.read(seq['gt']))
    return dest

def read_ground_truth(sequence_dir):
    return pd.read_csv(Path(sequence_dir)/'gt'/'gt.txt',header=None)

def filter_target_ids(frame_data, excluded_ids=()):
    excluded=np.asarray(tuple(excluded_ids),dtype=int)
    out={}
    for frame,d in frame_data.items():
        keep=~np.isin(d['gt_ids'],excluded)
        out[frame]={k: d[k][keep].copy() for k in ('boxes','features','gt_ids')}
    return out

def save_feature_cache(path, frames):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={}
    for frame,d in frames.items():
        for key in ('boxes','features','gt_ids'):
            payload[f'{frame}_{key}']=d[key]
    np.savez_compressed(path,**payload)

def load_feature_cache(path):
    with np.load(path,allow_pickle=False) as npz:
        frames=sorted({int(k.split('_',1)[0]) for k in npz.files})
        return {f:{key:npz[f'{f}_{key}'].copy() for key in ('boxes','features','gt_ids')} for f in frames}
