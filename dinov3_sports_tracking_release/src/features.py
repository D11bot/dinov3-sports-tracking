"""I extract frozen DINOv3 embeddings from player crops."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEFAULT_MODEL='facebook/dinov3-vits16-pretrain-lvd1689m'

def load_dinov3(model_id=DEFAULT_MODEL, token=None):
    from transformers import AutoImageProcessor, AutoModel
    proc=AutoImageProcessor.from_pretrained(model_id, token=token)
    model=AutoModel.from_pretrained(model_id, token=token)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    return proc,model.to(device).eval(),device

def extract_features(image, boxes, processor, model, device, batch_size=16):
    if not len(boxes):return np.empty((0,384),dtype=np.float32)
    crops=[]
    for x,y,w,h in boxes:
        x1=max(0,int(x));y1=max(0,int(y))
        x2=min(image.width,int(x+w));y2=min(image.height,int(y+h))
        if x2<=x1 or y2<=y1:raise ValueError('Invalid crop')
        crops.append(image.crop((x1,y1,x2,y2)))
    result=[]
    for start in range(0,len(crops),batch_size):
        inputs=processor(images=crops[start:start+batch_size],return_tensors='pt').to(device)
        with torch.inference_mode():
            e=F.normalize(model(**inputs).pooler_output,dim=-1)
        result.append(e.cpu().numpy())
    return np.concatenate(result,axis=0)

def extract_sequence(sequence_dir, processor, model, device, num_frames=120, batch_size=16):
    from .data import read_ground_truth
    sequence_dir=Path(sequence_dir); gt=read_ground_truth(sequence_dir)
    result={}
    for frame_id in range(1,num_frames+1):
        path=sequence_dir/'img1'/f'{frame_id:06d}.jpg'
        if not path.exists():
            raise FileNotFoundError(f'Expected consecutive frame: {path}')
        image=Image.open(path).convert('RGB')
        rows=gt[gt.iloc[:,0]==frame_id]
        boxes=[];ids=[]
        for _,row in rows.iterrows():
            if int(row.iloc[6])!=1:continue
            x,y,w,h=map(float,row.iloc[2:6]);x1=max(0,int(x));y1=max(0,int(y))
            x2=min(image.width,int(x+w));y2=min(image.height,int(y+h))
            if x2<=x1 or y2<=y1:continue
            boxes.append([x1,y1,x2-x1,y2-y1]);ids.append(int(row.iloc[1]))
        boxes=np.array(boxes,dtype=float).reshape(-1,4)
        result[frame_id]={'boxes':boxes,'features':extract_features(image,boxes,processor,model,device,batch_size),
                          'gt_ids':np.asarray(ids,dtype=int)}
    return result
