"""I use small synthetic examples to check my tracking code."""
import numpy as np
import pytest
from src.association import bbox_iou
from src.data import save_feature_cache,load_feature_cache,filter_target_ids
from src.simple_tracker import SimpleMOTTracker
from src.kalman_tracker import KalmanMOTTracker, CalibratedKalmanMOTTracker, run_tracker
from src.evaluation import evaluate_tracking, evaluate_known_correspondence
from src.corruption import create_detection_gap


def fixture_frames():
    result={}
    for t in range(1,9):
        result[t]={'boxes':np.array([[10+t,12,20,40],[100+t,25,20,40]],dtype=float),
                   'features':np.array([[1.,0.],[0.,1.]],dtype=float),
                   'gt_ids':np.array([7,9])}
    return result

def test_iou_and_single_frame_assignment():
    assert bbox_iou([0,0,10,10],[0,0,10,10]) == 1
    assert bbox_iou([0,0,10,10],[100,100,10,10]) == 0
    track=SimpleMOTTracker(method='fusion')
    out=track.update(1,fixture_frames()[1]['boxes'],fixture_frames()[1]['features'])
    assert len(set(out))==2

def test_kalman_preserves_identity_and_r_baseline():
    data=fixture_frames()
    raw=KalmanMOTTracker(method='kf_dinov3')
    cal=CalibratedKalmanMOTTracker(method='kf_dinov3',r_variance=16)
    for frame,obs in data.items():
        a=raw.update(frame,obs['boxes'],obs['features'])
        b=cal.update(frame,obs['boxes'],obs['features'])
        np.testing.assert_array_equal(a,b)
    outputs,_=run_tracker(data,'kf_motion')
    assert evaluate_tracking(outputs)['Identity Changes']==0
    assert evaluate_known_correspondence(outputs,data)['Contaminated_Tracks']==0

def test_feature_cache_and_filter(tmp_path):
    data=fixture_frames()
    path=tmp_path/'features.npz'
    save_feature_cache(path,data)
    loaded=load_feature_cache(path)
    assert list(loaded)==list(data)
    filtered=filter_target_ids(loaded,[9])
    assert len(filtered[1]['boxes'])==1
    assert loaded[1]['boxes'].shape==(2,4)

def test_gap_does_not_modify_reference():
    data=fixture_frames()
    corrupted=create_detection_gap(data,7,3,2)
    assert len(corrupted[3]['boxes'])==1
    assert len(data[3]['boxes'])==2
    assert len(corrupted[5]['boxes'])==2

def test_frames_must_be_consecutive():
    data=fixture_frames();del data[3]
    with pytest.raises(ValueError,match='consecutive'):
        run_tracker(data,'kf_motion')
