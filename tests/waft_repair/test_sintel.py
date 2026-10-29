"""Sintel scene boundary and original-RGB interface checks, without datasets."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from waft_repair.distributed_pipeline import _raw_case
from waft_repair.motion.data import read_spec
from waft_repair.motion.sintel import make_sintel_spec,TRAIN_SCENES,DEV_SCENES
from test_motion_native import fixture


def test_complete_sintel_spec_groups_both_renderings_in_the_same_scene_split(tmp_path):
    root=tmp_path/'sintel'/'training'
    # Metadata fixtures only: 796 TRAIN and245 dev physical pairs/rendering.
    for index,scene in enumerate(TRAIN_SCENES+DEV_SCENES):
        count=(45 if index<4 else 44) if scene in TRAIN_SCENES else 49
        for part in ('flow','clean','final'):(root/part/scene).mkdir(parents=True)
        for frame in range(1,count+2):
            for rendering in ('clean','final'):(root/rendering/scene/f'frame_{frame:04d}.png').write_bytes(b'image fixture')
            if frame<=count:(root/'flow'/scene/f'frame_{frame:04d}.flo').write_bytes(b'flow fixture')
    result=make_sintel_spec(root.parent,tmp_path/'spec')
    assert result['train_pairs']==1592 and result['dev_pairs']==490
    rows=json.loads(Path(result['spec']).read_text())['samples'];groups={}
    for row in rows:
        groups.setdefault(row['group'],set()).add(row['split'])
        assert '/clean/' in row['image1'] or '/final/' in row['image1']
    assert all(len(splits)==1 for splits in groups.values())
    assert {name for name,splits in groups.items() if splits=={'dev'}}==set(DEV_SCENES)


def test_raw_case_rebuilds_original_observations_and_separates_GT(tmp_path):
    torch.set_num_threads(1)
    system,one,two=fixture()
    for name,value in [('one',one),('two',two)]:
        Image.fromarray(value[0].permute(1,2,0).numpy().astype(np.uint8)).save(tmp_path/(name+'.png'))
    np.savez(tmp_path/'gt.npz',flow=np.zeros((57,61,2),np.float32),valid=np.ones((57,61),bool))
    spec=tmp_path/'images.json';spec.write_text(json.dumps(dict(schema='waft-native-images/v1',samples=[dict(
        id='synthetic/0',split='train',image1='one.png',image2='two.png',ground_truth='gt.npz',ground_truth_format='npz')])))
    row=read_spec(spec)[0]
    prepared,gt,valid,baseline=_raw_case(system,row,torch.device('cpu'))
    assert 'gt' not in prepared and 'valid' not in prepared and 'teacher' not in prepared
    assert len(baseline['flow'])==len(baseline['info'])==5
    assert gt.shape==(1,2,57,61) and valid.all() and not prepared['original'].requires_grad
    assert not torch.cuda.is_initialized()
