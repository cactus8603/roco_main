"""Explicit scene split and original Clean/Final RGB pairs, without features."""
from pathlib import Path

from .io import atomic_json, require

DEV_SCENES=('bandage_1','bandage_2','mountain_1','shaman_2','shaman_3')
TRAIN_SCENES=('alley_1','alley_2','ambush_2','ambush_4','ambush_5','ambush_6',
              'ambush_7','bamboo_1','bamboo_2','cave_2','cave_4','market_2',
              'market_5','market_6','sleeping_1','sleeping_2','temple_2','temple_3')


def make_sintel_spec(sintel_root,output):
    """Same fixed 18 TRAIN/5 dev scenes for both renderings; dev is eval-only."""
    root=Path(sintel_root).resolve(strict=True)
    if (root/'training').is_dir():root=root/'training'
    require({p.name for p in (root/'flow').iterdir() if p.is_dir()}==set(TRAIN_SCENES+DEV_SCENES),
            'Expected the complete original 23 Sintel training scenes')
    samples=[]
    for rendering in ('clean','final'):
        for split,scenes in (('train',TRAIN_SCENES),('dev',DEV_SCENES)):
            for scene in scenes:
                for flow in sorted((root/'flow'/scene).glob('frame_*.flo')):
                    index=int(flow.stem.split('_')[1])
                    first=(root/rendering/scene/f'frame_{index:04d}.png').resolve(strict=True)
                    second=(root/rendering/scene/f'frame_{index+1:04d}.png').resolve(strict=True)
                    samples.append(dict(id=f'sintel/{rendering}/{scene}/{flow.stem}',split=split,
                        image1=str(first),image2=str(second),ground_truth=str(flow.resolve(strict=True)),
                        ground_truth_format='sintel',group=scene))
    require(sum(row['split']=='train' for row in samples)==1592
            and sum(row['split']=='dev' for row in samples)==490,'Full Sintel1592 TRAIN/490 dev required')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'images.json',dict(schema='waft-native-images/v1',samples=samples))
    atomic_json(output/'split.json',dict(train_scenes=TRAIN_SCENES,dev_scenes=DEV_SCENES,
        renderings=['clean','final'],support_policy='finite_abs_lt_1000',dev_policy='evaluation-only'))
    return dict(spec=str(output/'images.json'),train_pairs=1592,dev_pairs=490)
