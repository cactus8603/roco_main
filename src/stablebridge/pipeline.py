"""One GT-free inference pipeline for stereo and flow.

The first implementation deliberately stores observed features, not accepted
displacements. An accepted output therefore cannot become its own evidence.
Unresolved observations stay available for later queries; legal memory access
does not certify geometric identity or make a corrupt observation trustworthy.
"""
from dataclasses import asdict, dataclass
import time
import cv2
import numpy as np

from .contracts import FrameEvidence, QueryIdentity, ReadBudget, TimeSnapshot
from .matching import FeatureRematcher, MatchConfig, query_and_anchor_masks
from .memory import EvidenceStore


@dataclass(frozen=True)
class PipelineConfig:
    mode: str = 'M0'
    memory_capacity: int = 8
    memory_bytes: int = 32 * 1024 * 1024
    memory_reads: int = 2
    retrieval_policy: str = 'diversity'
    local_operators: tuple = ('median3', 'gaussian1')
    diagnostic_reverse_pairs: bool = False
    diagnostic_augmentations: tuple = ()
    diagnostic_action_native_evidence: bool = False

    def __post_init__(self):
        if self.mode not in ('M0', 'M1'):
            raise ValueError('Only pair M0 and causal M1 are implemented')
        if self.retrieval_policy not in ('fifo', 'quality', 'diversity', 'query_conditioned'):
            raise ValueError('Unknown retrieval policy')
        if any(op not in ('median3', 'gaussian1') for op in self.local_operators):
            raise ValueError('Unknown local operator')
        if not isinstance(self.diagnostic_reverse_pairs,bool):
            raise ValueError('diagnostic_reverse_pairs must be boolean')
        if any(name not in ('roll_x8', 'roll_y8') for name in self.diagnostic_augmentations):
            raise ValueError('Unknown diagnostic augmentation')
        if len(set(self.diagnostic_augmentations)) != len(self.diagnostic_augmentations):
            raise ValueError('Diagnostic augmentations must be unique')
        if not isinstance(self.diagnostic_action_native_evidence,bool):
            raise ValueError('diagnostic_action_native_evidence must be boolean')


def local_operator(image, name):
    if name == 'median3':
        return cv2.medianBlur(image, 3)
    if name == 'gaussian1':
        return cv2.GaussianBlur(image, (5, 5), 1.)
    raise ValueError(name)


def _orient_reverse_displacement(task, displacement):
    """Map a reverse-input prediction back to the original target grid.

    Flow is inferred directly as image1->image0.  Stereo first horizontally
    flips the swapped pair so the frozen left-to-right checkpoint sees its
    trained epipolar direction; unflipping changes the sign of x displacement.
    """
    field=np.asarray(displacement,np.float32)
    if field.ndim != 3 or field.shape[0] != 2 or not np.isfinite(field).all():
        raise ValueError('Reverse displacement must be finite 2xHxW')
    if task=='flow':return field.copy()
    if task!='stereo':raise ValueError('task must be stereo or flow')
    output=field[:,:,::-1].copy();output[0]*=-1
    return output


def _predict_reverse(backbone, task, image0, image1, origin_xy):
    if task=='stereo':
        source=np.ascontiguousarray(image1[:,::-1]);target=np.ascontiguousarray(image0[:,::-1])
    else:
        source,target=image1,image0
    prediction=backbone.predict(source,target,origin_xy=origin_xy)
    return _orient_reverse_displacement(task,prediction.displacement),prediction.metadata


def _augmentation_shift(name):
    """Return an exact shared image translation as native (dx, dy)."""
    shifts={'roll_x8':(8,0),'roll_y8':(0,8)}
    if name not in shifts:raise ValueError(name)
    return shifts[name]


def _augment_pair(image0, image1, name):
    """Apply an invertible shared permutation; wrapped borders are later masked."""
    dx,dy=_augmentation_shift(name)
    return (np.roll(image0,(dy,dx),axis=(0,1)).copy(),
            np.roll(image1,(dy,dx),axis=(0,1)).copy())


def _restore_augmented_displacement(displacement, name):
    """Map a shared-translation prediction back to the original source grid."""
    field=np.asarray(displacement,np.float32)
    if field.ndim != 3 or field.shape[0] != 2 or not np.isfinite(field).all():
        raise ValueError('Augmented displacement must be finite 2xHxW')
    dx,dy=_augmentation_shift(name)
    return np.roll(field,(-dy,-dx),axis=(1,2)).copy()


class StableBridge:
    def __init__(self, backbone, config=None, match_config=None, acceptor=None):
        self.backbone = backbone
        self.config = config or PipelineConfig()
        self.rematcher = FeatureRematcher(match_config or MatchConfig())
        self.acceptor = acceptor
        self.memory = EvidenceStore(capacity=self.config.memory_capacity,
                                    max_feature_bytes=self.config.memory_bytes,
                                    mode=self.config.mode)
        self.sequence = None
        self.last_frame = None

    def reset(self, sequence):
        self.sequence = str(sequence)
        self.memory.reset(self.sequence)
        self.last_frame = None

    def _solved_donor(self, prepared, u0, local_fields, anchors, query_mask):
        """Repair only donor sites that the fixed neighbor bank will consult."""
        h, w = anchors.shape
        needed = np.zeros_like(anchors)
        qy,qx = np.nonzero(query_mask)
        for dx,dy in self.rematcher.config.source_neighbors:
            xx,yy=qx+dx,qy+dy
            inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
            needed[yy[inside],xx[inside]]=True
        y, x = np.nonzero(anchors & needed)
        donor = u0.copy()
        if not len(x) or not local_fields:
            return donor, 0
        xy = np.stack((x, y), -1).astype(np.float32)
        fields = [u0] + list(local_fields)
        candidates = np.stack([field[:, y, x].T for field in fields])
        baseline = candidates[0]
        raw, valid, _, _, delta = self.rematcher.score(prepared, xy, candidates, baseline)
        scores = raw + self.rematcher.config.movement_penalty * delta
        eligible = valid & (delta <= self.rematcher.config.max_update_px)
        scores[~eligible] = np.inf
        best = scores.argmin(0)
        ids = np.arange(len(x))
        change = (best != 0) & valid[0] & (scores[best, ids] + self.rematcher.config.min_improvement < scores[0])
        donor[:, y[change], x[change]] = candidates[best[change], ids[change]].T
        return donor, int(change.sum())

    def predict(self, pair, *, arms=('solved',), compute_reference=True, return_diagnostic_state=False):
        if 'gt' in pair or pair.get('metadata', {}).get('gt_read'):
            raise ValueError('Inference must not receive labels')
        allowed = {'generic', 'raw', 'solved', 'wrong_support', 'temporal', 'temporal_wrong', 'temporal_duplicate'}
        if not set(arms) <= allowed:
            raise ValueError('Unknown experimental arm')
        metadata = pair['metadata']; task = metadata['task']; frame = metadata['source_frame']
        if self.sequence is None:
            raise ValueError('Call reset with task/scene/condition/checkpoint namespace first')
        if self.last_frame is not None and frame <= self.last_frame:
            raise ValueError('Sequence inference must use strictly increasing source frames')
        if task != self.backbone.task:
            raise ValueError('Task and frozen checkpoint differ')
        if (self.config.diagnostic_reverse_pairs or self.config.diagnostic_augmentations or
                self.config.diagnostic_action_native_evidence) and not return_diagnostic_state:
            raise ValueError('Pair diagnostics require ephemeral diagnostic state')
        started = time.perf_counter()
        im0, im1 = pair['image0'], pair['image1']
        origin = tuple(pair['origin_xy'])
        pred = self.backbone.predict(im0, im1, origin_xy=origin)
        local_fields = [];local_predictions=[];processed_pairs=[(im0,im1)]
        forward_seconds = pred.metadata.get('forward_seconds', 0.)
        for op in self.config.local_operators if compute_reference else ():
            processed=(local_operator(im0,op),local_operator(im1,op));processed_pairs.append(processed)
            local = self.backbone.predict(*processed,origin_xy=origin)
            local_predictions.append(local)
            local_fields.append(local.displacement)
            forward_seconds += local.metadata.get('forward_seconds', 0.)
        reverse_fields=[];reverse_seconds=0.
        if self.config.diagnostic_reverse_pairs:
            for source,target in processed_pairs:
                reverse,metadata_reverse=_predict_reverse(self.backbone,task,source,target,origin)
                reverse_fields.append(reverse)
                reverse_seconds+=metadata_reverse.get('forward_seconds',0.)
            forward_seconds+=reverse_seconds
        augmentation_fields={};augmentation_seconds=0.
        for augmentation in self.config.diagnostic_augmentations:
            fields=[]
            for source,target in processed_pairs:
                augmented=_augment_pair(source,target,augmentation)
                prediction=self.backbone.predict(*augmented,origin_xy=origin)
                fields.append(_restore_augmented_displacement(prediction.displacement,augmentation))
                augmentation_seconds+=prediction.metadata.get('forward_seconds',0.)
            augmentation_fields[augmentation]=tuple(fields)
        forward_seconds+=augmentation_seconds
        prepared = self.rematcher.prepare(pred, im0, im1)
        action_prepared=()
        if self.config.diagnostic_action_native_evidence:
            action_prepared=(prepared,)+tuple(
                self.rematcher.prepare(prediction,*images)
                for prediction,images in zip(local_predictions,processed_pairs[1:]))
        mask, anchors, risk = query_and_anchor_masks(pred.displacement, pred.raw_uncertainty, im0, im1, self.rematcher.config)
        # Compute support repair once and charge it separately to solved arms.
        donor_start = time.perf_counter()
        solved, donor_changes = self._solved_donor(prepared, pred.displacement, local_fields, anchors, mask)
        donor_seconds = time.perf_counter() - donor_start
        h, w = im0.shape[:2]
        query = QueryIdentity(self.sequence, metadata['source_view'], frame,
                              metadata['target_view'], metadata['target_frame'],
                              roi=(origin[0], origin[1], w, h),
                              metadata={'descriptor': prepared['source'].mean(axis=(1, 2)).tolist()})
        snapshot = TimeSnapshot(query.cutoff, mode=self.config.mode)
        retrieved = self.memory.retrieve(query, snapshot,
                         ReadBudget(max_reads=self.config.memory_reads if self.config.mode == 'M1' else 0,
                                    max_read_bytes=self.config.memory_bytes),
                         policy=self.config.retrieval_policy)
        records = tuple(record for record in retrieved.records if record.frame < frame)
        results = {}
        for arm in arms:
            arm_start = time.perf_counter()
            donor = solved if arm != 'raw' else pred.displacement
            if arm == 'wrong_support':
                donor = np.roll(solved, w//3, axis=2)
            history = records if arm.startswith('temporal') else ()
            if arm == 'temporal_duplicate':
                history = records + records
            if arm == 'temporal_wrong':
                from dataclasses import replace
                history = tuple(replace(r, payload=replace(r.payload,
                           features=-r.payload.features)) for r in records)
            results[arm] = self.rematcher.refine(pred, im0, im1, task,
                              prepared=prepared, mask=mask, anchors=anchors, donor=donor,
                              local_candidates=local_fields, temporal_records=history,
                              origin_xy=origin, arm=arm, acceptor=self.acceptor)
            results[arm]['diagnostics'].update({'seconds': time.perf_counter()-arm_start,
                                                'donor_repair_seconds': donor_seconds if arm != 'raw' else 0.,
                                                'donor_changed_pixels': donor_changes if arm != 'raw' else 0})
        # Write after publication. Only raw observations are retained, never
        # output vectors, GT, or compressed purported independent votes.
        if self.config.mode == 'M1':
            observation = f'{self.sequence}/{metadata["source_view"]}/{frame}'
            feature = prepared['source']
            # Absolute current-pair cost is a heuristic quality index. Global
            # means of percentile ranks would be constant and cannot rank frames.
            yy,xx=np.indices((h,w));sample_xy=np.stack((xx[::16,::16].ravel(),yy[::16,::16].ravel()),-1).astype(np.float32)
            sampled_u=pred.displacement[:,::16,::16].reshape(2,-1).T
            paired_cost,valid,_,_,_=self.rematcher.score(prepared,sample_xy,sampled_u[None],sampled_u)
            quality=float(np.mean(np.where(valid,np.exp(-np.minimum(paired_cost,20)),0)))
            self.memory.add(FrameEvidence(self.sequence, frame, metadata['source_view'], feature,
                origin_xy=origin, image_hw=(h,w), quality=quality,
                source_groups=(observation,), observation_ids=(observation,),
                availability=query.cutoff,
                metadata={'descriptor': feature.mean(axis=(1,2)).tolist(),
                          'feature_schema': 'croco_encoder_norm_shared_projection_v1',
                          'feature_spec': prepared['feature_spec'],
                          'matching_quality_is_not_calibrated_probability': True}))
        self.last_frame = frame
        augmentation_calls=sum(len(fields) for fields in augmentation_fields.values())
        forward_calls=1+len(local_fields)+len(reverse_fields)+augmentation_calls
        output = {'baseline': pred.displacement, 'reference_predictions': [pred.displacement]+local_fields,
                'arms': results, 'query_mask': mask, 'anchor_mask': anchors,
                'risk': risk, 'costs': {'forward_calls':forward_calls,
                    'forward_seconds': forward_seconds, 'donor_seconds': donor_seconds,
                    'diagnostic_reverse_forward_calls':len(reverse_fields),
                    'diagnostic_reverse_forward_seconds':reverse_seconds,
                    'diagnostic_augmentation_forward_calls':augmentation_calls,
                    'diagnostic_augmentation_forward_seconds':augmentation_seconds,
                    'total_seconds': time.perf_counter()-started,
                    'retrieval': dict(retrieved.costs), 'memory': self.memory.cost_summary()},
                'query': asdict(query), 'snapshot': asdict(snapshot),
                'model': pred.metadata, 'input_metadata': metadata,
                'retained_output_propagation': False}
        if return_diagnostic_state:
            # Ephemeral, in-process state for explicitly labelled offline
            # diagnostics. It contains no GT and is removed before artifacts
            # are serialized.
            output['_diagnostic_state'] = {'prediction':pred,'prepared':prepared,
                'local_fields':tuple(local_fields),'image0':im0,'image1':im1,
                'origin_xy':origin,'runtime_anchors':anchors,
                'reverse_fields':tuple(reverse_fields),
                'augmentation_fields':augmentation_fields,
                'action_predictions':(pred,*local_predictions),
                'action_prepared':action_prepared}
        return output
