"""GT-free candidate generation and real candidate-conditioned feature sampling.

This is an explicit research baseline, not a calibrated safety guarantee. Dense
output is preserved while query work is bounded. Features are frozen CroCo
encoder features compressed with a deterministic, shared random projection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from scipy.stats import rankdata
import torch
import torch.nn.functional as F


CANDIDATE_FEATURE_SCHEMA = 'candidate_features_v1_12'


@dataclass(frozen=True)
class MatchConfig:
    feature_dim: int = 64
    projection_seed: int = 20260916
    feature_stride: int = 16
    max_queries: int = 4096
    query_fraction: float = .1
    anchor_quantile: float = .5
    max_candidates: int = 24
    source_neighbors: tuple = ((-16, 0), (16, 0), (0, -16), (0, 16), (-32, 0), (32, 0))
    feature_weight: float = .5
    movement_penalty: float = .001
    min_improvement: float = .015
    max_update_px: float = 32.
    require_baseline_support: bool = True
    chunk_size: int = 512

    def __post_init__(self):
        if not 0 < self.query_fraction <= 1 or not 0 < self.anchor_quantile < 1:
            raise ValueError('Invalid mask fractions')
        if min(self.feature_dim, self.feature_stride, self.max_queries, self.max_candidates, self.chunk_size) < 1:
            raise ValueError('Budgets must be positive')
        if not 0 <= self.feature_weight <= 1 or self.max_update_px <= 0 or self.min_improvement < 0 or self.movement_penalty < 0:
            raise ValueError('Invalid score/acceptance configuration')
        if not all(np.isfinite(v) for v in (self.feature_weight, self.max_update_px,
                                            self.min_improvement, self.movement_penalty)):
            raise ValueError('Score/acceptance values must be finite')
        if any(len(offset) != 2 or any(not isinstance(v, (int, np.integer)) for v in offset)
               for offset in self.source_neighbors):
            raise ValueError('Source neighbor offsets must be integer (dx, dy)')


def compress_features(features, dimensions=64, seed=20260916):
    f = np.asarray(features, np.float32)
    if dimensions < 1 or f.ndim != 3 or min(f.shape) < 1 or not np.isfinite(f).all():
        raise ValueError('Features must be finite [C,H,W]')
    if f.shape[0] > dimensions:
        matrix = np.random.default_rng(seed).standard_normal((dimensions, f.shape[0])).astype(np.float32)
        matrix /= np.sqrt(f.shape[0])
        f = (matrix @ f.reshape(f.shape[0], -1)).reshape(dimensions, *f.shape[1:])
    return f / np.maximum(np.linalg.norm(f, axis=0, keepdims=True), 1e-8)


def image_descriptor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    pad = np.pad(gray, 1, mode='edge')
    h, w = gray.shape
    channels = []
    for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
        delta = pad[1+dy:1+dy+h, 1+dx:1+dx+w] - gray
        channels.append(delta / np.sqrt(delta * delta + .04 ** 2))
    return np.stack(channels)


def _sample(feature, xy, stride=1):
    """Local pixel xy -> feature center coordinates. Unsupported samples masked."""
    f = torch.as_tensor(feature, dtype=torch.float32)
    if f.ndim != 3:
        raise ValueError('Expected CHW feature')
    coords = torch.as_tensor(xy, dtype=torch.float32, device=f.device)
    shape = coords.shape[:-1]
    ix = (coords[..., 0] + .5) / stride - .5
    iy = (coords[..., 1] + .5) / stride - .5
    hf, wf = f.shape[1:]
    valid = torch.isfinite(coords).all(-1) & (ix >= 0) & (ix <= wf-1) & (iy >= 0) & (iy <= hf-1)
    grid = torch.stack((2*(ix+.5)/wf-1, 2*(iy+.5)/hf-1), -1)
    grid = torch.nan_to_num(grid, nan=2., posinf=2., neginf=-2.)
    values = F.grid_sample(f[None], grid.reshape(1, 1, -1, 2), align_corners=False, padding_mode='zeros')
    values = values[0, :, 0].T.reshape(*shape, f.shape[0])
    return values.numpy(), valid.numpy()


def _rank01(array):
    values = np.asarray(array)
    if not np.isfinite(values).all():
        raise ValueError('Risk ranking requires finite observations')
    if values.size < 2:
        return np.full(values.shape, .5, np.float32)
    # Equal observations carry equal risk; raster order is not uncertainty.
    return ((rankdata(values.ravel(), method='average') - 1) / (values.size - 1)).astype(np.float32).reshape(values.shape)


def query_and_anchor_masks(u0, uncertainty, image0, image1, config):
    h, w = u0.shape[1:]
    yy, xx = np.indices((h,w), dtype=np.float32)
    q = np.stack((xx,yy),-1) + u0.transpose(1,2,0)
    a, b = image_descriptor(image0), image_descriptor(image1)
    warped, valid = _sample(b,q)
    photo = np.abs(a.transpose(1,2,0)-warped).mean(-1)
    photo[~valid] = 2.
    mean = np.stack([cv2.boxFilter(v,-1,(5,5)) for v in u0])
    variation = np.linalg.norm(u0-mean,axis=0)
    # Both official task uncertainty parameterizations are monotone increasing.
    risk = (_rank01(uncertainty) + _rank01(photo) + _rank01(variation)) / 3
    count = min(config.max_queries, max(1, int(np.ceil(config.query_fraction * risk.size))))
    threshold = np.partition(risk.ravel(), risk.size-count)[risk.size-count]
    selected = np.flatnonzero(risk.ravel() > threshold)
    tied = np.flatnonzero(risk.ravel() == threshold)
    remaining = count - len(selected)
    # Spread unavoidable top-k ties over the raster; never manufacture a risk
    # gradient or exceed query_fraction just because many values are equal.
    chosen_ties = tied[np.floor((np.arange(remaining) + .5) * len(tied) / remaining).astype(int)] if remaining else []
    selected = np.concatenate((selected, chosen_ties)).astype(np.int64)
    mask = np.zeros((h,w),bool);mask.ravel()[selected] = True
    anchors = (risk <= np.quantile(risk,config.anchor_quantile)) & ~mask & valid
    return mask, anchors, risk


class FeatureRematcher:
    def __init__(self, config=None):
        self.config = config or MatchConfig()

    def prepare(self, prediction, image0, image1):
        cfg = self.config
        if prediction.source_features.shape[0] != prediction.target_features.shape[0]:
            raise ValueError('Source and target must share the same encoder feature dimension')
        expected_shape = (image0.shape[0] // cfg.feature_stride, image0.shape[1] // cfg.feature_stride)
        if image0.shape[:2] != image1.shape[:2] or any(
                tuple(f.shape[1:]) != expected_shape for f in (prediction.source_features, prediction.target_features)):
            raise ValueError('Encoder grid does not match the declared feature stride and shared image context')
        metadata = getattr(prediction, 'metadata', {})
        return {
            'source': compress_features(prediction.source_features,cfg.feature_dim,cfg.projection_seed),
            'target': compress_features(prediction.target_features,cfg.feature_dim,cfg.projection_seed),
            'desc0': image_descriptor(image0), 'desc1': image_descriptor(image1),
            'feature_spec': {'dimensions': min(cfg.feature_dim, prediction.source_features.shape[0]),
                             'input_dimensions': prediction.source_features.shape[0],
                             'projection_seed': cfg.projection_seed, 'stride': cfg.feature_stride,
                             'checkpoint_sha256': metadata.get('checkpoint', {}).get('sha256'),
                             'feature_layer': metadata.get('feature_layer')},
            'origin_xy': tuple(metadata.get('origin_xy', (0, 0))),
        }

    def score(self, prepared, xy, candidates, baseline):
        """Return raw paired costs and validity for K x N x 2 displacements."""
        cfg = self.config
        if candidates.ndim != 3 or candidates.shape[-1] != 2 or xy.shape != candidates.shape[1:] or baseline.shape != xy.shape:
            raise ValueError('Expected candidates [K,N,2] and query/baseline [N,2]')
        for key in ('source', 'target', 'desc0', 'desc1'):
            if np.asarray(prepared[key]).ndim != 3 or not np.isfinite(prepared[key]).all():
                raise ValueError('Prepared features/descriptors must be finite CHW arrays')
        if prepared['source'].shape[0] != prepared['target'].shape[0]:
            raise ValueError('Source and target feature dimensions differ')
        if not np.isfinite(candidates).all() or not np.isfinite(xy).all() or not np.isfinite(baseline).all():
            raise ValueError('Candidate coordinates must be finite')
        count, n, _ = candidates.shape
        costs = np.full((count,n),np.inf,np.float32)
        feature_cost = np.full_like(costs,np.inf);photo_cost = np.full_like(costs,np.inf)
        validity = np.zeros_like(costs,bool)
        for start in range(0,n,cfg.chunk_size):
            sl=slice(start,min(n,start+cfg.chunk_size));p=xy[sl]
            sf, sv = _sample(prepared['source'],p,cfg.feature_stride)
            sd, sdv = _sample(prepared['desc0'],p)
            target_xy = p[None] + candidates[:,sl]
            tf,tv = _sample(prepared['target'],target_xy,cfg.feature_stride)
            td,tdv = _sample(prepared['desc1'],target_xy)
            sf /= np.maximum(np.linalg.norm(sf,axis=-1,keepdims=True),1e-8)
            tf /= np.maximum(np.linalg.norm(tf,axis=-1,keepdims=True),1e-8)
            fc = np.clip(1-np.sum(sf[None]*tf,axis=-1),0,2)
            pc = np.abs(sd[None]-td).mean(-1)
            valid = sv[None]&sdv[None]&tv&tdv
            feature_cost[:,sl]=fc;photo_cost[:,sl]=pc
            costs[:,sl]=np.where(valid,cfg.feature_weight*fc+(1-cfg.feature_weight)*pc,np.inf)
            validity[:,sl]=valid
        delta = np.linalg.norm(candidates-baseline[None],axis=-1)
        return costs, validity, feature_cost, photo_cost, delta

    def spatial_candidates(self, xy, u0, anchors, donor, task, local_candidates=()):
        x,y=xy[:,0].astype(int),xy[:,1].astype(int);h,w=anchors.shape
        baseline=u0[:,y,x].T
        candidates=[baseline.copy()];names=['identity'];groups=['current_pair']
        for index,field in enumerate(local_candidates):
            candidates.append(field[:,y,x].T.copy());names.append(f'local_{index}');groups.append('current_pair')
        for dx,dy in self.config.source_neighbors:
            xx,yy=x+dx,y+dy
            inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
            xx=np.clip(xx,0,w-1);yy=np.clip(yy,0,h-1)
            use=inside&anchors[yy,xx]
            candidate=baseline.copy();candidate[use]=donor[:,yy[use],xx[use]].T
            candidates.append(candidate);names.append(f'support_{dx}_{dy}');groups.append('current_pair')
        offsets=((-2,0),(2,0),(-8,0),(8,0)) if task=='stereo' else ((-2,0),(2,0),(0,-2),(0,2),(-8,0),(8,0),(0,-8),(0,8))
        for dx,dy in offsets:
            candidates.append(baseline+np.array([dx,dy],np.float32));names.append(f'residual_{dx}_{dy}');groups.append('current_pair')
        result=np.stack(candidates)
        if task=='stereo':result[...,1]=0
        return result,names,groups

    def temporal_candidates(self, xy, prepared, anchors, origin_xy, task):
        """Two-stage feature association via arrived anchor, with native endpoints.

        Association confidence remains an observational score, not a proof of
        identity. Each source group supplies a separate candidate, never an
        average of unrelated motions. Current target is always rescored later.
        """
        result=[];groups=[];diagnostics=[]
        cfg=self.config
        # Reserve identity before association work. Deduplicate identical roots
        # and contents before the cap so repeated copies cannot crowd out a
        # distinct source or create additional candidate votes.
        chosen=[]
        for record in anchors:
            payload = record.payload if hasattr(record, 'payload') else record
            roots=tuple(sorted(set(payload.source_groups))) or (f'{payload.sequence}/{payload.view}/{payload.frame}',)
            if any(roots == old_roots and np.array_equal(payload.features, old.features)
                   for old, old_roots in chosen):
                continue
            if len(chosen) >= cfg.max_candidates - 1:
                break
            chosen.append((payload, roots))
        if not chosen or len(xy) == 0:
            return result, groups, diagnostics
        sf,sv=_sample(prepared['source'],xy,cfg.feature_stride)
        sf /= np.maximum(np.linalg.norm(sf, axis=-1, keepdims=True), 1e-8)
        target=prepared['target'];ht,wt=target.shape[1:]
        target_flat=target.reshape(target.shape[0],-1).T
        target_flat = target_flat / np.maximum(np.linalg.norm(target_flat, axis=-1, keepdims=True), 1e-8)
        for payload, roots in chosen:
            history=np.asarray(payload.features,np.float32)
            if history.ndim != 3 or min(history.shape) < 1 or not np.isfinite(history).all() or history.shape[0] != sf.shape[-1]:
                raise ValueError('Memory feature version/dimension differs')
            if payload.metadata.get('feature_spec') is not None and payload.metadata['feature_spec'] != prepared.get('feature_spec'):
                raise ValueError('Memory features use another projection/stride version')
            hist_flat=history.reshape(history.shape[0],-1).T
            hist_flat = hist_flat / np.maximum(np.linalg.norm(hist_flat, axis=-1, keepdims=True), 1e-8)
            candidate=np.empty((len(xy),2),np.float32);confidence=np.empty(len(xy),np.float32)
            anchor_xy=np.empty((len(xy),2),np.float32)
            for start in range(0,len(xy),cfg.chunk_size):
                sl=slice(start,start+cfg.chunk_size)
                sim=sf[sl] @ hist_flat.T
                ids=sim.argmax(axis=1)
                anchor_xy[sl] = np.stack(((ids % history.shape[2] + .5) * cfg.feature_stride - .5 + payload.origin_xy[0],
                                          (ids // history.shape[2] + .5) * cfg.feature_stride - .5 + payload.origin_xy[1]), axis=-1)
                bridge=hist_flat[ids]
                target_sim=bridge @ target_flat.T
                if task=='stereo':
                    rows=np.repeat(np.arange(ht),wt)
                    source_rows=(xy[sl,1]+.5)/cfg.feature_stride-.5
                    target_sim[np.abs(rows[None]-source_rows[:,None])>1.1]=-np.inf
                target_ids=target_sim.argmax(axis=1)
                tx=(target_ids%wt+.5)*cfg.feature_stride-.5
                ty=(target_ids//wt+.5)*cfg.feature_stride-.5
                # Preserve within-token source offset for the coarse hypothesis.
                source_center=(np.floor((xy[sl]+.5)/cfg.feature_stride)+.5)*cfg.feature_stride-.5
                endpoint=np.stack((tx,ty),-1)+(xy[sl]-source_center)
                candidate[sl]=endpoint-xy[sl]
                confidence[sl]=np.minimum(sim[np.arange(len(ids)),ids],target_sim[np.arange(len(ids)),target_ids])
            if task=='stereo':candidate[:,1]=0
            candidate[~sv]=0
            result.append(candidate)
            groups.append('|'.join(roots))
            finite_confidence = confidence[np.isfinite(confidence) & sv]
            diagnostics.append({'anchor_frame':int(payload.frame),'view':payload.view,'origin_xy':list(payload.origin_xy),
                'mean_association_similarity':float(finite_confidence.mean()) if len(finite_confidence) else None,
                'valid_source_queries':int(sv.sum()), 'source_group':groups[-1],
                'example_native_endpoints': {'source': (xy[0]+origin_xy).tolist(),
                                             'anchor':anchor_xy[0].tolist(),
                                             'current_target':(xy[0]+candidate[0]+origin_xy).tolist()},
                'association_kind':'source_to_anchor_to_current_target_feature_hypothesis',
                'cycle_constraint_verified':False, 'historical_displacement_used':False,
                'prior_not_independent_verification':True})
        return result,groups,diagnostics

    def refine(self, prediction, image0, image1, task, *, prepared=None, mask=None, anchors=None,
               donor=None, local_candidates=(), temporal_records=(), origin_xy=(0,0), arm='solved', acceptor=None):
        cfg=self.config;u0=np.asarray(prediction.displacement,np.float32)
        if acceptor is not None:
            if getattr(acceptor, 'feature_schema', None) != CANDIDATE_FEATURE_SCHEMA:
                raise ValueError('Acceptor and matcher candidate feature schemas differ')
            if getattr(acceptor, 'max_update_px', None) != cfg.max_update_px or \
                    getattr(acceptor, 'require_baseline_support', None) != cfg.require_baseline_support:
                raise ValueError('Acceptor calibration and matcher eligibility policies differ')
        if task not in ('stereo','flow') or u0.ndim != 3 or u0.shape[0] != 2 or not np.isfinite(u0).all():raise ValueError('Invalid task or baseline')
        if task == 'stereo' and np.any(u0[1] != 0):
            raise ValueError('Stereo baseline must have zero vertical displacement')
        if image0.shape != (*u0.shape[1:], 3) or image1.shape != image0.shape or np.shape(prediction.raw_uncertainty) != u0.shape[1:]:
            raise ValueError('Images, uncertainty, and native baseline grid must have matching shapes')
        auto_mask,auto_anchors,risk=query_and_anchor_masks(u0,prediction.raw_uncertainty,image0,image1,cfg)
        mask=auto_mask if mask is None else np.asarray(mask,bool)
        anchors=auto_anchors if anchors is None else np.asarray(anchors,bool)&~mask
        if mask.shape != u0.shape[1:] or anchors.shape != mask.shape:
            raise ValueError('Query and anchor masks must match the native image grid')
        if np.count_nonzero(mask) > cfg.max_queries:
            raise ValueError('Explicit query mask exceeds max_queries; divide work into declared batches')
        if not np.any(mask):
            # Keep a complete empty candidate bank so evaluation can still
            # publish the identity candidate oracle and preserve fixed row
            # counts for cases whose registered diagnostic mask is empty.
            return {'output':u0.copy(),'query_mask':mask,'anchor_mask':anchors,'accepted':mask.copy(),'risk':risk,
                    'query_xy':np.empty((0,2),np.float32),'candidates':np.empty((1,0,2),np.float32),
                    'candidate_features':np.empty((1,0,12),np.float32),
                    'candidate_valid':np.empty((1,0),bool),'candidate_costs':np.empty((1,0),np.float32),
                    'best_indices':np.empty((0,),np.int64),'candidate_names':['identity'],
                    'candidate_groups':['current_pair'],'feature_schema':CANDIDATE_FEATURE_SCHEMA,
                    'diagnostics':{'query_count':0,'accepted_count':0,'candidate_count':1,
                                   'candidate_evaluations':0,'config':asdict(cfg),
                                   'acceptor':'learned' if acceptor is not None else 'fixed_observational_rule',
                                   'selection_rule':'empty_query_identity_v1'}}
        prepared=prepared or self.prepare(prediction,image0,image1)
        yy,xx=np.nonzero(mask);xy=np.stack((xx,yy),-1).astype(np.float32)
        donor=u0 if donor is None else donor
        if any(np.shape(field) != u0.shape or not np.isfinite(field).all() for field in (donor, *local_candidates)):
            raise ValueError('Donor and local candidate fields must be finite and match baseline shape')
        candidates,names,groups=self.spatial_candidates(xy,u0,anchors,donor,task,local_candidates)
        if arm=='generic':
            # Same candidate budget; replace all donor slots by broader residual search.
            base=candidates[0]
            for k,name in enumerate(names):
                if name.startswith('support_'):
                    direction=-1 if k%2 else 1;step=4*(1+k//2)
                    candidates[k]=base+np.array([direction*step,0 if task=='stereo' else direction*(step//2)],np.float32)
                    names[k]=f'generic_search_{k}'
        extras,extra_groups,association=self.temporal_candidates(xy,prepared,temporal_records,origin_xy,task)
        if extras:
            remaining=max(1,cfg.max_candidates-len(extras))
            candidates=np.concatenate((candidates[:remaining],np.stack(extras)),axis=0)
            names=names[:remaining]+[f'history_{k}' for k in range(len(extras))]
            groups=groups[:remaining]+extra_groups
        candidates=candidates[:cfg.max_candidates];names=names[:cfg.max_candidates];groups=groups[:cfg.max_candidates]
        # Deduplicate identical lineage-and-value slots; repeating a record creates no votes.
        kept=[]
        for k in range(len(names)):
            if not any(groups[k]==groups[j] and np.array_equal(candidates[k],candidates[j]) for j in kept):kept.append(k)
        candidates=candidates[kept];names=[names[k] for k in kept];groups=[groups[k] for k in kept]
        baseline=u0[:,yy,xx].T
        raw,valid,fc,pc,delta=self.score(prepared,xy,candidates,baseline)
        penalized=raw+cfg.movement_penalty*delta
        # A cheap but inadmissible proposal cannot cancel another valid update.
        # Identity remains the fallback even when its target is unobserved.
        fixed_eligible=valid&(delta<=cfg.max_update_px)
        if cfg.require_baseline_support:fixed_eligible &= valid[0][None]
        selection_cost=np.where(fixed_eligible,penalized,np.inf)
        selection_cost[0]=penalized[0]
        best=selection_cost.argmin(axis=0);ids=np.arange(len(xy))
        baseline_cost=raw[0]
        acceptable=fixed_eligible[best,ids]&(best!=0)
        acceptable &= penalized[best,ids]+cfg.min_improvement < baseline_cost
        # Finite observable inputs only; missing support is represented separately.
        cand_features=np.stack((np.broadcast_to(risk[yy,xx],raw.shape),np.clip(raw,0,4),np.clip(fc,0,2),np.clip(pc,0,2),
                               np.broadcast_to(np.clip(baseline_cost,0,4),raw.shape),np.clip(delta/32,0,32),
                               candidates[...,0]/128,candidates[...,1]/128,valid.astype(np.float32),
                               np.broadcast_to(valid[0],raw.shape).astype(np.float32),
                               np.full(raw.shape,float(task=='stereo'),np.float32),
                               np.broadcast_to(np.array([name.startswith('history_') for name in names],np.float32)[:,None],raw.shape)),axis=-1).astype(np.float32)
        if acceptor is not None:
            gain,harm=acceptor.predict(cand_features)
            if np.shape(gain) != raw.shape or np.shape(harm) != raw.shape or not np.isfinite(gain).all() \
                    or not np.isfinite(harm).all() or np.any((harm < 0) | (harm > 1)):
                raise ValueError('Acceptor outputs must be finite [K,N] gain and harm probabilities')
            eligible=valid&(delta<=cfg.max_update_px)
            if cfg.require_baseline_support:eligible &= valid[0][None]
            gain=np.where(eligible,gain,-np.inf);gain[0]=0
            best=gain.argmax(axis=0)
            acceptable=(best!=0)&eligible[best,ids]&(gain[best,ids]>acceptor.gain_threshold)&(harm[best,ids]<=acceptor.harm_threshold)
        output=u0.copy();accepted=np.zeros_like(mask)
        output[:,yy[acceptable],xx[acceptable]]=candidates[best[acceptable],ids[acceptable]].T
        accepted[yy[acceptable],xx[acceptable]]=True
        if task=='stereo':output[1]=0
        assert np.array_equal(output[:,~mask],u0[:,~mask])
        return {'output':output,'query_mask':mask,'anchor_mask':anchors,'accepted':accepted,'risk':risk,
                'query_xy':xy,'candidates':candidates,'candidate_features':cand_features,'candidate_valid':valid,
                'feature_schema':CANDIDATE_FEATURE_SCHEMA,
                'candidate_costs':raw,'best_indices':best,'candidate_names':names,'candidate_groups':groups,
                'diagnostics':{'query_count':len(xy),'accepted_count':int(acceptable.sum()),'candidate_count':len(candidates),
                    'candidate_evaluations':int(len(candidates)*len(xy)),'direct_support_fraction':float(valid.mean()),
                    'association':association,'config':asdict(cfg),'acceptor':'learned' if acceptor is not None else 'fixed_observational_rule',
                    'selection_rule':'eligibility_then_gain_argmax_v1' if acceptor is not None else 'eligibility_then_penalized_cost_argmin_v2',
                    'cost_note':'Search association and feature preparation are additional measured costs; no independence or calibrated safety claim.'}}
