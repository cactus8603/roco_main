"""CLI for the current motion-aligned model; legacy commands remain available."""
from __future__ import annotations


def _waft(command):
    command.add_argument('--waft-source',required=True,help='Pinned official WAFT source checkout')
    command.add_argument('--waft-checkpoint',required=True,help='Complete official source-only zero-shot.pth')
    command.add_argument('--device',default='cpu')


def register(sub):
    split=sub.add_parser('native-split-kitti',help='Write the fixed KITTI160 TRAIN/40 dev RGB/GT specification')
    split.add_argument('--kitti-root',required=True)
    split.add_argument('--output',required=True,help='New directory for images.json and split.json')
    split.add_argument('--split-file',help='Optional explicit published split JSON')
    prep=sub.add_parser('native-prepare',help='Cache original F0/four taps/fifth baseline prior and separate original GT')
    prep.add_argument('--spec',required=True);prep.add_argument('--output',required=True);_waft(prep)
    prep.add_argument('--reserve-gib',type=float,default=20)
    prep.add_argument('--memory-fraction',type=float)
    fit=sub.add_parser('native-train',help='Train the current motion adapter through complete frozen native5 WAFT')
    fit.add_argument('--manifest',required=True);fit.add_argument('--output',required=True);_waft(fit)
    fit.add_argument('--epochs',type=int,default=32);fit.add_argument('--seed',type=int,default=20261028)
    fit.add_argument('--warmup-updates',type=int,default=160)
    fit.add_argument('--resume',help='Full checkpoint.pt belonging to this public run')
    fit.add_argument('--stop-after-updates',type=int,help='Stop at this absolute update without changing the full schedule')
    fit.add_argument('--checkpoint-every',type=int,default=32)
    fit.add_argument('--activation-offload',choices=('cpu','none'),default='cpu')
    fit.add_argument('--reserve-gib',type=float,default=20)
    fit.add_argument('--memory-fraction',type=float)
    evaluation=sub.add_parser('native-evaluate',help='Save all5 predictions and native original-support scores without updates')
    evaluation.add_argument('--manifest',required=True);evaluation.add_argument('--repair-checkpoint',required=True)
    evaluation.add_argument('--output',required=True);evaluation.add_argument('--split',choices=('train','dev','all'),default='dev')
    evaluation.add_argument('--expected-sha256');_waft(evaluation)
    infer=sub.add_parser('native-infer',help='GT-free two-image inference with the current repair model')
    infer.add_argument('--repair-checkpoint',required=True);infer.add_argument('--image1',required=True)
    infer.add_argument('--image2',required=True);infer.add_argument('--output',required=True)
    infer.add_argument('--expected-sha256');_waft(infer)
    converted=sub.add_parser('native-import-checkpoint',help='Export repair-only weights from an existing native training checkpoint')
    converted.add_argument('--checkpoint',required=True);converted.add_argument('--output',required=True)
    converted.add_argument('--expected-sha256')


def dispatch_native(args):
    if args.command=='native-split-kitti':
        from .motion.data import make_kitti_spec
        return make_kitti_spec(args.kitti_root,args.output,args.split_file)
    if args.command=='native-import-checkpoint':
        from .motion.checkpoint import import_checkpoint
        return import_checkpoint(args.checkpoint,args.output,expected_sha256=args.expected_sha256)
    from . import native_pipeline as pipeline
    core=dict(waft_source=args.waft_source,waft_checkpoint=args.waft_checkpoint,device=args.device)
    if args.command=='native-prepare':
        return pipeline.prepare(args.spec,args.output,**core,reserve_gib=args.reserve_gib,memory_fraction=args.memory_fraction)
    if args.command=='native-train':
        return pipeline.train(args.manifest,args.output,**core,epochs=args.epochs,seed=args.seed,
            warmup_updates=args.warmup_updates,resume=args.resume,stop_after_updates=args.stop_after_updates,
            checkpoint_every=args.checkpoint_every,activation_offload=args.activation_offload,
            reserve_gib=args.reserve_gib,memory_fraction=args.memory_fraction)
    if args.command=='native-evaluate':
        return pipeline.evaluate(args.manifest,args.repair_checkpoint,args.output,**core,
                                 split=args.split,expected_sha256=args.expected_sha256)
    if args.command=='native-infer':
        return pipeline.infer(args.repair_checkpoint,args.image1,args.image2,args.output,**core,
                              expected_sha256=args.expected_sha256)
    raise ValueError('Unknown native command')
