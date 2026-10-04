"""StableBridge command-line entry points. Long jobs use detached launcher."""
import argparse
import json
from pathlib import Path
import traceback
from .util import save_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    geometry=sub.add_parser('geometry')
    geometry.add_argument('--run-dir',required=True)
    geometry.add_argument('--device',default='cuda:0')
    geometry.add_argument('--model-profile',default='main',choices=['main','legacy_spring'])
    geometry.add_argument('--native-only',action='store_true')
    run=sub.add_parser('run')
    run.add_argument('--config',required=True);run.add_argument('--run-dir',required=True)
    run.add_argument('--device',default='cuda:0');run.add_argument('--acceptor')
    report=sub.add_parser('report');report.add_argument('--run-dir',required=True)
    collect=sub.add_parser('collect-supervision');collect.add_argument('--runs',nargs='+',required=True)
    collect.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='geometry':
            from .runner import geometry_run
            geometry_run(args.run_dir,device=args.device,profile=args.model_profile,
                         contexts=() if args.native_only else ((512,768),))
        elif args.command=='run':
            from .runner import run_experiment
            run_experiment(args.config,args.run_dir,device=args.device,acceptor_path=args.acceptor)
        elif args.command=='report':
            from .runner import materialize_results
            materialize_results(args.run_dir)
        elif args.command=='collect-supervision':
            from .runner import collect_supervision
            collect_supervision(args.runs,args.output)
    except Exception as error:
        if getattr(args,'run_dir',None):
            save_json(Path(args.run_dir)/'status.json',{'state':'failed','error':repr(error),
                                                       'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
