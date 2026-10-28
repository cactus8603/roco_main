"""Command line tools for frozen WAFT feature repair."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Train motion-aligned repair through frozen native WAFT and predict flow; legacy target tools are also available.")
    command.add_argument("--threads", type=int, default=2, help="CPU worker threads (default: 2)")
    sub = command.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run a small synthetic CPU training and feature prediction example")
    demo.add_argument("--output", required=True)
    demo.add_argument("--epochs", type=int, default=2)
    doctor = sub.add_parser("doctor", help="Report installed dependencies without starting CUDA")
    doctor.add_argument("--waft-source")
    doctor.add_argument("--waft-checkpoint")
    manifest = sub.add_parser("manifest", help="Verify feature packets and build a portable training manifest")
    manifest.add_argument("--spec", required=True)
    manifest.add_argument("--output", required=True)
    ground_truth = sub.add_parser("convert-gt", help="Convert KITTI PNG or Sintel FLO ground truth to the checked NPZ format")
    ground_truth.add_argument("--format", choices=("kitti", "sintel"), required=True)
    ground_truth.add_argument("--flow", required=True)
    ground_truth.add_argument("--visibility", help="KITTI flow_noc PNG or Sintel occlusions PNG")
    ground_truth.add_argument("--output", required=True)
    cached = sub.add_parser("import-cache", help="Import checked per-pair visible-target descriptors")
    cached.add_argument("--directory", required=True)
    cached.add_argument("--output", required=True)
    dense = sub.add_parser("import-expensive", help="Import qualified entries from the expensive target cache")
    dense.add_argument("--entries", required=True)
    dense.add_argument("--output", required=True)
    prepare = sub.add_parser("prepare", help="Encode images and prepare cheap or expensive offline target features")
    prepare.add_argument("--spec", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--target", choices=("cheap", "expensive"), default="cheap")
    prepare.add_argument("--target-steps", type=int, choices=(32, 256, 512),
                         help="Expensive target AdamW updates (default: 512); WAFT always uses 5 iterations")
    prepare.add_argument("--waft-source", required=True)
    prepare.add_argument("--waft-checkpoint", required=True)
    prepare.add_argument("--device", default="cpu")
    fit = sub.add_parser("train", help="Fit only the repair model to a checked feature manifest")
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--epochs", type=int, default=10)
    fit.add_argument("--seed", type=int, default=20261026)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--backend", choices=("reference", "native"), default="reference")
    fit.add_argument("--resume")
    fit.add_argument("--max-updates", type=int)
    fit.add_argument("--checkpoint-every", type=int)
    prediction = sub.add_parser("infer-features", help="Repair a feature packet using a saved repair checkpoint")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--input", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--device", default="cpu")
    prediction.add_argument("--backend", choices=("reference", "native"))
    images = sub.add_parser("infer-images", help="Predict native-resolution flow from two images, without GT or targets")
    images.add_argument("--checkpoint", required=True)
    images.add_argument("--image1", required=True)
    images.add_argument("--image2", required=True)
    images.add_argument("--output", required=True)
    images.add_argument("--waft-source", required=True)
    images.add_argument("--waft-checkpoint", required=True)
    images.add_argument("--device", default="cpu")
    images.add_argument("--backend", choices=("reference", "native"), default="reference")
    from .native_cli import register
    register(sub)
    return command


def main(argv=None) -> None:
    command = parser()
    args = command.parse_args(argv)
    if args.command == "prepare" and args.target_steps is not None and args.target != "expensive":
        command.error("--target-steps is only valid with --target expensive")
    if args.threads < 1:
        command.error("--threads must be positive")
    import torch
    torch.set_num_threads(args.threads)
    try:
        result = dispatch(args)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, ImportError) as error:
        command.exit(2, f"waft-repair: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


def dispatch(args):
    if args.command.startswith("native-"):
        from .native_cli import dispatch_native
        return dispatch_native(args)
    if args.command == "doctor":
        import torch
        versions = {}
        for package in ("torch", "numpy", "torchvision", "timm", "einops", "transformers", "mamba-ssm", "causal-conv1d"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        result = {"packages": versions, "cuda_initialized": torch.cuda.is_initialized(),
                  "reference_backend": "legacy synthetic small CPU grids",
                  "legacy_slim_backend": "CUDA and compiled Mamba/causal-conv1d required",
                  "current_motion_backend": "PyTorch convolutions; official WAFT CPU or CUDA; no Mamba required"}
        if bool(args.waft_source) != bool(args.waft_checkpoint):
            raise ValueError("Pass --waft-source and --waft-checkpoint together")
        if args.waft_source:
            from .backend import load_backend
            backend = load_backend(args.waft_source, args.waft_checkpoint, device="cpu")
            result["waft"] = backend.metadata
        return result
    if args.command == "demo":
        from .data import make_demo
        from .training import train, infer_features
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        manifest = make_demo(output / "data")
        fit = train(manifest, output / "training", device="cpu", backend="reference", epochs=args.epochs, seed=7)
        predicted = infer_features(output / "training/checkpoint.pt", output / "data/example-1-input.pt",
                                   output / "repaired-features.pt", device="cpu", backend="reference")
        return {"synthetic_only": True, "pretrained_waft_loaded": False, "training": fit, "prediction": predicted}
    if args.command == "manifest":
        from .data import prepare_manifest
        value = prepare_manifest(args.spec, args.output)
        return {"manifest": str(Path(args.output).resolve()), "samples": len(value["samples"]), "target": value["target_kind"]}
    if args.command == "convert-gt":
        from .ground_truth import convert_ground_truth
        return convert_ground_truth(args.format, args.flow, args.output, args.visibility)
    if args.command == "import-cache":
        from .data import import_cache
        pairs = sorted(Path(args.directory).glob("pair_*.json"))
        if not pairs:
            raise ValueError("No pair_*.json descriptors in --directory")
        value = import_cache(pairs, args.output)
        return {"manifest": str(Path(args.output).resolve()), "samples": len(value["samples"]), "target": value["target_kind"]}
    if args.command == "import-expensive":
        from .data import import_dense_cache
        value = import_dense_cache(args.entries, args.output)
        return {"manifest": str(Path(args.output).resolve()), "samples": len(value["samples"]), "target": value["target_kind"]}
    if args.command == "prepare":
        from .backend import prepare_dataset
        manifest = prepare_dataset(args.spec, args.output, args.waft_source, args.waft_checkpoint,
                                   device=args.device, target_kind=args.target, target_steps=args.target_steps)
        return {"manifest": str(manifest), "target": args.target}
    if args.command == "train":
        from .training import train
        return train(args.manifest, args.output, epochs=args.epochs, seed=args.seed,
                     device=args.device, backend=args.backend, resume=args.resume,
                     max_updates=args.max_updates, checkpoint_every=args.checkpoint_every)
    if args.command == "infer-features":
        from .training import infer_features
        return infer_features(args.checkpoint, args.input, args.output, device=args.device, backend=args.backend)
    if args.command == "infer-images":
        from .backend import infer_images
        return infer_images(args.checkpoint, args.image1, args.image2, args.output,
                            args.waft_source, args.waft_checkpoint, device=args.device, backend=args.backend)
    raise ValueError("Unsupported command")
