import argparse

from .predict import ThresholdConfig
from .submission import create_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a submission from a saved run")
    parser.add_argument("run_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--data-path")
    parser.add_argument("--template-path")
    parser.add_argument("--device")
    parser.add_argument("--checkpoint", choices=('best.pt', 'last.pt'), default='best.pt',
                        help='Checkpoint in run_dir/ckpt; use last.pt for blind finetuning')
    parser.add_argument("--mask-threshold", type=float)
    parser.add_argument("--cls-threshold", type=float)
    parser.add_argument("--min-area", type=float)
    parser.add_argument('--area-cap', type=float, default=0., help='clip doubted masks below this area; 0 blanks them')
    parser.add_argument('--n-bins', type=int, help='histogram grid for explicit thresholds; default: run snapshot')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--post-workers', type=int, default=8, help='CPU threads for masks and PNGs; 0 runs serially')
    args = parser.parse_args()
    values = (args.mask_threshold, args.cls_threshold, args.min_area)
    if any(value is not None for value in values) and not all(value is not None for value in values):
        parser.error("provide all three thresholds, or omit them to use the run summary")
    if values[0] is None and (args.area_cap != 0.0 or args.n_bins is not None):
        parser.error('--area-cap and --n-bins require explicit thresholds')
    thresholds = None
    if values[0] is not None:
        from src.training.runs import Run

        snapshot = Run.open(args.run_dir).snapshot
        n_bins = args.n_bins if args.n_bins is not None else int(snapshot.get('eval', snapshot).get('n_bins', 256))
        thresholds = ThresholdConfig(*values, area_cap=args.area_cap, n_bins=n_bins)
    path = create_submission(args.run_dir, args.output_dir, thresholds=thresholds,
                             data_path=args.data_path, template_path=args.template_path, device=args.device,
                             checkpoint_name=args.checkpoint, batch_size=args.batch_size,
                             workers=args.workers, post_workers=args.post_workers)
    print(path)


if __name__ == "__main__":
    main()
