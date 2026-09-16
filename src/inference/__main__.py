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
    args = parser.parse_args()
    values = (args.mask_threshold, args.cls_threshold, args.min_area)
    if any(value is not None for value in values) and not all(value is not None for value in values):
        parser.error("provide all three thresholds, or omit them to use the run summary")
    thresholds = ThresholdConfig(*values) if values[0] is not None else None
    path = create_submission(args.run_dir, args.output_dir, thresholds=thresholds,
                             data_path=args.data_path, template_path=args.template_path, device=args.device,
                             checkpoint_name=args.checkpoint)
    print(path)


if __name__ == "__main__":
    main()
