"""Frozen checkpoint evaluation for original diagnostics and final holdout."""

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.collation import ValidationCollator
from src.data.data_workspace import DataWorkspace
from src.data.dataset import AIIJCDataset
from src.data.geometry import restore_probability
from src.eval.diagnostics import EvaluationReport
from src.eval.protocol import EvaluationProtocol, GroupConnections, file_digest, rows_digest
from src.inference.predict import ThresholdConfig
from src.inference.submission import InferenceConfig
from src.progress import ConsoleProgress
from src.training.builders import AmpContext, DataLoaderThreadLimits, build_model
from src.training.metric import AICAccumulator
from src.training.runs import Run
from src.training.selection import RetunedSelection
from src.training.transfer import BatchTransfer
from src.training.validation import DeviceHistogramAccumulator


def historical_originals(metadata, validation, originals, pairs):
    groups = GroupConnections(metadata, pairs)
    train = metadata.loc[~metadata.chng_img_path.isin(validation.chng_img_path)]
    train_groups = set(train.group_id.map(groups.find))
    val_groups = set(validation.group_id.map(groups.find))
    connected = originals.group_id.map(groups.find)
    keep = connected.isin(val_groups) & ~connected.isin(train_groups)
    return originals.loc[keep].reset_index(drop=True), originals.loc[connected.isin(val_groups) & ~keep].reset_index(drop=True)


def claim_holdout(run_dir, provenance):
    """Exclusive per-run marker; a failed attempt remains visible and is not silently retried."""
    with (Path(run_dir) / 'holdout_claim.json').open('x', encoding='utf-8') as stream:
        json.dump(provenance, stream, indent=2)


class CheckpointEvaluator:
    def __init__(self, run_dir, *, device='cuda', batch_size=4, workers=2):
        if batch_size <= 0 or workers < 0:
            raise ValueError('batch_size must be positive and workers non-negative')
        self.device, self.batch_size, self.workers = torch.device(device), batch_size, workers
        # Fail before a holdout claim when the requested backend cannot run.
        torch.empty(0, device=self.device)
        self.run = Run.open(run_dir)
        self.config = InferenceConfig.from_snapshot(self.run.snapshot)
        self.checkpoint_path = self.run.dir / 'ckpt/best.pt'
        self.checkpoint_digest = file_digest(self.checkpoint_path)
        self.thresholds = ThresholdConfig.from_summary(self.run.summary, self.run.snapshot)
        RetunedSelection.verify(self.run.summary, self.run.snapshot, self.checkpoint_path)

    def evaluate(self, rows, directory, *, purpose):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        rows = rows.copy().reset_index(drop=True)
        if rows.empty:
            raise ValueError('No evaluation rows')
        if 'target_kind' not in rows:
            rows['target_kind'] = 'provided'
        rows = EvaluationReport.add_jpeg_metadata(rows, self.config.data_path / 'train_stage1', self.workers)
        provenance = dict(purpose=purpose, checkpoint_sha256=self.checkpoint_digest,
                          thresholds=asdict(self.thresholds), rows_digest=rows_digest(rows),
                          resolution='original', device=str(self.device), amp=self.config.amp,
                          batch_size=self.batch_size, source_run=str(self.run.dir.resolve()),
                          torch_version=torch.__version__, thresholds_tuned=False)
        (directory / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        rows.to_parquet(directory / 'rows.parquet', index=False)
        DataLoaderThreadLimits.apply()
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
        model = build_model(self.config.model, aux_weight=self.config.aux_weight, pretrained=False)
        model.load_state_dict(checkpoint['ema'] if checkpoint.get('ema') is not None else checkpoint['model'])
        del checkpoint
        model = model.to(self.device, memory_format=torch.channels_last).eval()
        amp = AmpContext(self.device, torch.bfloat16 if self.config.amp == 'bf16' else torch.float16,
                         self.device.type == 'cuda' and self.config.amp != 'off', False)
        dataset = AIIJCDataset(DataWorkspace(self.config.data_path), rows, False,
                              self.config.image_size, self.config.seed, mode='val', original_targets=True)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                            collate_fn=ValidationCollator(), worker_init_fn=DataLoaderThreadLimits.apply,
                            pin_memory=self.device.type == 'cuda')
        snapshot = self.run.snapshot
        evaluation = snapshot.get('eval', snapshot)
        selected_weight = self.run.summary['best'].get('small_mask_weight',
            evaluation.get('selection_small_mask_weight', evaluation.get('small_mask_weight', 1.0)))
        acc = AICAccumulator(n_bins=int(evaluation.get('n_bins', 256)),
                             small_mask_weight=selected_weight)
        histograms = DeviceHistogramAccumulator(acc, self.device)
        try:
            with torch.inference_mode():
                for batch in ConsoleProgress.iterate(loader, f'{self.run.dir.name}: {purpose}'):
                    with amp.autocast():
                        out = model(batch['image'].to(self.device, memory_format=torch.channels_last),
                                    jpeg=BatchTransfer.move_jpeg(batch['jpeg'], self.device))
                    probs, cls = out['logits'].float().sigmoid(), out['cls_logits'].float().sigmoid().flatten()
                    for i, mask in enumerate(batch['original_mask']):
                        restored = restore_probability(probs[i:i+1], mask.shape[-2:])
                        histograms.update(restored, mask.to(self.device, non_blocking=True).reshape(1, 1, *mask.shape[-2:]), cls[i:i+1])
                histograms.flush()
            if file_digest(self.checkpoint_path) != self.checkpoint_digest:
                raise ValueError('Checkpoint changed during evaluation')
            acc.save(directory / 'predictions.npz')
            report = EvaluationReport(acc, rows, self.thresholds)
            report.save(directory)
            return report.summary()
        finally:
            del model, loader
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def holdout(self):
        snapshot = self.run.snapshot
        protocol = EvaluationProtocol.load(snapshot.get('dataset', snapshot)['protocol_path'])
        protocol.verify_run(self.run.snapshot)
        summary = self.run.summary
        if not summary.get('training_complete'):
            raise ValueError('Complete development training before evaluating holdout')
        if ThresholdConfig.from_summary(summary, snapshot) != self.thresholds:
            raise ValueError('Selected thresholds changed after evaluator initialization')
        retuned = RetunedSelection.verify(summary, snapshot, self.checkpoint_path, check_histograms=True)
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
        protocol.verify_run(checkpoint.get('cfg', {}))
        point = checkpoint.get('operating_point', {})
        selected = {key: value for key, value in asdict(self.thresholds).items() if key != 'n_bins'}
        point = dict(point, area_cap=point.get('area_cap', 0.0))
        if not retuned and any(point.get(key) != value for key, value in selected.items()):
            raise ValueError('Holdout thresholds differ from selected checkpoint operating point')
        del checkpoint
        for name, expected in [('training', self.run.snapshot['training_rows_digest']),
                               ('development', self.run.snapshot['development_rows_digest'])]:
            if rows_digest(pd.read_parquet(self.run.dir / f'{name}_rows.parquet')) != expected:
                raise ValueError(f'Actual {name} rows differ from protocol')
        claim_holdout(self.run.dir, dict(protocol_digest=protocol.digest, checkpoint_sha256=self.checkpoint_digest,
                                        thresholds=asdict(self.thresholds)))
        result = self.evaluate(protocol.rows('holdout'), self.run.dir / 'holdout', purpose='independent_holdout')
        self.run.save_summary({'holdout_evaluated': True, 'holdout': result})
        return result
