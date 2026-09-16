from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml
from dotenv import dotenv_values

import global_config
from src.data.augmentation.base import AugmentationConfig
from src.training.metric import DEFAULT_MASK_GRID

PIPELINE_VERSION = 'jpeg640_v1'
DEFAULT_PROTOCOL_PATH = 'runs/validation_protocol_20260908/protocol'


class ConfigSection:
    """Strict keys with dataclass defaults; snapshots contain resolved values."""

    @classmethod
    def from_dict(cls, data):
        data = _mapping(data, cls.__name__)
        section = cls.__name__.removesuffix('Config').lower()
        _check_keys(data, set(), section, optional={f.name for f in fields(cls)})
        return cls(**_float_fields(cls, data))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PathsConfig(ConfigSection):
    data_path: Path
    runs_path: Path

    @staticmethod
    def current_data_path() -> Path:
        """Resolve data on this machine, independent of saved run paths."""
        settings = {**dotenv_values(global_config.PROJECT_ROOT / '.env'), **os.environ}
        return _path(settings.get('AIIJC_DATA_PATH') or global_config.DATA_PATH,
                     global_config.PROJECT_ROOT)

    @classmethod
    def from_dict(cls, data, *, base_dir=Path('.')):
        data = _mapping(data, 'paths')
        _check_keys(data, set(), 'paths', optional={'data_path', 'runs_path'})
        settings = {**dotenv_values(global_config.PROJECT_ROOT / '.env'), **os.environ}
        return cls(
            cls.current_data_path(),
            _path(settings.get('AIIJC_RUNS_PATH') or global_config.RUNS_PATH, global_config.PROJECT_ROOT),
        )

    def to_dict(self):
        return dict(data_path=str(self.data_path), runs_path=str(self.runs_path))


@dataclass(frozen=True)
class ModelConfig(ConfigSection):
    encoder: str = 'pvt_v2_b2'
    jpeg_channels: tuple[int, ...] = (64, 96, 128)
    jpeg_pretrained: str | None = 'DCT_djpeg.pth'
    jpeg_specialize_width: bool = True
    jpeg_pointwise_matmul: bool = True
    jpeg_triton_backward: bool = False
    jpeg_similarity: bool = False
    disentangle_levels: tuple[int, ...] = ()
    disentangle_mode: str = 'fuse'
    disentangle_reduction: int = 16
    disentangle_cross_strides: tuple[int, ...] = ()
    disentangle_attention_width: int = 128
    disentangle_attention_heads: int = 4
    disentangle_parallel_16_32: bool = False
    disentangle_return_to_stride4: bool = False
    bifpn_width: int = 64
    bifpn_repeats: int = 0
    pristine_reference: bool = False

    def __post_init__(self):
        _non_empty_str(self.encoder, 'model.encoder')
        if type(self.jpeg_triton_backward) is not bool:
            raise ValueError('model.jpeg_triton_backward must be a boolean')
        if type(self.pristine_reference) is not bool:
            raise ValueError('model.pristine_reference must be a boolean')
        if type(self.bifpn_width) is not int or self.bifpn_width < 1:
            raise ValueError('model.bifpn_width must be a positive integer')
        if type(self.bifpn_repeats) is not int or self.bifpn_repeats < 0:
            raise ValueError('model.bifpn_repeats must be a nonnegative integer')
        if type(self.jpeg_similarity) is not bool:
            raise ValueError('model.jpeg_similarity must be a boolean')
        if type(self.jpeg_specialize_width) is not bool:
            raise ValueError('model.jpeg_specialize_width must be a boolean')
        if type(self.jpeg_pointwise_matmul) is not bool:
            raise ValueError('model.jpeg_pointwise_matmul must be a boolean')
        object.__setattr__(self, 'jpeg_channels', tuple(self.jpeg_channels))
        if len(self.jpeg_channels) != 3 or any(type(c) is not int or c <= 0 for c in self.jpeg_channels):
            raise ValueError('model.jpeg_channels must contain three positive integers')
        if self.jpeg_pretrained is not None:
            _non_empty_str(self.jpeg_pretrained, 'model.jpeg_pretrained')
        object.__setattr__(self, 'disentangle_levels', tuple(self.disentangle_levels))
        if type(self.disentangle_parallel_16_32) is not bool:
            raise ValueError('disentangle_parallel_16_32 must be a boolean')
        if self.disentangle_parallel_16_32 and (
                self.disentangle_mode != 'fuse' or
                tuple(self.disentangle_cross_strides) != (32,) or
                not {16, 32}.issubset(self.disentangle_levels)):
            raise ValueError('parallel_16_32 requires fuse, levels 16/32 and cross_strides [32]')
        if type(self.disentangle_return_to_stride4) is not bool:
            raise ValueError('model.disentangle_return_to_stride4 must be a boolean')
        if self.disentangle_return_to_stride4 and (
                self.disentangle_mode != 'fuse' or not {4, 32}.issubset(self.disentangle_levels)):
            raise ValueError('disentangle_return_to_stride4 requires fuse mode and levels 4 and 32')
        object.__setattr__(self, 'disentangle_cross_strides', tuple(self.disentangle_cross_strides))
        strides = self.disentangle_levels + self.disentangle_cross_strides
        if any(type(stride) is not int or stride < 1 for stride in strides):
            raise ValueError('model.disentangle strides must be positive integers')
        if len(set(self.disentangle_levels)) != len(self.disentangle_levels):
            raise ValueError('model.disentangle_levels must be unique')
        if self.disentangle_mode not in {'supervision', 'fuse'}:
            raise ValueError("model.disentangle_mode must be 'supervision' or 'fuse'")
        for key in ('disentangle_reduction', 'disentangle_attention_width', 'disentangle_attention_heads'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'model.{key} must be a positive integer')
        if self.disentangle_attention_width % self.disentangle_attention_heads:
            raise ValueError('model.disentangle_attention_width must divide by the head count')
        if self.disentangle_cross_strides and not self.disentangle_levels:
            raise ValueError('disentangle_cross_strides requires disentangle_levels')
        if self.disentangle_cross_strides and self.disentangle_mode != 'fuse':
            raise ValueError('disentangle_cross_strides requires disentangle_mode=fuse')
        if set(self.disentangle_cross_strides) - set(self.disentangle_levels):
            raise ValueError('every disentangle cross stride must also be a disentangle level')


@dataclass(frozen=True)
class DatasetConfig(ConfigSection):
    image_size: int = 640
    rgb_uint8_transport: bool = True
    # A storage location, never an enable/disable switch. The manifest is mandatory.
    protocol_path: str = DEFAULT_PROTOCOL_PATH

    def __post_init__(self):
        if type(self.rgb_uint8_transport) is not bool:
            raise ValueError('dataset.rgb_uint8_transport must be a boolean')
        if type(self.image_size) is not int or self.image_size < 8 or self.image_size % 8:
            raise ValueError('dataset.image_size must be divisible by 8 and at least 8')
        _non_empty_str(self.protocol_path, 'dataset.protocol_path')


@dataclass(frozen=True)
class TrainConfig(ConfigSection):
    device: str = 'cuda'
    devices: str | tuple[int, ...] = 'auto'
    distributed_backend: str = 'auto'
    workers: int = 10
    encoder_lr: float = 1e-4
    jpeg_lr: float = 3e-4
    head_lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 6
    samples_per_epoch: int = 24000
    full_pass_epochs: int = 0
    foreach_grad_normalization: bool = False
    train_all_data: bool = False
    negative_fraction: float = .25
    batch_size: int = 4
    grad_accum_steps: int = 4
    warmup_fraction: float = .05
    min_lr_factor: float = .02
    amp: str = 'bf16'
    ema_decay: float = .999
    grad_clip: float = 1.0
    resume: bool = False
    # Model weights only; relative paths are resolved against paths.runs_path.
    finetune_from: str | None = None
    finetune_weights: str = 'model'
    plain_nonunit_focus: bool = False
    reference_lr: float = 3e-4

    def __post_init__(self):
        self.validate()

    def validate(self):
        if type(self.foreach_grad_normalization) is not bool:
            raise ValueError('foreach_grad_normalization must be boolean')
        if type(self.train_all_data) is not bool:
            raise ValueError('train_all_data must be boolean')
        if self.train_all_data and (self.full_pass_epochs != self.epochs or self.plain_nonunit_focus):
            raise ValueError('train_all_data requires all epochs to be full passes and no domain focus')
        if type(self.plain_nonunit_focus) is not bool:
            raise ValueError('plain_nonunit_focus must be boolean')
        if self.devices != 'auto':
            if (not isinstance(self.devices, (list, tuple)) or not self.devices
                    or any(type(index) is not int or index < 0 for index in self.devices)
                    or len(set(self.devices)) != len(self.devices)):
                raise ValueError('train.devices must be auto or a nonempty list of unique GPU indices')
            object.__setattr__(self, 'devices', tuple(self.devices))
            if self.device != 'cuda':
                raise ValueError('Explicit train.devices requires train.device=cuda')
        if self.distributed_backend not in {'auto', 'nccl', 'gloo'}:
            raise ValueError('train.distributed_backend must be auto, nccl or gloo')
        if not isinstance(self.device, str) or not (self.device in {'cpu', 'mps'} or self.device.startswith('cuda')):
            raise ValueError("train.device must be 'cpu', 'mps', 'cuda' or 'cuda:<index>'")
        for name in ('epochs', 'samples_per_epoch', 'batch_size', 'grad_accum_steps'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'train.{name} must be a positive integer')
        if type(self.workers) is not int or self.workers < 0:
            raise ValueError('train.workers must be a nonnegative integer')
        if type(self.full_pass_epochs) is not int or not 0 <= self.full_pass_epochs <= self.epochs:
            raise ValueError('train.full_pass_epochs must be an integer in [0, epochs]')
        for name in ('encoder_lr', 'jpeg_lr', 'head_lr', 'reference_lr'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'train.{name} must be finite and positive')
        for name in ('weight_decay', 'grad_clip'):
            _nonnegative(getattr(self, name), f'train.{name}')
        for name in ('warmup_fraction', 'min_lr_factor'):
            _check_probability_grid((getattr(self, name),), f'train.{name}')
        if not 0 < self.negative_fraction < 1:
            raise ValueError('train.negative_fraction must be in (0, 1)')
        if self.finetune_weights not in {'model', 'ema'}:
            raise ValueError('finetune_weights must be model or ema')
        if not 0 <= self.ema_decay < 1:
            raise ValueError('train.ema_decay must be in [0, 1)')
        if self.amp not in {'off', 'fp16', 'bf16'}:
            raise ValueError("train.amp must be 'off', 'fp16' or 'bf16'")
        if type(self.resume) is not bool:
            raise ValueError('train.resume must be boolean')
        if self.finetune_from is not None:
            _non_empty_str(self.finetune_from, 'train.finetune_from')


@dataclass(frozen=True)
class EvalConfig(ConfigSection):
    n_bins: int = 256
    selection_small_mask_weight: float = 1.6
    mask_thresholds: tuple[float, ...] = DEFAULT_MASK_GRID
    cls_thresholds: tuple[float, ...] = (0., .2, .4, .5, .6, .7, .8, .9, .95)
    min_areas: tuple[float, ...] = (0.,)

    def __post_init__(self):
        if isinstance(self.selection_small_mask_weight, bool) or not math.isfinite(self.selection_small_mask_weight) or self.selection_small_mask_weight <= 0:
            raise ValueError('eval.selection_small_mask_weight must be finite and positive')
        if type(self.n_bins) is not int or self.n_bins <= 0:
            raise ValueError('eval.n_bins must be a positive integer')
        for name in ('mask_thresholds', 'cls_thresholds', 'min_areas'):
            values = tuple(float(value) for value in getattr(self, name))
            object.__setattr__(self, name, values)
            _check_probability_grid(values, f'eval.{name}')


@dataclass(frozen=True)
class LossConfig(ConfigSection):
    dice_weight: float = 1.0
    aux_weight: float = .4
    patch_weight: float = 0.
    edge_weight: float = 0.
    edge_band: int = 3
    edge_max_pos_weight: float = 50.
    reference_weight: float = 0.
    boundary_weight: float = 0.
    boundary_radius: int = 4

    def __post_init__(self):
        for name in ('dice_weight', 'aux_weight', 'patch_weight', 'edge_weight', 'reference_weight', 'boundary_weight'):
            _nonnegative(getattr(self, name), f'loss.{name}')
        if type(self.edge_band) is not int or self.edge_band < 1 or not self.edge_band % 2:
            raise ValueError('loss.edge_band must be a positive odd integer')
        if not math.isfinite(self.edge_max_pos_weight) or self.edge_max_pos_weight < 1:
            raise ValueError('loss.edge_max_pos_weight must be finite and at least 1')
        if type(self.boundary_radius) is not int or self.boundary_radius < 1:
            raise ValueError('loss.boundary_radius must be a positive integer')


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    paths: PathsConfig = field(default_factory=lambda: PathsConfig.from_dict({}))
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    @classmethod
    def from_dict(cls, data, *, base_dir='.'):
        data = _mapping(data, 'experiment')
        _check_keys(data, {'run_name'}, 'experiment', optional={
            'paths', 'seed', 'model', 'augmentation', 'dataset', 'train', 'eval', 'loss', 'pipeline_version'})
        if data.get('pipeline_version', PIPELINE_VERSION) != PIPELINE_VERSION:
            raise ValueError('Unsupported pipeline_version; migrate historical snapshots explicitly')
        return cls(
            run_name=_non_empty_str(data['run_name'], 'run_name'),
            paths=PathsConfig.from_dict(data.get('paths', {}), base_dir=Path(base_dir)),
            seed=data.get('seed', 42),
            model=ModelConfig.from_dict(data.get('model', {})),
            augmentation=_augmentation_from_dict(data.get('augmentation', {})),
            dataset=DatasetConfig.from_dict(data.get('dataset', {})),
            train=TrainConfig.from_dict(data.get('train', {})),
            eval=EvalConfig.from_dict(data.get('eval', {})),
            loss=LossConfig.from_dict(data.get('loss', {})),
        )

    def __post_init__(self):
        _non_empty_str(self.run_name, 'run_name')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if self.augmentation.final_full_frame_epochs > self.train.epochs:
            raise ValueError('augmentation.final_full_frame_epochs must not exceed train.epochs')
        if self.train.full_pass_epochs > self.augmentation.final_full_frame_epochs:
            raise ValueError('train.full_pass_epochs requires matching final_full_frame_epochs')
        if (self.loss.patch_weight or self.loss.edge_weight) and not self.model.disentangle_levels:
            raise ValueError('patch/edge supervision requires model.disentangle_levels')
        if self.loss.reference_weight and not self.model.pristine_reference:
            raise ValueError('reference supervision requires model.pristine_reference')

    def to_dict(self):
        result = {'pipeline_version': PIPELINE_VERSION, 'run_name': self.run_name,
                  'paths': self.paths.to_dict(), 'seed': self.seed}
        for name in ('model', 'dataset', 'train', 'eval', 'loss'):
            result[name] = getattr(self, name).to_dict()
        result['augmentation'] = asdict(self.augmentation)
        return result

    def to_flat_dict(self):
        plain = {'pipeline_version': PIPELINE_VERSION, 'run_name': self.run_name,
                 **self.paths.to_dict(), 'seed': self.seed}
        for name in ('model', 'dataset', 'train', 'eval', 'loss'):
            plain.update(getattr(self, name).to_dict())
        plain.update(asdict(self.augmentation))
        return plain


class SnapshotAdapter:
    """Read selected historical JPEG/local checkpoints, never resurrect experiments."""

    RENAMES = {
        'model': {'encoder_name': 'encoder', 'forensic_channels': 'jpeg_channels'},
        'train': {'fmap_lr': 'jpeg_lr', 'lr': 'head_lr', 'epoch_size': 'samples_per_epoch',
                  'full_train_epochs': 'full_pass_epochs', 'accum_steps': 'grad_accum_steps',
                  'warmup_frac': 'warmup_fraction'},
        'eval': {'small_mask_weight': 'selection_small_mask_weight'},
    }

    @classmethod
    def normalize(cls, snapshot):
        version = snapshot.get('pipeline_version')
        if version not in {PIPELINE_VERSION, 'emcad_v1'}:
            raise ValueError('Unsupported historical pipeline; use codex/emcad-baseline')
        if version == 'emcad_v1':
            model = snapshot.get('model', snapshot)
            required = {'forensic_mode': 'jpeg', 'fusion_variant': 'local',
                        'jpeg_variant': 'baseline', 'norm': 'batch', 'sync_batchnorm': True,
                        'use_forensics': True, 'dct_aux_weight': 0., 'forensic_contrastive_dim': 0,
                        'local_image_size': 0, 'luma_image_size': 0, 'wavelet_image_size': 0,
                        'strided_resize': False, 'noise_encoder_name': None, 'decoder_kwargs': {}}
            # Missing historical mode/fusion/SyncBN keys have their original defaults.
            old_defaults = {**required, 'forensic_mode': 'maps', 'fusion_variant': 'baseline', 'sync_batchnorm': False}
            if any(model.get(k, old_defaults[k]) != v for k, v in required.items()):
                raise ValueError('Unsupported historical architecture; use codex/emcad-baseline')
            if snapshot.get('dataset', snapshot).get('resize_mode', 'stretch') != 'stretch':
                raise ValueError('Unsupported historical resize mode')
            train = snapshot.get('train', snapshot)
            if train.get('jpeg_pair_training', False) or train.get('sampling_strategy', 'negative_fraction') != 'negative_fraction':
                raise ValueError('Unsupported historical training recipe')
            loss = snapshot.get('loss', snapshot)
            fixed = {'dice_scope': 'all', 'pixel_loss': 'bce', 'dice_area_reference': 0.,
                     'boundary_weight': 0., 'forensic_intra_weight': 0.}
            if any(loss.get(k, v) != v for k, v in fixed.items()):
                raise ValueError('Unsupported historical loss recipe')
            auxiliary = model.get('aux_weight', .4)
            override = loss.get('aux_loss_weight')
            if override is not None and (auxiliary > 0) != (override > 0):
                raise ValueError('Unsupported historical auxiliary head/loss combination')
        result = {'pipeline_version': PIPELINE_VERSION,
                  'run_name': snapshot.get('run_name', snapshot.get('paths', {}).get('run_name', 'inference')),
                  'seed': snapshot.get('seed', 42)}
        for section, schema in [('paths', PathsConfig), ('model', ModelConfig), ('dataset', DatasetConfig),
                                ('train', TrainConfig), ('eval', EvalConfig), ('loss', LossConfig),
                                ('augmentation', AugmentationConfig)]:
            values = dict(snapshot.get(section, snapshot))
            if version == PIPELINE_VERSION and section in snapshot:
                _check_keys(values, set(), section, optional={item.name for item in fields(schema)})
            if version == 'emcad_v1':
                for before, after in cls.RENAMES.get(section, {}).items():
                    if before in values:
                        values[after] = values.pop(before)
                if section == 'loss':
                    values['aux_weight'] = snapshot.get('model', snapshot).get('aux_weight', .4)
                    if values.get('aux_loss_weight') is not None:
                        values['aux_weight'] = values['aux_loss_weight']
            result[section] = {item.name: values[item.name] for item in fields(schema) if item.name in values}
        return result


class _ConfigLoader:
    """Single-parent YAML inheritance; resolved snapshots have no parent dependency."""

    def load(self, path: Path, chain=()):
        path = path.resolve()
        if path in chain:
            raise ValueError('Cyclic config inheritance: ' + ' -> '.join(map(str, (*chain, path))))
        with path.open(encoding='utf-8') as stream:
            data = dict(_mapping(yaml.safe_load(stream), str(path)))
        if 'extends' not in data:
            return data
        parent = data.pop('extends')
        if not isinstance(parent, str) or not parent.strip():
            raise ValueError(f'extends in {path} must be a non-empty string')
        return self._merge(self.load(path.parent / parent, (*chain, path)), data)

    def _merge(self, base, overrides):
        merged = dict(base)
        for key, value in overrides.items():
            merged[key] = self._merge(merged[key], value) if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping) else value
        return merged


class RuntimeEnvironment:
    """Machine overrides for recipes; snapshot deserialization stays unchanged."""

    FIELDS = {'batch_size': int, 'grad_accum_steps': int, 'amp': str,
              'device': str, 'workers': int, 'devices': yaml.safe_load,
              'distributed_backend': str}

    def apply(self, data):
        settings = {**dotenv_values(global_config.PROJECT_ROOT / '.env'), **os.environ}
        train = dict(_mapping(data.get('train', {}), 'train'))
        for name, cast in self.FIELDS.items():
            key = 'AIIJC_' + ('ACCUM_STEPS' if name == 'grad_accum_steps' else name.upper())
            value = settings.get(key)
            if value is None or not value.strip():
                continue
            try:
                train[name] = cast(value.strip())
            except (ValueError, yaml.YAMLError) as exc:
                raise ValueError(f'{key} must be a valid {cast.__name__}') from exc
        return {**data, 'train': train}


def load_experiment_config(path):
    path = Path(path)
    data = RuntimeEnvironment().apply(_ConfigLoader().load(path))
    return ExperimentConfig.from_dict(data, base_dir=path.parent)


def _augmentation_from_dict(data):
    data = dict(_mapping(data, 'augmentation'))
    _check_keys(data, set(), 'augmentation', optional={f.name for f in fields(AugmentationConfig)})
    data = _float_fields(AugmentationConfig, data)
    for name in ('crop_scale_range', 'jpeg_recompression_quality_range'):
        if name in data:
            cast = float if name == 'crop_scale_range' else int
            data[name] = tuple(cast(value) for value in data[name])
    config = AugmentationConfig(**data)
    if len(config.crop_scale_range) != 2 or not 0 < config.crop_scale_range[0] <= config.crop_scale_range[1] <= 1:
        raise ValueError('augmentation.crop_scale_range must satisfy 0 < min <= max <= 1')
    if len(config.jpeg_recompression_quality_range) != 2 or not 1 <= config.jpeg_recompression_quality_range[0] <= config.jpeg_recompression_quality_range[1] <= 100:
        raise ValueError('augmentation.jpeg_recompression_quality_range must satisfy 1 <= min <= max <= 100')
    _check_probability_grid((config.jpeg_recompression_probability,), 'augmentation.jpeg_recompression_probability')
    return config


def _float_fields(cls, data):
    """PyYAML can read scientific notation such as 1e-4 as a string."""
    result = dict(data)
    for item in fields(cls):
        if item.name in result and item.type in (float, 'float'):
            result[item.name] = float(result[item.name])
    return result


def _mapping(data, section):
    if not isinstance(data, Mapping):
        raise TypeError(f'{section} must be a mapping')
    return data


def _check_keys(data, allowed, section, *, optional=frozenset()):
    unknown = sorted(set(data) - allowed - optional)
    if unknown:
        raise ValueError(f"unknown keys in {section}: {', '.join(unknown)}")
    missing = sorted(allowed - set(data))
    if missing:
        raise ValueError(f"missing keys in {section}: {', '.join(missing)}")


def _non_empty_str(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value.strip()


def _nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and non-negative')


def _path(value, base_dir):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def _check_probability_grid(values, name):
    if not values or any(not 0 <= v <= 1 for v in values):
        raise ValueError(f'{name} must contain probabilities in [0, 1]')


__all__ = ['AugmentationConfig', 'DatasetConfig', 'EvalConfig', 'ExperimentConfig', 'ModelConfig',
           'LossConfig', 'PathsConfig', 'TrainConfig', 'load_experiment_config', 'PIPELINE_VERSION']
