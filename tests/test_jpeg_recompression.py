import cv2
import numpy as np
import pytest

from src.data.augmentation.base import AugmentationConfig, AugmentationStage
from src.data.augmentation.pipeline import AugmentationPipeline
from src.data.augmentation.transforms.random_jpeg_recompression import RandomJPEGRecompression
from src.data.data_sample import DataSample
from src.forensic.jpeg_input import JPEGInput


def encode(image, quality):
    ok, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                             [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buffer.tobytes()


def make_sample(shape=(48, 64), *, native=True):
    image = np.random.default_rng(17).integers(0, 256, (*shape, 3), dtype=np.uint8)
    mask = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    jpeg = JPEGInput.read(encode(image, 70), include_coefficients=True) if native else None
    return DataSample(image=image, mask=mask, qtable=None if jpeg is None else jpeg.qtable,
                      jpeg=jpeg)


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('seed', [0, 1, 7, 42])
def test_shift_crops_rgb_and_gt_before_encoding_and_rebuilds_native_jpeg(native, seed):
    sample = make_sample(native=native)
    original_image, original_mask = sample.image.copy(), sample.mask.copy()
    transform = RandomJPEGRecompression((90, 91), 1.0, grid_shift_probability=1.0)
    output = transform.apply(sample, np.random.default_rng(seed))

    top = sample.image.shape[0] - output.image.shape[0]
    left = sample.image.shape[1] - output.image.shape[1]
    assert 0 <= top <= 7 and 0 <= left <= 7 and (top or left)
    np.testing.assert_array_equal(output.mask, original_mask[top:, left:])
    expected_bytes = encode(original_image[top:, left:], 90)
    expected_rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(expected_bytes, np.uint8),
                                           cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(output.image, expected_rgb)
    expected_jpeg = JPEGInput.read(expected_bytes, include_coefficients=True)
    np.testing.assert_array_equal(output.qtable, expected_jpeg.qtable)
    if native:
        assert output.jpeg.geometry == (0, 0, *output.mask.shape, 0, 0, 0)
        assert output.jpeg.source_size == output.mask.shape
        assert output.jpeg.orientation == 1
        np.testing.assert_array_equal(output.jpeg.bins, expected_jpeg.bins)
        np.testing.assert_array_equal(output.jpeg.coefficients, expected_jpeg.coefficients)
    np.testing.assert_array_equal(sample.image, original_image)
    np.testing.assert_array_equal(sample.mask, original_mask)
    assert output.image.flags.c_contiguous and output.mask.flags.c_contiguous


@pytest.mark.parametrize('seed,branch', [(0, 'untouched'), (2, 'aligned'), (34, 'shifted')])
def test_pipeline_uses_unconditional_shift_probability(seed, branch):
    sample = make_sample()
    pipeline = AugmentationPipeline(AugmentationConfig(
        jpeg_recompression_probability=0.3, jpeg_grid_shift_probability=0.05,
        jpeg_recompression_quality_range=(90, 91)))
    output = pipeline.apply(AugmentationStage.BEFORE_FORENSICS, sample,
                            np.random.default_rng(seed))
    if branch == 'untouched':
        assert output is sample
    elif branch == 'aligned':
        assert output.image.shape == sample.image.shape
        assert output.jpeg is not sample.jpeg
        np.testing.assert_array_equal(output.mask, sample.mask)
    else:
        assert output.image.shape != sample.image.shape


def test_disabled_shift_preserves_legacy_encoding_and_rng_sequence():
    sample = make_sample()
    rng, reference = np.random.default_rng(8), np.random.default_rng(8)
    reference.random()  # Existing recompression gate.
    quality = int(reference.integers(60, 100))
    expected = encode(sample.image, quality)
    output = RandomJPEGRecompression((60, 100), 1.0).apply(sample, rng)
    np.testing.assert_array_equal(output.jpeg.bins, JPEGInput.read(expected).bins)
    np.testing.assert_array_equal(output.qtable, JPEGInput.read(expected).qtable)
    assert output.mask is sample.mask
    assert rng.random() == reference.random()


@pytest.mark.parametrize('shape', [(8, 8), (8, 13), (13, 8), (9, 9)])
def test_shift_leaves_at_least_one_dct_block_and_supports_absent_masks(shape):
    sample = make_sample(shape, native=False)
    sample = DataSample(image=sample.image)
    output = RandomJPEGRecompression((90, 91), 1.0, grid_shift_probability=1.0).apply(
        sample, np.random.default_rng(9))
    assert min(output.image.shape[:2]) >= 8
    assert output.mask is None
    if shape == (8, 8):
        assert output.image.shape == sample.image.shape


@pytest.mark.parametrize('probability,shift', [(0.3, -0.1), (0.3, 0.4), (0.0, 0.1),
                                              (1.1, 0.0), (0.3, float('nan'))])
def test_invalid_shift_probabilities_are_rejected(probability, shift):
    with pytest.raises(ValueError, match='probability'):
        AugmentationConfig(jpeg_recompression_probability=probability,
                           jpeg_grid_shift_probability=shift)
    with pytest.raises(ValueError, match='probability'):
        RandomJPEGRecompression((80, 96), probability, grid_shift_probability=shift)
