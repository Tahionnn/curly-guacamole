import pandas as pd
import pytest

from src.training.domain_focus import PlainNonunitFocus


def test_focus_keeps_only_target_positives_and_all_negatives():
    rows = pd.DataFrame(dict(
        role=['train'] * 5, domain=['plain', 'plain', 'coco', 'coco', 'plain'],
        q_kind=['nonunit', 'unit', 'nonunit', 'unit', 'unknown'],
        is_negative=[False, False, False, True, False], chng_img_path=list('abcde')))
    actual = PlainNonunitFocus.select(rows, '.')
    assert actual.chng_img_path.tolist() == ['a', 'd']
    rows.loc[0, 'role'] = 'development'
    with pytest.raises(ValueError, match='train rows only'):
        PlainNonunitFocus.select(rows, '.')
