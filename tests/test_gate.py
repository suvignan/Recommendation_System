"""The params.yaml gate, on hand-made numbers (no data): each rule must fail on its own when it should."""
from recsys.components.model_evaluation import gate_checks

POPULARITY = {'all': {'recall': 0.02}, 'cold': {'recall': 0.02}}


def failed(final, bad_lists=0, reference=None):
    checks = gate_checks(final, POPULARITY, bad_lists, reference, 1e-6,
                         min_lift_over_popularity=2.0, cold_must_beat_popularity=True)
    return [check['check'] for check in checks if not check['passed']]


def test_good_build_passes():
    assert failed({'all': {'recall': 0.05}, 'cold': {'recall': 0.03}}) == []          # 2.5x, cold beats popularity


def test_each_rule_fails_on_its_own():
    assert failed({'all': {'recall': 0.03}, 'cold': {'recall': 0.03}}) == ['live gate: lift over popularity']   # only 1.5x
    assert failed({'all': {'recall': 0.05}, 'cold': {'recall': 0.01}}) == ['live gate: cold beats popularity']
    assert failed({'all': {'recall': 0.05}, 'cold': {'recall': 0.03}}, bad_lists=3) == \
        ['sanity: every list = k distinct articles, none missing']
    assert failed({'all': {'recall': 0.05}, 'cold': {'recall': 0.03}}, reference=0.0495) == ['snapshot parity']


def test_parity_tolerates_only_tiny_differences():
    assert failed({'all': {'recall': 0.0495000004}, 'cold': {'recall': 0.03}}, reference=0.0495) == []
