from halo.config.loader import load_config
from halo.utils.provenance import config_hash
from halo.utils.seeding import set_deterministic_seed


def test_config_loads():
    cfg = load_config('configs/default.json')
    assert cfg.runtime.seed == 7
    assert cfg.data.resample_hz == 1


def test_seed_deterministic():
    a = set_deterministic_seed(17)
    b = set_deterministic_seed(17)
    assert a == b == 17


def test_config_hash_stable():
    cfg = load_config('configs/default.json').to_dict()
    h1 = config_hash(cfg)
    h2 = config_hash(cfg)
    assert h1 == h2
    assert len(h1) == 64
