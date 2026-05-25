from models.gpu_probe import probe

def test_probe():
    info = probe()
    assert isinstance(info.available, bool)
