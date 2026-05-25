from models.backend_selector import choose

def test_choose():
    choice, g = choose(force="cpu")
    assert choice.backend == "llamacpp"
