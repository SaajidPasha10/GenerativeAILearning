import numpy as np

from src.decoding.top_p_sampling import TopPSampling


def test_top_p():

    logits = np.array([
        0.40,
        0.30,
        0.15,
        0.10,
        0.05
    ])

    np.random.seed(42)

    token = TopPSampling.sample(
        logits,
        p=0.80,
        temperature=1.0
    )

    assert token in [0, 1, 2]