from mleam.models import FinnisSinclair
from simple_model_training import train_model


if __name__ == "__main__":
    params = {
        ("d", "NiNi"): 3.69,
        ("A", "NiNi"): 1.3**2,
        ("beta", "NiNi"): 0.0,
        ("c", "NiNi"): 2.7,
        ("c0", "NiNi"): 47,
        ("c1", "NiNi"): -33,
        ("c2", "NiNi"): 6.0,
        ("d", "NiPt"): 4.0,
        ("A", "NiPt"): 1.3**2,
        ("beta", "NiPt"): 0.0,
        ("c", "NiPt"): 3.0,
        ("c0", "NiPt"): 47,
        ("c1", "NiPt"): -33,
        ("c2", "NiPt"): 6.0,
        ("d", "PtPt"): 4.400,
        ("A", "PtPt"): 1.3**2,
        ("beta", "PtPt"): 0.0,
        ("c", "PtPt"): 3.25,
        ("c0", "PtPt"): 47.1349,
        ("c1", "PtPt"): -33.767,
        ("c2", "PtPt"): 6.254,
    }

    hypers = {"beta_trainable": False, "offset_trainable": False}
    train_model(
        FinnisSinclair,
        "FinnisSinclair/example/",
        params,
        hypers,
        epochs=200,
    )
