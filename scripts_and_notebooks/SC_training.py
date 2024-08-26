from mleam.models import SuttonChen
from simple_model_training import train_model


if __name__ == "__main__":
    # Adapted from the original SuttonChen publication
    params = {
        ("n", "NiNi"): 9,
        ("n", "NiPt"): 10,
        ("n", "PtPt"): 10,
        ("m", "NiNi"): 6,
        ("m", "NiPt"): 7,
        ("m", "PtPt"): 8,
        ("r0", "PtPt"): 2.775,
        ("r0", "NiPt"): 2.633,
        ("r0", "NiNi"): 2.491,
    }

    params = {
        **params,
        ("c", "NiNi"): 3.52 * (1.5707e-2) ** (1 / params[("n", "NiNi")]),
        ("a", "NiNi"): 3.52 * (1.5707e-2 * 39.432) ** (2 / params[("m", "NiNi")]),
        ("c", "NiPt"): 3.72 * (1.7770e-2) ** (1 / params[("n", "NiPt")]),
        ("a", "NiPt"): 3.72 * (1.7770e-2 * 36.92) ** (2 / params[("m", "NiPt")]),
        ("c", "PtPt"): 3.92 * (1.9833e-2) ** (1 / params[("n", "PtPt")]),
        ("a", "PtPt"): 3.92 * (1.9833e-2 * 34.408) ** (2 / params[("m", "PtPt")]),
    }

    hypers = {"offset_trainable": False}
    train_model(
        SuttonChen,
        "SuttonChen/example/",
        params,
        hypers,
        epochs=200,
    )
