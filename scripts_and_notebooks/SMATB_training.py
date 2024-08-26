from mleam.models import SMATB
from simple_model_training import train_model

if __name__ == "__main__":
    # From Cheng et al.
    params = {
        ("A", "NiNi"): 0.0845,
        ("A", "NiPt"): 0.1346,
        ("A", "PtPt"): 0.1602,
        ("p", "NiNi"): 11.73,
        ("p", "NiPt"): 14.838,
        ("p", "PtPt"): 13.00,
        ("xi", "NiNi"): 1.405,
        ("xi", "NiPt"): 2.3338,
        ("xi", "PtPt"): 2.1855,
        ("q", "NiNi"): 1.93,
        ("q", "NiPt"): 3.036,
        ("q", "PtPt"): 3.13,
    }

    hypers = {"r0_trainable": False, "offset_trainable": False}
    train_model(
        SMATB,
        "SMATB/example/",
        params,
        hypers,
        epochs=200,
    )
