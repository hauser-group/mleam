from mleam.models import Johnson
from simple_model_training import train_model


if __name__ == "__main__":
    params = {
        ("A", "NiNi"): 0.0337,
        ("A", "NiPt"): 0.1441,
        ("A", "PtPt"): 0.1420,
        ("p", "NiNi"): 11.75,
        ("p", "NiPt"): 14.71,
        ("p", "PtPt"): 12.94,
        ("xi", "NiNi"): 1.459,
        ("xi", "NiPt"): 2.288,
        ("xi", "PtPt"): 2.188,
        ("q", "NiNi"): 1.92,
        ("q", "NiPt"): 3.21,
        ("q", "PtPt"): 3.19,
        ("F0", "Ni"): 0.024,
        ("F0", "Pt"): 0.017,
        ("eta", "Ni"): 0.53,
        ("eta", "Pt"): 0.51,
        ("F1", "Ni"): 0.97,
        ("F1", "Pt"): 0.97,
        ("zeta", "Ni"): 0.48,
        ("zeta", "Pt"): 0.48,
    }

    hypers = {"offset_trainable": False}
    model = train_model(
        Johnson,
        "Johnson/example/",
        params,
        hypers,
        epochs=100,
    )
    for weight in sorted(model.weights, key=lambda x: x.name):
        print(weight)
