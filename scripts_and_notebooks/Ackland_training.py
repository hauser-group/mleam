from simple_model_training import train_model
import numpy as np
from mleam.models import Ackland


if __name__ == "__main__":
    r0_NiNi = 2.491
    r0_NiPt = 2.633
    r0_PtPt = 2.775

    params = {
        ("r_k", "NiNi"): r0_NiNi * np.sqrt(np.arange(6) + 1),
        ("r_k", "NiPt"): r0_NiPt * np.sqrt(np.arange(6) + 1),
        ("r_k", "PtPt"): r0_PtPt * np.sqrt(np.arange(6) + 1),
        ("a_k", "NiNi"): np.array([16.25, 0.141, 0.0, 0.0, 0.0, 0.0]),
        ("a_k", "NiPt"): np.array([14.5, 0.199, 0.0, 0.0, 0.0, 0.0]),
        ("a_k", "PtPt"): np.array([12.1, 0.157, 0.0, 0.0, 0.0, 0.0]),
        ("R_k", "NiNi"): r0_NiNi * np.sqrt(np.arange(6) + 1),
        ("R_k", "NiPt"): r0_NiPt * np.sqrt(np.arange(6) + 1),
        ("R_k", "PtPt"): r0_PtPt * np.sqrt(np.arange(6) + 1),
        ("A_k", "NiNi"): np.array([4.0629, 0.5660, 0.0613, 0.0766, -0.0442, 0.0295]),
        ("A_k", "NiPt"): np.array([9.5750, 0.8250, 0.0175, 0.1372, -0.0668, 0.0336]),
        ("A_k", "PtPt"): np.array([3.4146, 0.3841, -0.0019, 0.1682, -0.1322, 0.0664]),
    }

    hypers = {"offset_trainable": False}
    model = train_model(
        Ackland,
        "IMN2024/Ackland/reference/",
        params,
        hypers,
        epochs=200,
    )
    for weight in sorted(model.weights, key=lambda x: x.name):
        print(weight)
