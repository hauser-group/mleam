import numpy as np
from mleam.models import Ackland
from simple_model_training import train_model


if __name__ == "__main__":
    r0_NiNi = 2.491
    r0_NiPt = 2.633
    r0_PtPt = 2.775

    params = {
        ("r_k", "NiNi"): r0_NiNi * np.sqrt(np.arange(6) + 1),
        ("r_k", "NiPt"): r0_NiPt * np.sqrt(np.arange(6) + 1),
        ("r_k", "PtPt"): r0_PtPt * np.sqrt(np.arange(6) + 1),
        ("a_k", "NiNi"): np.array([-0.1850, 0.2698, -0.3334, 0.8480, -0.8979, 0.3790]),
        ("a_k", "NiPt"): np.array([0.2303, 1.1475, -0.03636, 0.5315, -0.2854, 0.1404]),
        ("a_k", "PtPt"): np.array([0.04378, 0.8517, -0.4155, 0.8331, -0.5546, 0.2275]),
        ("R_k", "NiNi"): r0_NiNi * np.sqrt(np.arange(6) + 1),
        ("R_k", "NiPt"): r0_NiPt * np.sqrt(np.arange(6) + 1),
        ("R_k", "PtPt"): r0_PtPt * np.sqrt(np.arange(6) + 1),
        ("A_k", "NiNi"): np.array([4.059, 0.4585, 0.07906, 0.1371, -0.1045, 0.04742]),
        ("A_k", "NiPt"): np.array([9.574, 0.8252, 0.01747, 0.1372, -0.06684, 0.03361]),
        ("A_k", "PtPt"): np.array(
            [3.414, 3.840e-1, -1.869e-3, 1.682e-1, -1.322e-1, 6.637e-02]
        ),
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
