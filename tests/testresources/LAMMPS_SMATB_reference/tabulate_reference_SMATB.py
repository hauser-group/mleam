import numpy as np

from atsim.potentials import EAMPotential, Potential
from atsim.potentials.eam_tabulation import SetFL_FS_EAMTabulation


def poly_cutoff(r, a, b):
    if r <= a:
        return 1.0
    elif r > b:
        return 0.0
    else:
        r_scaled = (r - a) / (b - a)
        return 1.0 - 10.0 * r_scaled**3 + 15 * r_scaled**4 - 6 * r_scaled**5


def smooth_cutoff(r, a, b):
    if r <= a:
        return 1.0
    elif r >= b:
        return 0.0
    else:
        return 1.0 / (1.0 + np.exp(((a - b) * (2 * r - a - b)) / ((r - a) * (r - b))))


def embedding_function(rho):
    return -np.sqrt(rho)


def make_density(xi, q, r_0, a, b):
    def rho(r):
        return xi**2 * np.exp(-2 * q * (r / r_0 - 1)) * poly_cutoff(r, a, b)

    return rho


def make_pair_pot(A, p, r_0, a, b):
    """Factor two to compensate for the 1/2 in the definition"""

    def phi(r):
        return 2 * A * np.exp(-p * (r / r_0 - 1)) * poly_cutoff(r, a, b)

    return phi


def main():
    params = {
        ("A", "PtPt"): 0.1602,
        ("A", "NiPt"): 0.1346,
        ("A", "NiNi"): 0.0845,
        ("xi", "PtPt"): 2.1855,
        ("xi", "NiPt"): 2.3338,
        ("xi", "NiNi"): 1.405,
        ("p", "PtPt"): 13.00,
        ("p", "NiPt"): 14.838,
        ("p", "NiNi"): 11.73,
        ("q", "PtPt"): 3.13,
        ("q", "NiPt"): 3.036,
        ("q", "NiNi"): 1.93,
        ("r0", "PtPt"): 2.77,
        ("r0", "NiPt"): 2.63,
        ("r0", "NiNi"): 2.49,
        ("cut_a", "PtPt"): 4.08707719,
        ("cut_b", "PtPt"): 5.0056268338740553,
        ("cut_a", "NiPt"): 4.08707719,
        ("cut_b", "NiPt"): 4.4340500673763259,
        ("cut_a", "NiNi"): 3.62038672,
        ("cut_b", "NiNi"): 4.4340500673763259,
    }
    # Create EAMPotential
    eam_potentials = [
        EAMPotential(
            "Ni",
            28,
            58.6934,
            embedding_function,
            {
                "Ni": make_density(
                    params[("xi", "NiNi")],
                    params[("q", "NiNi")],
                    params[("r0", "NiNi")],
                    params[("cut_a", "NiNi")],
                    params[("cut_b", "NiNi")],
                ),
                "Pt": make_density(
                    params[("xi", "NiPt")],
                    params[("q", "NiPt")],
                    params[("r0", "NiPt")],
                    params[("cut_a", "NiPt")],
                    params[("cut_b", "NiPt")],
                ),
            },
            latticeConstant=3.524,
            latticeType="fcc",
        ),
        EAMPotential(
            "Pt",
            78,
            195.084,
            embedding_function,
            {
                "Ni": make_density(
                    params[("xi", "NiPt")],
                    params[("q", "NiPt")],
                    params[("r0", "NiPt")],
                    params[("cut_a", "NiPt")],
                    params[("cut_b", "NiPt")],
                ),
                "Pt": make_density(
                    params[("xi", "PtPt")],
                    params[("q", "PtPt")],
                    params[("r0", "PtPt")],
                    params[("cut_a", "PtPt")],
                    params[("cut_b", "PtPt")],
                ),
            },
            latticeConstant=3.9242,
            latticeType="fcc",
        ),
    ]
    pair_potentials = [
        Potential(
            "Ni",
            "Ni",
            make_pair_pot(
                params[("A", "NiNi")],
                params[("p", "NiNi")],
                params[("r0", "NiNi")],
                params[("cut_a", "NiNi")],
                params[("cut_b", "NiNi")],
            ),
        ),
        Potential(
            "Ni",
            "Pt",
            make_pair_pot(
                params[("A", "NiPt")],
                params[("p", "NiPt")],
                params[("r0", "NiPt")],
                params[("cut_a", "NiPt")],
                params[("cut_b", "NiPt")],
            ),
        ),
        Potential(
            "Pt",
            "Pt",
            make_pair_pot(
                params[("A", "PtPt")],
                params[("p", "PtPt")],
                params[("r0", "PtPt")],
                params[("cut_a", "PtPt")],
                params[("cut_b", "PtPt")],
            ),
        ),
    ]

    # Number of grid points and cut-offs for tabulation.
    cutoff_rho = 100.0
    nrho = 10000

    cutoff = 6.0
    nr = 10000

    tabulation = SetFL_FS_EAMTabulation(
        pair_potentials, eam_potentials, cutoff, nr, cutoff_rho, nrho
    )

    with open("NiPt.eam.fs", "w") as outfile:
        tabulation.write(outfile)


if __name__ == "__main__":
    main()
