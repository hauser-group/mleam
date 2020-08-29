import numpy as np
from mlpot.descriptors import DescriptorSet
import json

ds = DescriptorSet(['Ni', 'Pt'])
ds.add_Artrith_Kolpak_set()

types = ['Ni']*5 + ['Pt']*8
xyzs = np.array([-0.014480168, 0.034207338, 3.241557994,
                 2.898730059, -1.086507438, -0.962365452,
                 0.628479665, -3.011745934, 1.021810156,
                 -1.106421727, -0.938238334, 0.966816473,
                 0.014835984, -0.034293064, -3.241593337,
                 -2.898807639, 1.086543157, 0.962013396,
                 2.255579722, 1.959448215, 1.257961041,
                 1.106258420, 0.938154989, -0.966654169,
                 -2.255797799, -1.959412496, -1.257693960,
                 -0.628657260, 3.011781654, -1.021547837,
                 0.767345025, 2.434319406, 3.240815029,
                 2.475190116, 0.450660540, 3.269695405]).reshape((-1, 3))

Gs, dGs = ds.eval_with_derivatives(types, xyzs)
Gs = [Gi.tolist() for Gi in Gs]
dGs = [dGi.tolist() for dGi in dGs]

with open('NiPt13.json', 'w') as fout:
    json.dump({'types': types, 'Gs': Gs, 'dGs': dGs}, fout)
