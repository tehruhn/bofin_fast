from qutip import *
from heom_fmotd_NL import FermionicHEOMSolver as FermionicHEOMSolverPY
from heom_fmotd import FermionicHEOMSolver 
import numpy as np
import matplotlib.pyplot as plt

d1 = destroy(2)
e1 = 1.
H0 = e1*d1.dag()*d1 
Qops = [d1.dag(),d1,d1.dag(),d1]
rho_0 = basis(2,0)*basis(2,0).dag()

Ncc = 2
eta_list = [[(0.0025-0.0015778478288730755j), -0.0002602364577000628j, -
0.0002748355117099625j, -0.00030957579190690006j, -0.00039011715472080447j, 
-0.000952586093590817j, (-0+0.00277224236367158j), 
(-0+0.000992956474830042j)], [(0.0025-0.0015778478288730755j), -
0.0002602364577000628j, -0.0002748355117099625j, -
0.00030957579190690006j, -0.00039011715472080447j, -
0.000952586093590817j, (-0+0.00277224236367158j), 
(-0+0.000992956474830042j)], [(0.0025-0.0015778478288730755j), 
-0.0002602364577000628j, -0.0002748355117099625j, 
-0.00030957579190690006j, -0.00039011715472080447j, 
-0.000952586093590817j, (-0+0.00277224236367158j), 
(-0+0.000992956474830042j)], [(0.0025-0.0015778478288730755j), 
-0.0002602364577000628j, -0.0002748355117099625j, -
0.00030957579190690006j, -0.00039011715472080447j, -
0.000952586093590817j, (-0+0.00277224236367158j), 
(-0+0.000992956474830042j)]]

gamma_list = [[(1+1j), (0.08121642500626945+1j), (0.24364927501940756+1j),
 (0.4060825063507268+1j), (0.5691281251822908+1j), 
 (0.7622622998221965+1j), (1.1932614618196118+1j), 
 (3.4693716936476298+1j)], [(1-1j), (0.08121642500626945-1j), 
 (0.24364927501940756-1j), (0.4060825063507268-1j),
  (0.5691281251822908-1j), (0.7622622998221965-1j), 
  (1.1932614618196118-1j), (3.4693716936476298-1j)],
   [(1-1j), (0.08121642500626945-1j), (0.24364927501940756-1j), 
   (0.4060825063507268-1j), (0.5691281251822908-1j), 
   (0.7622622998221965-1j), (1.1932614618196118-1j), 
   (3.4693716936476298-1j)], [(1+1j), (0.08121642500626945+1j), 
   (0.24364927501940756+1j), (0.4060825063507268+1j),
    (0.5691281251822908+1j), (0.7622622998221965+1j),
     (1.1932614618196118+1j), (3.4693716936476298+1j)]]


import time

# CHECK PYTHON VERSION

# start = time.time()
# resultHEOM1 = FermionicHEOMSolverPY(H0, Qops,  eta_list, gamma_list, Ncc)
# end = time.time()
# print("original code",end- start)

# CHECK CPP VERSION

# start = time.time()
# resultHEOM2 = FermionicHEOMSolver(H0, Qops,  eta_list, gamma_list, Ncc)
# end = time.time()
# print("new code", end - start)

start = time.time()
resultHEOM = FermionicHEOMSolver(H0, Qops,  eta_list, gamma_list, Ncc)
end = time.time()
# print("original code",end- start)
# print(resultHEOM)

tlist = np.linspace(0,100,1000)

out1 = resultHEOM.run(rho_0, tlist)
# fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,8))
# axes.plot(tlist, expect(out1.states,rho_0), 'g', linewidth=2, label="P11")
# plt.show()
print(out1.states[-1])