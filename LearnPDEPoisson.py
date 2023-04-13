
    
    #!/usr/bin/env python3
# 

import argparse
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import NeumannBC 
from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        单纯形网格（三角形、四面体）网格上任意次有限元方法求解 Poisson 方程
        边界条件为纯 Neumann 条件
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--GD',
        default=2, type=int,
        help='模型问题的维数, 默认求解 2 维问题.')

parser.add_argument('--nrefine',
        default=4, type=int,
        help='初始网格加密的次数, 默认初始加密 4 次.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

degree = args.degree
GD = args.GD
nrefine = args.nrefine
maxit = args.maxit

if GD == 2:
    from fealpy.pde.poisson_2d import CosCosData as PDE
elif GD == 3:
    from fealpy.pde.poisson_3d import CosCosCosData as PDE

pde = PDE()
mesh = pde.init_mesh(n=nrefine)

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.float64)

for i in range(maxit):
    space = LagrangeFiniteElementSpace(mesh, p=degree)
    NDof[i] = space.number_of_global_dofs()

    uh = space.function()
    A = space.stiff_matrix()
    F = space.source_vector(pde.source)
    bc = NeumannBC(space, pde.neumann) 

    # Here is the case for pure Neumann bc, we also need modify A
    A, F = bc.apply(F, A=A) 
    uh[:] = spsolve(A, F)[:-1] # we add a addtional dof

    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value)

    if i < maxit-1:
        mesh.uniform_refine()


if GD == 2:
    fig = plt.figure()
    axes = fig.add_subplot(projection='3d')
    uh.add_plot(axes, cmap = 'rainbow')
    # uh.add_plot(plt, cmap='rainbow')
    
elif GD == 3:
    print('The 3d function plot is not been implemented!')

showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)
show_error_table(NDof, errorType, errorMatrix)

plt.show()
