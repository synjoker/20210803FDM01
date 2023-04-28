import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from numba import cuda
import math

figure = plt.figure()
ax = Axes3D(figure)
figure.add_axes(ax)
sigma2 = 0.5
N=71
left=-5
bottem = left   
right=5
up = right
L=right-left
h=L/(N-1)
x=np.linspace(left,right,N)
y=np.linspace(left,right,N)
dx_array = x
dy_array = y
# X = np.arange(-10,10,0.1)
# Y = np.arange(-10,10,0.1)
X,Y = np.meshgrid(x,y)

Z = np.exp(-1*(Y*Y+X*X)/2/sigma2)

ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
plt.show()

'''global parameters'''
# space

# print(dx_array)
# iterative method: Jacobi, Gauss-Seidel, SOR, SIP, MSD, CG
iter_method="CG"
err=1e-4

# Z = np.exp(-1*(Y*Y+X*X))*4*(Y*Y+X*X-1)
# np.exp(-1*(Y*Y+X*X)/2/sigma2)


# Dirichlet-type BCs
def set_init_boundary():
    f_p0 = np.zeros((N, N))
    for i in range(N):
        f_p0[0, i] = np.exp(-1 * (y[i] ** 2 + left ** 2))  # left border
        f_p0[N - 1, i] = np.exp(-1 * (y[i] ** 2 + right ** 2))  # right border
        f_p0[i, 0] = np.exp(-1 * (x[i] ** 2 + bottem ** 2))  # bottom
        f_p0[i, N - 1] = np.exp(-1 * (x[i] ** 2 + up ** 2))  # top
    #print(f_p0)
    return f_p0


def set_source():
    S_p = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            S_p[i, j] = np.exp(-1*(y[j]**2+x[i]**2))*4*(y[j]**2+x[i]**2-1)
    #print(S_p)
    return S_p
def matrix_product(A,f_p0):
    Q=np.zeros(f_p0.shape)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[0]):
            Q[i]=Q[i]+A[i,j]*f_p0[j]
    return Q

@cuda.jit()
def update_SOR(f_p0,S_p,beta):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    Niter=0
    while 1:
        f_p=np.copy(f_p0)
        for i in range(1,N-1):
            for j in range(1,N-1):
                f_p[i,j]=beta/4*(f_p[i-1,j]+f_p[i,j-1]+f_p0[i+1,j]+f_p0[i,j+1]-h*h*S_p[i,j])+(1-beta)*f_p0[i,j]
        Niter=Niter+1
        # compare f_p and f_p0
        resi=0
        for i in range(1,N-1):
            for j in range(1,N-1):
                resi=resi+np.abs(f_p[i,j]-f_p0[i,j])/np.abs(f_p[i,j]+1e-8)
        resi=resi/(N-1)/(N-1)
        f_p0=f_p
        #print("Niter =",Niter,"resi =",resi)
        if resi<err:
            break
    # plot heatmap
    plt.figure()
    sns.heatmap(f_p0,cmap="RdBu_r").invert_yaxis()
    plt.show()
    return Niter, f_p0

"""main function"""
import time
if __name__ == '__main__':
    f_p0=set_init_boundary()
    S_p=set_source()
    iter_method = 'SOR'
    start_time =time.time()
    
    if iter_method == "SOR":
        F_p0 = cuda.to_device(f_p0)
        threadsperblock = (32, 32)
        blockspergrid_x = int(math.ceil(N / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(N / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        cuda.synchronize()
     
        sor_beta=1.9
        Niter, F_p0=update_SOR[blockspergrid, threadsperblock](F_p0,S_p,sor_beta)
        print("SOR iteration number",Niter," N =",N," residual =",err," beta =",sor_beta)
        f_p = F_p0.copy_to_host()
    else:
        print("Choose right iterative method")
        exit(-1)
    end_time = time.time()
    # analytical solution
    solu=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            solu[i,j]=np.exp(-1*(x[i]**2+y[j]**2))
    # plt.figure()
    # sns.heatmap(solu,cmap="RdBu_r").invert_yaxis()
    # plt.show()
    print("total time {}s ".format(end_time - start_time))
    plt.figure()
    sns.heatmap(solu-f_p,cmap="RdBu_r").invert_yaxis()
    plt.show()