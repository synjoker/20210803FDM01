import numpy as np
import math
import sys
#分解矩阵
def DLU(A):
    D=np.zeros(np.shape(A))
    L=np.zeros(np.shape(A))
    U=np.zeros(np.shape(A))
    for i in range(A.shape[0]):
        D[i,i]=A[i,i]
        for j in range(i):
            L[i,j]=-A[i,j]
        for k in list(range(i+1,A.shape[1])):
            U[i,k]=-A[i,k]
    L=np.mat(L)
    D=np.mat(D)
    U=np.mat(U)
    return D,L,U
 
#迭代
def Jacobi_iterative(A,b,x0,maxN,p):  #x0为初始值，maxN为最大迭代次数，p为允许误差
    D,L,U=DLU(A)
    if len(A)==len(b):
        D_inv=np.linalg.inv(D)
        D_inv=np.mat(D_inv)
        B=D_inv * (L+U)
        B=np.mat(B)
        f=D_inv*b
        f=np.mat(f)
    else:
        print('维数不一致')
        sys.exit(0)  # 强制退出
    
    a,b=np.linalg.eig(B) #a为特征值集合，b为特征值向量
    c=np.max(np.abs(a)) #返回谱半径
    if c<1:
        print('迭代收敛')
    else:
        print('迭代不收敛')
        sys.exit(0)  # 强制退出
#以上都是判断迭代能否进行的过程，下面正式迭代
    k=0
    while k<maxN:
        x=B*x0+f
        k=k+1
        eps=np.linalg.norm(x-x0,ord=2)
        if eps<p:
            break
        else:
            x0=x
    return k,x
 
# A = np.array([[8,-3,2],[4,11,-1],[5,3,12]])
# b = np.array([[20],[33],[36]])
A = np.mat([[10,3,1],[2,-10,3],[1,3,10]])
b = np.mat([[14],[-5],[14]])
x0=np.mat([[0],[0],[0]])
maxN=100
p=0.00000001
print("原系数矩阵a:")
print(A, "\n")
print("原值矩阵b:")
print(b, "\n")
k,x=Jacobi_iterative(A,b,x0,maxN,p)
print("迭代次数")
print(k, "\n")
print("数值解")
print(x, "\n")