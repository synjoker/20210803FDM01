import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

folder = ['0.00','0.02','0.04','0.06','0.08','0.10','0.12','0.14','0.16','0.18','0.20','0.22','0.24','0.26','0.28','0.30','0.32','0.34',
                '0.36','0.38','0.40','0.42','0.44','0.46','0.48','0.50']
folder = ['0.0','0.5','1.0', '2.0']

# noall = ['0',  '1', '2']
noall = ['0']
import gc

for foldername in folder:
    for no in noall:
        print("{} and {} start!!".format(foldername, no))
        # mag = np.loadtxt("pic/{}/{}/mag.txt".format(foldername, no))
        # angle = np.loadtxt("pic/{}/{}/angle.txt".format(foldername, no))
        mag = np.loadtxt("{}/mag.txt".format(foldername))
        angle = np.loadtxt("{}/angle.txt".format(foldername))
        N = 50


        m = mag[0:N, 0:N]
        an = angle[0:N, 0:N]
        # print(m.shape)
        # print(an.shape)
        # print(m)

        n0=1.000129
        # C为透镜厚度、L0为镜子到背景距离
        # c = 25
        # c = 10
        c=25
        l0=8000
        k=2*n0/(2*l0+c)/c

        C = np.multiply(np.cos(an), m) + np.multiply(np.sin(an), m) 
        C = k*C
        # print(C)
        # print(C.shape)

        # print(C)
        C = np.array(C)
        C = np.reshape(C, (-1, 1))
        # print(C.shape)
        C = C.astype(np.float32)
        # print(C)

        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # N=700 # 矩阵维数
        Imat = -np.eye(N)/4
        print(C.shape)
        def Gmatrix(N): # 生成G矩阵
            mtr = np.eye(N)
            for i in range(0, N - 1):
                mtr[i][i + 1] = -1 / 4
            for i in range(1, N):
                mtr[i][i - 1] = -1 / 4
            return mtr

        def zero_mat(N,m): # 生成m个横向连接N阶零矩阵,m>=1
            zero_matrix = np.zeros((N,N))
            for i in range(m-1):
                zero_matrix = np.block([zero_matrix,np.zeros((N,N))])
            return zero_matrix

        def zero_arr(N, m):  # 生成m个纵向连接N阶列向量,m>=1
            zero_array = np.zeros((N,1))
            for i in range(m-1):
                zero_array = np.block([[zero_array],[np.zeros((N,1))]])
            return zero_array

        def Kmatrix(N,Gmat,Imat):
            Kmar_list = []
            for i in range(N):
                if i == 0:
                    Kmat_line = np.block([np.block([Gmat, Imat]),zero_mat(N,N-2)])
                    Kmar_list.append(Kmat_line)
                elif i == 1:
                    Kmat_line = np.block([Imat, Kmar_list[-1][:,:-N]])
                    Kmar_list.append(Kmat_line)
                else:
                    Kmat_line = np.block([np.zeros((N,N)), Kmar_list[-1][:,:-N]])
                    Kmar_list.append(Kmat_line)
            Kmat = Kmar_list[0]
            for j in range(N-1):
                Kmat = np.block([[Kmat],[Kmar_list[j+1]]])
            return Kmat

        def Barray(N): # 方程右边
            barr = np.ones((N,1))/4
            barr = np.block([[barr],[zero_arr(N,N-2)]])
            barr = np.block([[barr],[np.ones((N,1))/4]])
            return barr

        Gmat = Gmatrix(N)
        Kmat = Kmatrix(N,Gmat,Imat)

        print(Kmat.shape, C.shape)
        # Barr = Barray(N)
        # print(Barr.shape)
        # Phi = np.linalg.solve(Kmat,Barr)
        Phi = np.linalg.solve(Kmat, C)

        def trans(N,Phi):
            # 把线性方程组解得的列矢量分割，从左到右排列转化为方矩阵，方便可视化
            phi = Phi[0:N]
            for i in range(N-1):
                j = i+1
                phi = np.block([phi,Phi[(j*N):((j+1)*N)]])
            return phi

        Phi = trans(N,Phi)
        x = np.linspace(0,1,N)
        y = np.linspace(0,1,N)
        X,Y = np.meshgrid(x,y)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X,Y,Phi,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
        ax.contourf(X,Y,Phi,zdir='z',offset=1,cmap='rainbow')
        plt.title('Difference methods for Laplace equation',fontsize='12')
        plt.savefig("pic/{}/{}/phi2.jpg".format(foldername, no))
        # plt.show()
        np.savetxt("pic/{}/{}/phi2.txt".format(foldername, no), Phi)
        gc.collect()
