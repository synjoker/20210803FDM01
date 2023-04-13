import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


mag = np.loadtxt("save_20220822_013654/mag.txt")
# mag_sht = np.mean(mag)
# mag = mag-mag_sht
# angle = np.loadtxt("../pic/0.00/0/angle.txt")
# mag[np.where(mag>=0.02)]=0
# mag[np.where(mag<=-0.02)]=0

print(mag.shape)
y = np.linspace(0,1,mag.shape[0])
x = np.linspace(0,1,mag.shape[1])
X,Y = np.meshgrid(x,y)

fig = plt.figure()
ax = Axes3D(fig)
# cs = plt.contour(X, Y, mag, cmap = plt.cm.rainbow)
# plt.clabel(cs,fontsize=10)
# ax.plot_surface(X,Y,mag,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
ax.plot_surface(X,Y,mag)
# ax.contourf(X,Y,Phi,zdir='z',offset=1,cmap='rainbow')
plt.title('Difference methods for Laplace equation',fontsize='12')
plt.show()
