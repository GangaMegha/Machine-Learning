import matplotlib.pyplot as plt 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

epsilon = [0.9, 0.9, 0.9, 0.9, 0.9, 0.01, 0.09, 0.09, 0.09, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.05,
			0.05, 0.05, 0.01, 0.005, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7]
minpts = [2, 6, 10, 20, 40, 2, 2, 10, 20, 20, 40, 10, 4, 4, 10, 20, 40, 20, 10, 4, 4, 2, 10, 30, 4, 4,
			10, 40]
purity = [26.0054, 26.0054, 26.0054, 26.0054, 26.0054, 22.7882, 26.0054, 4.5576,  0,
			0.8043, 0, 23.8606, 26.0054,  26.0054, 26.0054, 26.0054, 26.0054, 5.63, 0, 12.6005, 0,
			1.0724, 26.0054, 26.0054, 26.0054, 26.0054, 26.0054, 26.0054      ]
purity = np.ones_like(purity)*100 - purity

fig_train = plt.figure(1)
ax_train = Axes3D(fig_train, rect=[0, 0, .95, 1], elev=4, azim=116)
ax_train.scatter(epsilon, minpts, purity, cmap=plt.cm.spectral, edgecolor='k')
plt.title(r"Purity vs $\epsilon$ & MinPts")
plt.xlabel(r"$\epsilon$")
plt.ylabel("MinPts")

fig2 = plt.figure(2)
plt.scatter(epsilon,purity)
plt.title(r"Purity vs $\epsilon$ ")
plt.xlabel(r"$\epsilon$")
plt.ylabel("Purity")

fig3 = plt.figure(3)
plt.scatte(rminpts,purity)
plt.title(r"Purity vs MinPts")
plt.xlabel("MinPts")
plt.ylabel("Purity")
plt.show()