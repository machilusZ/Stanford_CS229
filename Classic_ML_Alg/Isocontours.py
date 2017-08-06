import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
x = np.arange(-8.0, 8.0, delta)
y = np.arange(-8.0, 8.0, delta)
X, Y = np.meshgrid(x, y)

Z0 = mlab.bivariate_normal(X, Y, 1.**0.5, 2.**0.5, 1, 1,0)

plt.figure()
levels = np.arange(-8, 8, 0.01)
CS = plt.contour(X, Y, Z0, levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Problem 3a')

Z1 = mlab.bivariate_normal(X, Y, 2.**0.5,3.**0.5,-1,2,1)
plt.figure()
levels = np.arange(-8, 8, 0.01)
CS = plt.contour(X, Y, Z1, levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Problem 3b')

Z2 = mlab.bivariate_normal(X, Y, 2.**0.5, 1.**0.5, 0, 2,1)
Z3 = mlab.bivariate_normal(X, Y, 2.**0.5, 1.**0.5, 2,0, 1)
Z4 = Z2 - Z3
plt.figure()
levels = np.arange(-8, 8, 0.015)
CS = plt.contour(X, Y, Z4, levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Problem 3c')

Z5 = mlab.bivariate_normal(X, Y, 2.**0.5, 1.**0.5, 0, 2,1)
Z6 = mlab.bivariate_normal(X, Y, 2.**0.5, 3.**0.5, 2,0, 1)
Z7 = Z5 - Z6
plt.figure()
levels = np.arange(-8, 8, 0.015)
CS = plt.contour(X, Y, Z7, levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Problem 3d')

Z8 = mlab.bivariate_normal(X, Y, 2.**0.5, 1.**0.5, 1,1,0)
Z9 = mlab.bivariate_normal(X, Y, 2.**0.5, 2.**0.5, -1,-1, 1)
Z10 = Z8 - Z9
plt.figure()
levels = np.arange(-8, 8, 0.015)
CS = plt.contour(X, Y, Z10, levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Problem 3e')


plt.show()

