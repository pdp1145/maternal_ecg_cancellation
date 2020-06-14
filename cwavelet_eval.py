# import pywt
from scipy import signal as sps
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

x = np.arange(512)
y = np.zeros([512,])
y[256] = 1.0
# y = np.sin(2*np.pi*x/32)
plt.plot(x, y)
plt.show()

widths = np.arange(1, 129)*0.75
cwtmatr = sps.cwt(y, sps.ricker, widths)
plt.imshow(cwtmatr)   # , extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()

# for ci in np.arange(0, 128):
#      plt.plot(cwtmatr[ci,:]) # doctest: +SKIP
#      plt.show() # doctest: +SKIP

#
#
# coef, freqs = pywt.cwt(y,np.arange(1,129),'morl', 1.0, 'conv')
# # plt.matshow(coef) # doctest: +SKIP
#



fig = plt.figure()
ax = plt.axes(projection='3d')

X = np.arange(0, 512)
Y = np.arange(0, 128)
X, Y = np.meshgrid(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, cwtmatr, cmap='viridis', edgecolor='none')

arf = 12
