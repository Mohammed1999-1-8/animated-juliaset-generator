import matplotlib.pyplot as plt
import numpy as np
import numba
@numba.njit(parallel = True)
def julia(c,X,Y):
	plane = np.zeros((X,Y),dtype = numba.complex128)
	result = np.zeros((X,Y))
	mask = np.zeros((X,Y),dtype = numba.boolean)
	C = np.multiply(np.ones((X,Y),dtype = numba.complex128),c)
	for i in range(X):
		D = min(X,Y)
		for k in range(Y):
			plane[i][k] = complex(2*(i - X/2)/D, 2*(k - Y/2)/D )
	depth = 50
	threashold = 100
	for i in range (depth):
		plane = np.multiply(plane,plane) + C
		abs_pic = np.absolute(plane)
		recent_blowup = np.logical_xor(np.logical_or((abs_pic >= threashold),mask),mask)
		aux = np.multiply(recent_blowup,i)
		result = result + aux
		mask = (abs_pic > threashold) | mask 
	return (result)
for i in range (-10,10):
	a = julia(complex(0.3,(0.6+i/1000)),512,1024)
	plt.pause(0.024)
	plt.clf()
	plt.imshow(a)
	plt.draw()
