import matplotlib.pyplot as plt
import numpy as np
import numba
@numba.njit(parallel = True)
def julia(c,X,Y):
	plane = np.zeros((Y,X),dtype = numba.complex128)
	result = np.zeros((Y,X))
	mask = np.zeros((Y,X),dtype = numba.boolean)
	C = np.multiply(np.ones((Y,X),dtype = numba.complex128),c)
	D = min(X,Y)
	for i in range(Y):
		for k in range(X):
			plane[i][k] = complex(2*(k - X/2)/D,2*(i - Y/2)/D )
	depth = 50
	threashold = 100
	for i in range (depth):
		plane = np.multiply(plane,plane) + C
		abs_pic = np.absolute(plane)
		recent_blowup = np.logical_xor(np.logical_or((abs_pic >= threashold),mask),mask)
		aux = np.multiply(recent_blowup,i)
		result = result + aux
		mask = np.logical_or((abs_pic > threashold),mask) 
	return (result)
for i in range (0,100):
	a = julia(complex(-0.72 - i/1000,0.15),1080,720)
	plt.pause(0.004)
	plt.clf()
	plt.imshow(a)
	plt.draw()
