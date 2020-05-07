from sklearn.datasets import make_blobs
import numpy as np
from scipy import random, linalg
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
def PSD():
	M = random.rand(3,3)
	return np.matmul(M.T,M)

def main():
	make_blobs(n_samples=10, centers=3, n_features=3,  random_state=0)

	P = 10*PSD()
	a = np.random.multivariate_normal([0,0,0],P,100)
	np.save('aP',P)
	np.save('a',a)
	print(P)

	P = 10*PSD()
	b = np.random.multivariate_normal([10,10,10],P,100)
	np.save('bP',P)
	np.save('b',b)
	print(P)
	
	P = 10*PSD()
	c = np.random.multivariate_normal([-10,-10,10],P,100)
	np.save('cP',P)
	np.save('c',c)
	print(P)

	figure = plt.figure()
	axis = mplot3d.Axes3D(figure)
	axis.scatter(a[:,0], a[:,1], a[:,2],marker='+')
	axis.scatter(b[:,0], b[:,1], b[:,2],marker='o')
	axis.scatter(c[:,0], c[:,1], c[:,2],marker='*')

	plt.show()

if __name__ == '__main__':
	main()
