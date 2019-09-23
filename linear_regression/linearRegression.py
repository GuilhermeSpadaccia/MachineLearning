from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

def linearRegression(datasetFile, plot=False):
	dataset = genfromtxt(datasetFile, dtype=[('x','f8'),('y','f8')], delimiter=',')

	sumx  = 0
	sumy  = 0
	sumx2 = 0
	sumy2 = 0
	sumxy = 0
	x = []
	y = []

	n = len(dataset)*1.0

	for data in dataset:
		sumx += data[0]
		sumy += data[1]
		sumx2 += data[0]**2
		sumy2 += data[1]**2
		sumxy += data[0]*data[1]
		x.append(data[0])
		y.append(data[1])

	b = ((n*sumxy)-(sumx*sumy))/((n*sumx2)-(sumx**2))
	mediumx = sumx/n
	mediumy = sumy/n

	a = mediumy - b*mediumx

	print("Value of a: " + str(a))
	print("Value of b: " + str(b))

	minResult = a+b*min(x)
	maxResult = a+b*max(x)

	if (plot):
		plotRegression(x,y,minResult,maxResult,a)

	return (a, b, dataset)

def plotRegression(x,y,minResult,maxResult,a):
	fit = np.polyfit(x,y,1)
	fit_fn = np.poly1d(fit)

	plt.plot(x,y, 'yo')
	plt.plot([0,max(x)],[a,maxResult])
	plt.xlim(0, max(x))
	plt.ylim(0, max(y))

	plt.show()

linearRegression('dataset2.csv', True)