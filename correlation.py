from math import sqrt
from numpy import genfromtxt

'''
Pearson Correlation Coefficient

For more informations about PCC: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

The function pearson() expects a CSV file with 2 columns, separeted by coma, e.g:
1,1
1,2
2,3
3,5
...


'''

def pearson(datasetFile):
	dataset = genfromtxt(datasetFile, dtype=[('x','f8'),('y','f8')], delimiter=',')

	sumx  = 0
	sumy  = 0
	sumx2 = 0
	sumy2 = 0
	sumxy = 0
	n     = len(dataset)

	for data in dataset:
		sumx += data[0]
		sumy += data[1]
		sumx2 += data[0]**2
		sumy2 += data[1]**2
		sumxy += data[0]*data[1]

	r = (n*sumxy - sumx*sumy)/(sqrt(n*sumx2-(sumx**2))*sqrt(n*sumy2-(sumy**2)))

	return(r)


def pearsonTestCase():
	r = pearson('dataset.csv')
	rexpected = 0.988650972613

	if(abs(r - rexpected) < 0.0000001):
		print("Correlation correct")
	else:
		print("Correlation wrong")

	print("Expected value: " + str(0.988650972613))
	print("Recieved value: " + str(r))


pearsonTestCase()
