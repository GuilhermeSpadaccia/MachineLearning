from math import sqrt

'''
coeficient of pearson

All information needed is:
x
y
x**2
y**2
x*y
'''


def pearson(dataset):
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

	#print(sumx)
	#print(sumy)
	#print(sumx2)
	#print(sumy2)
	#print(sumxy)
	#print(n)
	print(r)

	return(r)


def pearsonTest():
	dataset = [[1,5],
           [2,12],
           [3,16],
           [4,22],
           [5,34],
           [6,38],
           [7,41],
           [8,45],
           [9,50]]

	r = pearson(dataset)
	rexpected = 0.988650972613

	if(abs(r - rexpected) < 0.0000001):
		print("Correlation correct")
	else:
		print("Correlation wrong")

	print("Expected value: " + str(0.988650972613))
	print("Recieved value: " + str(r))

pearsonTest()