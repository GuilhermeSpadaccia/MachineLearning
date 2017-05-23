

def linearRegression(dataset):
	sumx  = 0
	sumy  = 0
	sumx2 = 0
	sumy2 = 0
	sumxy = 0
	#I use this 1.0 just to trigger the float calculation
	#without it the result will be automaticly rounded
	n = len(dataset)*1.0

	for data in dataset:
		sumx += data[0]
		sumy += data[1]
		sumx2 += data[0]**2
		sumy2 += data[1]**2
		sumxy += data[0]*data[1]

	b = ((n*sumxy)-(sumx*sumy))/((n*sumx2)-(sumx**2))
	mediumx = sumx/n
	mediumy = sumy/n

	a = mediumy - b*mediumx

	print("Value of a: " + str(a))
	print("Value of b: " + str(b))

	#valueY = a+b*valueX -> here I'll predict the values
	#Example: print(a+b*3.5)


def linearRegressionTest():
	dataset = [[1,5],
	       [2,12],
	       [3,16],
	       [4,22],
	       [5,34],
	       [6,38],
	       [7,41],
	       [8,45],
	       [9,50]]

	linearRegression(dataset)

linearRegressionTest()