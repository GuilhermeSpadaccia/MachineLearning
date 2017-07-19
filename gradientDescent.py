from numpy import genfromtxt

def gradientDescent(datasetFile):
	dataset = genfromtxt(datasetFile, dtype=[('x','f8'),('y','f8')], delimiter=',')

	alpha = 1
	t0 = 0
	t1 = 0
	loop = True
	i = 0
	while(loop):
		tempt0 = t0 - alpha*deriv0(t0, t1, dataset)
		tempt1 = t1 - alpha*deriv0(t0, t1, dataset, True)

		t0 = tempt0
		t1 = tempt1

		print(t0)

		i += 1

		if(i == 50000):
			loop = False
		


def deriv0(t0, t1, dataset, stepTwo = False):
	part = 0
	results = 0
	m = len(dataset)

	for data in dataset:
		h0 = t0 + t1 * data[0]
		part = h0 - data[1]
		
		if(stepTwo):
			part = part * data[0]

		results += part

	results = results/m
	return results



gradientDescent("dataset1.csv")