from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import linearRegression as lr

def costFunc(datasetFile, plot=False):
	#rodar a regressao e pegar os valores de a e b (esses valores sao minha hipotese)
	a, b, dataset = lr.linearRegression(datasetFile, plot)

	n = len(dataset)*1.0

	sumLR = 0

	#calcular o resultado da regressao todos os ponto no grafico e substrair o valor de y
	for data in dataset:
		linReg = a+b*data[0]
		linReg -= data[1]
		sumLR += linReg**2

	#dividir o resultado por 2n
	print(sumLR/2*n)
	

costFunc('dataset2.csv', True)