import numpy as np 
import matplotlib.pyplot as plt
import math

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigDeriv(x):
	return sigmoid(x)*(1-sigmoid(x))
def forewardProp(w1, b1, w2, b2, x):
	z1 = np.dot(w1,x)+b1
	a1 = sigmoid(z1)
	z2 = np.dot(w2,a1)+b2
	a2 = sigmoid(z2)
	#print z1.shape, z2.shape
	return z1, a1, z2, a2

def backProp(y, a2, a1, w2, z1, x, m):
	dz2 = a2-y
	
	dw2 = np.dot(dz2,(a1.T))/m
	
	db2 = np.sum(dz2, axis=1, keepdims=True)/m
	dz1 = np.dot(w2.T,dz2)*(sigDeriv(z1))
	dw1 = np.dot(dz1,x.T)/m
	db1 = np.sum(dz1, axis=1, keepdims=True)/m
	
	return dw1, db1, dw2, db2

def trainNeuralNetwork(x, y, hiddenNodes, iterations, learningRate, trainingNumber):
	featureSize = x.shape[0]
	outputSize = y.shape[0]
	m=x.shape[1]
	
	try:
		w1=np.loadtxt('../w/w1_'+trainingNumber+'.txt')
		b1=np.loadtxt('../b/b1_'+trainingNumber+'.txt')
		b1=b1.reshape((b1.shape[0],1))
		#print b1.shape
		w2=np.loadtxt('../w/w2_'+trainingNumber+'.txt')
		b2=np.loadtxt('../b/b2_'+trainingNumber+'.txt')
		
		w2=w2.reshape((1,w2.shape[0]))
		print w2.shape
		
		b2=b2.reshape((1,1))
		#print 'shape of b2='+b2.shape+'\n', 'reshaping b2 to (1,1)'
		b2=b2.reshape((1,1))
		print 'weight matrices found\nloading weight matrices from ../w/'
		print w1.shape, w2.shape, b1.shape, b2.shape, '-- Dimensions of w1, w2, b1, b2 respectively'

	except:
		print 'weight matrices not found\ninitialising weights to random values\n'
		w1 = 0.001*np.random.randn(hiddenNodes,featureSize)
		b1 = 0.001*np.random.randn(hiddenNodes,1)
		w2 = 0.001*np.random.randn(outputSize, hiddenNodes)
		b2 = 0.001*np.random.randn(outputSize, 1)
		
		print w1.shape, w2.shape, b1.shape, b2.shape, 'shapes of w1, w2, b1, b2, respectively'

	#print w1.shape, w2.shape
	
	costArr = []
	
	for i in range(iterations):
		z1,a1,z2,a2=forewardProp(w1, b1, w2, b2, x)
		dw1,db1,dw2,db2=backProp(y,a2,a1,w2,z1,x,m)
		#print a2, y.shape, x.shape
		w1 = w1 - learningRate*(dw1)
		b1 = b1 - learningRate*db1
		w2 = w2 - learningRate*(dw2)
		b2 = b2 - learningRate*db2
		#print dw1,db1
		#print dw2, db2
		costArr.append(testAnn(w1, x, b1, w2, b2, y))

	return w1, b1, w2, b2, costArr
	
def testAnn(w1, x, b1, w2, b2, y):
	z1, a1, z2, a2 = forewardProp(w1, b1, w2, b2, x)
	a = a2
	loss = -(y*np.log(a) + (1-y)*np.log(1-a))
	m = y.shape[1]
	
	l1 =[]
	for elem in loss:
		for i in elem:
			if math.isnan(i):
				l1.append(0)
			elif math.isinf(i):
				l1.append(255)
			else:
				l1.append(i)
	l1 = np.array(l1)
	#print l1
	cost = np.sum(l1)/m
	#print cost
	return  cost

if __name__ == '__main__':
	x = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
	y = np.array([[1,0,0,1,0,1,1,1]])
	
	x = x.T
	np.random.seed(2)
	#y=y.T
	w1, b1, w2, b2, costArr = trainNeuralNetwork(x, y, 5, 10000, False, 1)
	np.savetxt("../w/w1.txt",w1)
	np.savetxt("../b/b1.txt",b1)
	np.savetxt("../w/w2.txt",w2)
	np.savetxt("../b/b2.txt",b2)
	z1, a1, z2, a2 = forewardProp(w1, b1, w2, b2, x)
	plt.plot(costArr)
	plt.show()
	print a2, y
	
	
	
