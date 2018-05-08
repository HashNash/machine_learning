
import ann
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys

def getDataset():
	trainingFolder = '../characters/'
	imgFList = os.listdir(trainingFolder)
	
	
	trainingNumber = sys.argv[1]	
	cmpFolder = 'pp'+trainingNumber
	x = []
	y = []
	i = 0
	for folder in imgFList:
		Y = 0	
		if folder == cmpFolder:	
			Y = 1
		imgList = os.listdir(trainingFolder+folder)
		for name in imgList:
			img = cv2.imread(trainingFolder+folder+'/'+name, 0)
			img = img.reshape((img.shape[0]*img.shape[1], 1))
			x.append(img)
			y.append(Y)
	x = np.array(x)
	y = np.array([y])
	
	x = x.reshape((x.shape[0], x.shape[1]))
	return x.T, y

def main():
	x, y = getDataset()
	#print x.shape, y.shape
	#define parameters for training
	
	trainingNumber = sys.argv[1]
	hiddenNodes = 30
	iterations = 5000
	#loadFromFile = False
	learningRate = 0.1
	#train the neural network
	w1, b1, w2, b2, costArr = ann.trainNeuralNetwork(0.01*x, y, hiddenNodes, iterations, learningRate, trainingNumber)
	np.savetxt('../w/w1_'+trainingNumber+'.txt', w1)
	np.savetxt('../w/w2_'+trainingNumber+'.txt', w2)
	np.savetxt('../b/b1_'+trainingNumber+'.txt', b1)
	np.savetxt('../b/b2_'+trainingNumber+'.txt', b2)
	
	plt.plot(costArr)
	plt.show()
	
if __name__=='__main__':
	main()
