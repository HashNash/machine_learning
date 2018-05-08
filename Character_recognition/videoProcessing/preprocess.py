import cv2
import numpy as np
import os

def getExtreme(img):
	assert (len(img.shape)==2)
	#get maximum and minimum pixel value in image
	mini = 255
	maxi = 0
	for i in img:
		for j in i:
			if j>maxi:
				maxi = j
			if j<mini:
				mini = j

	return mini, maxi

#takes only gray scale 2D image
def binarize(img, upper, deviation):
	assert (len(img.shape)==2)
	binImg = []
	Thresh = upper-deviation
	for i in img:
		x=[]
		for j in i:
			if (j<=Thresh): 
				x.append(0)
			else:
				x.append(255)
		binImg.append(x)
	return np.array(binImg)


#preprocessing function which returns binarized img of specified width and height
def preprocess(img, width, height):

	x = cv2.resize(img, (width, height))

	lower, upper = getExtreme(x)
	x = binarize(x, upper, 50)

	return x


def main():


	#define resizing scale
	rx = 20
	ry = 30

	src = '9/'
	dest = 'pp9/'

	imgList = os.listdir(src)

	for name in imgList:
		#read gray scale image
		img = cv2.imread(src+name, 0)
		img = preprocess(img, rx, ry)
		cv2.imwrite(dest+name, img)
	'''
	img = cv2.resize(img, (rx,ry))
	#get extreme values in image
	lower, upper = getExtreme(img)

	new_image = cv2.resize(img, (rx,ry))


	cv2.imwrite('new.jpg', new_image)

	binImg = binarize(new_image, upper, 50)
	print binImg
	cv2.imwrite('bin.jpg', binImg)
	cv2.imwrite('binC.jpg', 255-binImg)
	'''
if __name__ == '__main__':
	main()
