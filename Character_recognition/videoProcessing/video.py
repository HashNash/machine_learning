import numpy as np
import cv2
import preprocess
import ann

trNumber = '1'

w1=np.loadtxt("../w/w1_"+trNumber+".txt")
b1=np.loadtxt("../b/b1_"+trNumber+".txt")
b1=b1.reshape((b1.shape[0],1))
print b1.shape
w2=np.loadtxt("../w/w2_"+trNumber+".txt")
b2=np.loadtxt("../b/b2_"+trNumber+".txt")
try:
	assert(len(w2.shape)==2)
except:
	w2=w2.reshape((1,w2.shape[0]))
	print w2.shape
try:
	b2=b2.reshape((b2.shape[0],1))
except:
	print b2.shape, 'reshaping b2 to (1,1)'
	b2=b2.reshape((1,1))


cap = cv2.VideoCapture(0)
i=0
while(True):
  	# Capture frame-by-frame
	ret, frame = cap.read()
	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Display the resulting frame
	x = preprocess.preprocess(gray, 80, 100)
	#print x
	x = x.reshape((8000,1))
	x=x*0.01
	#print x
	z1, a1, z2, y = ann.forewardProp(w1, b1, w2, b2, x)
	print y
	i=i+1 
	'''
	i=0
	for j in range(10):
		if (Y[j]>Y[i]):
			i=j
	print i
	'''
	cv2.imshow('frame', gray)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
