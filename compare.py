#coding=utf-8
import os
import cv2
import types
import numpy as np
labelCap = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
num=4
labelCap =	labelCap[0:num]
# filename=labelCap[2]
imgOri = cv2.imread("/Users/shixiangyang/Desktop/2.jpg")
thres = 130
samplerate = 5
reslist=[]
grayOri = cv2.cvtColor( imgOri , cv2.COLOR_BGR2GRAY )
binaryOri = grayOri
# retOri, binaryOri = cv2.threshold( grayOri,thres,255,cv2.THRESH_BINARY )

print binaryOri[0][0]
print grayOri[0][0]

for filename in labelCap:
	imgA = cv2.imread("/Users/shixiangyang/Desktop/"+filename+".jpg")
	grayA = cv2.cvtColor( imgA,cv2.COLOR_BGR2GRAY )
	# retA, binaryA = cv2.threshold( grayA,thres,255,cv2.THRESH_BINARY )
	binaryA = grayA
	# print grayOri.dtype,grayOri.shape,grayOri.size
	# (1177,775)
	# print grayA.dtype,grayA.shape,grayA.size
	# (34,30)
	maxsizeY = grayOri.shape[0]
	maxsizeX = grayOri.shape[1]
	sizeY = grayA.shape[0]
	sizeX = grayA.shape[1]

	maxscore = 0
	targetY = 0
	targetX = 0

	for y in range(0,maxsizeY-sizeY+1,samplerate):
		for x in range(0,maxsizeX-sizeX+1,samplerate):
			score = 0
			for yy in range(0,sizeY,samplerate):
				for xx in range(0,sizeX,samplerate):
					if grayOri[y+yy][x+xx] > grayA[yy][xx]:
						diff = grayOri[y+yy][x+xx] - grayA[yy][xx]
					else:
						diff = grayA[yy][xx] - grayOri[y+yy][x+xx]
					if diff < 30:
						score = score+1
			if score > maxscore:
				maxscore = score
				targetY = y
				targetX = x
		# print y
	finalscore=1.0*maxscore/sizeX/sizeY*samplerate*samplerate
	print targetY,targetX,finalscore
	if finalscore>0.5:
		font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
		cv2.putText(imgOri,filename, (targetX,targetY+90),font,3, (0, 205, 206),5)
		print "              "+filename+" finished searching"
		reslist.append([targetX,targetY,filename])
reslist.sort(key=lambda x:(x[0]+1)+3*(x[1]+1),reverse=True)
print reslist

cv2.imshow("0",imgOri)
for k in reslist:
	imgK=imgOri[k[1]:,k[0]:]
	maxwidth=imgK.shape[1]-5
	maxheight=imgK.shape[0]-5
	for i in range(maxheight-1,0,-1):
		t=0
		for j in range(maxwidth):
			if imgK[i,j].sum()<3*250:
				t=t+1
		if t>10:
			break
		maxheight=maxheight-1
	for i in range(maxwidth-1,0,-1):
		t=0
		for j in range(maxheight):
			if imgK[j,i].sum()<3*250:
				t=t+1
		if t>10:
			break
		maxwidth=maxwidth-1
	cv2.imshow(k[2],imgK[:maxheight+5,:maxwidth+5])
	imgOri[k[1]:,k[0]:]=255*np.ones((imgK.shape[0],imgK.shape[1],3))

# cv2.imshow("1",binaryA)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('new.jpg',imgOri)
    cv2.destroyAllWindows()
