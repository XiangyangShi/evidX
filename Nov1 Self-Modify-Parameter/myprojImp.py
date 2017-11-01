import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

similarity_treshold=0.13
margin_treshold=250
labelCap = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
num = 5
ifwrite=0
writefolder='/Users/shixiangyang/Desktop/Oct31/'
ifcontour=0
ifallcontour=0
marginbox=3
# labelCap ="D"
labelCap = labelCap[0:num]
thres = 0
minarea,maxarea=100,1700
img = cv2.imread('/Users/shixiangyang/Desktop/3.jpg')
imgB=[]


def wait(img):
	k = cv2.waitKey(0)
	if k == 27:         # wait for ESC key to exit
		cv2.destroyAllWindows()
	elif k == ord('s'): # wait for 's' key to save and exit
		cv2.imwrite('new.jpg',image)
		cv2.destroyAllWindows()

def cutmargin(inputimg,margin_treshold1):
	minY = 0
	minX = 0
	maxY = inputimg.shape[0]
	maxX = inputimg.shape[1]
	# print minX,maxX,minY,maxY
	for i in range(maxY):
		if(np.mean(inputimg[i,:])<margin_treshold1):
			minY=i
			break
	for i in range(maxY)[::-1]:
		if(np.mean(inputimg[i,:])<margin_treshold1):
			maxY=i
			break
	for i in range(maxX):
		if(np.mean(inputimg[:,i])<margin_treshold1):
			minX=i
			break
	for i in range(maxX)[::-1]:
		if(np.mean(inputimg[:,i])<margin_treshold1):
			maxX=i
			break
	# print minX,maxX,minY,maxY
	return inputimg[minY:maxY,minX:maxX]

def similarity(subimage,imga,imgname):
	maxsizeY = imga.shape[0]
	maxsizeX = imga.shape[1]
	# sizeY = subimage.shape[0]
	# sizeX = subimage.shape[1]
	score = 0
	res=cv2.resize(subimage,(maxsizeX,maxsizeY))
	for y in range(maxsizeY):
		for x in range(maxsizeX):
			imga_avg=np.mean(imga[y,x])
			res_avg=np.mean(res[y,x])
			mins=imga_avg-res_avg
			# if (imga_avg-res_avg)<5:
			# 	print imga_avg-res_avg
			score = score + mins*mins/255.0/255
	score= score/maxsizeY/maxsizeX
	# print score
	# if score<similarity_treshold:
	# 	# cv2.imshow("imgA",imga)
	# 	# cv2.imshow("subimage",res)
	# 	print "The difference between groundtruth & ",imgname," score is: ",score
	# if (ifwrite):
	# 	cv2.imwrite(writefolder+str(score)+'.jpg',res)
	return score

def findCharacter(img):
	result = []
	simtable = [0.3]*num
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	# noise removal
	binary=thresh
		# kernel = np.ones((3,3),np.uint8)
		# binary = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
	# #sure background area
	# sure_bg = cv2.dilate(opening,kernel,iterations=1)
	# sure_fg = cv2.erode(opening,kernel,iterations=1)
	# #Marker labelling
	# ret, label_image = cv2.connectedComponents(opening)
	# Add one to all labels so that sure background is not 0, but 1
	binary, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  
	print len(contours)," contours in all"
	for contour in contours:
		x,y,w,h=cv2.boundingRect(contour)
		dx,dy,dw,dh=0,0,0,0
		if x>marginbox:
			dx=-marginbox
		if y>marginbox:
			dy=-marginbox
		if x+w<img.shape[1]-marginbox:
			dw=marginbox
		if y+h<img.shape[0]-marginbox:
			dh=marginbox
		# print x,y,w,h  wrong!
		area = cv2.contourArea(contour)
		if area>minarea and area<maxarea:
			# print contour
			ct=[contour]
			subimage=img[y+dy:y+h+dh,x+dx:x+w+dw]
			subimage=cutmargin(subimage,margin_treshold)
			for i in range(num):
				imgA = imgB[i]
				sim=similarity(subimage,imgA,labelCap[i])
				if(sim<simtable[i]):
					simtable[i]=sim
			print simtable
			for i in range(num):
				filename=labelCap[i]
				MaxSim=simtable[i]*1.2
				if MaxSim>0.15:
					MaxSim=0.15
				imgA = imgB[i]
				sim=similarity(subimage,imgA,filename)
				if(sim<MaxSim):
					# print sim
					# This is important to find out what we get
					if ifcontour or ifallcontour:
						cv2.drawContours(img,ct,-1,(0,255,0),3)
					if (ifwrite):
						cv2.imwrite(writefolder+str(score)+'.jpg',res)
					result.append([x,y,x+w,y+h,filename,""])
					print "The difference between groundtruth & ",filename," score is: ",sim
				else:
					if ifallcontour:
						cv2.drawContours(img,ct,-1,(255,0,255),3)
					# break
					# print score
					# if score<similarity_treshold:
					# 	# cv2.imshow("imgA",imga)
					# 	# cv2.imshow("subimage",res)
					# 	print "The difference between groundtruth & ",imgname," score is: ",score
					# if (ifwrite):
					# 	cv2.imwrite(writefolder+str(score)+'.jpg',res)
	cv2.imshow("result",img)
	# print x,y,w,h,len(contours)
	# result.sort(key=lambda x:(x[2]+1)+3*(x[3]+1),reverse=True)
	result.sort(key=lambda x:(x[4]),reverse=True)
	return result

# if __name__=='__main__':


for filename in labelCap:
	imga=cv2.imread('/Users/shixiangyang/Desktop/'+filename+'.jpg')
	imga=cutmargin(imga,margin_treshold)
	imgB.append(imga)

res = findCharacter(img)
print res
count = 0
#b and b prime will be detected
for index in range(len(res)-1):
	if res[index][4]==res[index+1][4]:
		count=count+1
		res[index][5]=str(count)
		res[index+1][5]=str(count+1)
	else:
		count=0
for k in res:
	imgK=img[k[1]:,k[0]:]
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

	cv2.imshow(k[4]+k[5],imgK[:maxheight+5,:maxwidth+5])
	img[k[1]:,k[0]:]=255*np.ones((imgK.shape[0],imgK.shape[1],3))

wait(img)





