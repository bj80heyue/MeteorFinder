import cv2
from tqdm import tqdm
import numpy as np
import os
from os import listdir
from os.path import join
import numba
from numba import jit
import shutil
import sys


def imread(path,rescale=0.5):
	#cvt RGB2GRAY & downsample
	img = cv2.imread(path,0)
	img = cv2.resize(img,(0,0),fx=rescale,fy=rescale,interpolation=cv2.INTER_NEAREST)
	return img

#kernel fo Sobel
kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
ky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
def grad(image):
	#Gradient
	grad_x = np.abs(cv2.filter2D(image,cv2.CV_8U,kx))
	grad_y = np.abs(cv2.filter2D(image,cv2.CV_8U,ky))
	gradxy = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
	return gradxy

def intersect(box1,box2):
	x01, y01, x02, y02 = box1
	x11, y11, x12, y12 = box2
 
	lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
	ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
	sax = abs(x01 - x02)
	sbx = abs(x11 - x12)
	say = abs(y01 - y02)
	sby = abs(y11 - y12)
	if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
		return True
	return False
	
def filtRects(lines):
	#Naive code Like NMS
	#lines[N,4]
	for rect_A in lines:
		for rect_B in lines:
			if rect_A is rect_B:
				continue
			elif intersect(rect_A,rect_B):
				lines.remove(rect_B)
	return lines	

@jit(nopython=True)
def findThresh(gray,recall_pix_num = 1000):
	#Find Appropriate Threshold
	hist_img, _ = np.histogram(gray, 256,range=[0,255])
	cdf_img = np.cumsum(hist_img)   
	num_total = cdf_img[-1]
	num_base = cdf_img[0]
	res = 10
	for i in range(245):
		index = 255-i
		if (num_total - cdf_img[index]) >recall_pix_num:
			res = index
			break
	return res

def dilate(img,d=7):
	kernel = np.ones((d, d), np.uint8)
	res = cv2.dilate(img, kernel)
	return res

def main(nameList,sensitivity = 30,rescale=0.5):
	#nameList: List of images stored in order of shooting
	if os.path.exists('MeterosOutput'):
		shutil.rmtree('MeterosOutput')
	os.makedirs('MeterosOutput')
	prev,now = None,None
	prev_grad,now_grad = None,None
	for name in tqdm(nameList):
		if 'JPG' not in name and 'jpg' not in name:
			continue
		if prev is None:
			prev = imread(join(head,name),rescale)
			prev_grad = grad(prev)
			continue
		
		prev_mask = dilate(prev_grad, 7)
		now = imread(join(head,name),rescale)
		now_grad = grad(now)
		diff = cv2.subtract(now_grad,prev_mask)	
		thresh = findThresh(diff)
		mask = cv2.threshold(diff,thresh,255,0)[1]
		lines = cv2.HoughLinesP(mask, 1, np.pi/180,sensitivity,minLineLength=30,maxLineGap=50)
		prev = now.copy()
		prev_grad = now_grad.copy()
		#visualize
		canvas = now.copy()
		if lines is not None:
			lines = lines.squeeze(1).tolist()
			lines = filtRects(lines)
			for line in lines:
				x1,y1,x2,y2 = line
				cv2.rectangle(canvas,(x1,y1),(x2,y2),(255,0,0),3)
			print('Find Meteor In',name,'Num:',len(lines))
			cv2.imwrite("MeterosOutput/%s"%(name),canvas)


if __name__  == '__main__':
	#Config
	#head = '/Users/heyue/Downloads/2018双子流星雨数据集'
	sensitivity = 30	
	rescale = 0.5
	if len(sys.argv)<=1:
		print("Please input the path of folder")
		sys.exit() 
	head = sys.argv[1]
	if len(sys.argv)>2:
		sensitivity = int(sys.argv[2])
	if len(sys.argv)>3:
		rescale = float(sys.argv[3])
		
	nameList = listdir(head)
	nameList.sort()
	main(nameList,sensitivity,rescale)
