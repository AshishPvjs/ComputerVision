import numpy as np
import cv2

def convolution(image, kernel):
	'''
	Convolution for RGB and Gray Images
	Limited to Square kernel of odd length
	'''
	kernel = np.flipud(np.fliplr(kernel))
	outputImage = np.zeros_like(image)
	l = int((kernel.shape[0]-1)/2)
	if len(image.shape) == 3:
		for k in range(3):
			timage = image[:,:,k]
			imagePadded = np.zeros((timage.shape[0]+2*l,timage.shape[1]+2*l))
			imagePadded[l:-l,l:-l] = timage
			for i in range(timage.shape[1]):
				for j in range(timage.shape[0]):
					(outputImage[:,:,k])[j,i]= max((kernel*imagePadded[j:j+2*l+1,i:i+2*l+1]).sum(),0)
	else:
		imagePadded = np.zeros((image.shape[0]+2*l,image.shape[1]+2*l))
		imagePadded[l:-l,l:-l] = image
		for i in range(image.shape[1]):
			for j in range(image.shape[0]):
				outputImage[j,i] = max((kernel*imagePadded[j:j+2*l+1,i:i+2*l+1]).sum(),0)
	return outputImage

def correlation(image, kernel):
	'''
	Correlation for RGB and Gray Images
	Limited to Square kernel of odd length
	'''
	outputImage = np.zeros_like(image)
	l = int((kernel.shape[0]-1)/2)
	if len(image.shape) == 3:
		for k in range(3):
			timage = image[:,:,k]
			imagePadded = np.zeros((timage.shape[0]+2*l,timage.shape[1]+2*l))
			imagePadded[l:-l,l:-l] = timage
			for i in range(timage.shape[1]):
				for j in range(timage.shape[0]):
					(outputImage[:,:,k])[j,i] = max((kernel*imagePadded[j:j+2*l+1,i:i+2*l+1]).sum(),0)
	else:
		imagePadded = np.zeros((image.shape[0]+2*l,image.shape[1]+2*l))
		imagePadded[l:-l,l:-l] = image
		for i in range(image.shape[1]):
			for j in range(image.shape[0]):
				outputImage[j,i] = max((kernel*imagePadded[j:j+2*l+1,i:i+2*l+1]).sum(),0)
	return outputImage

def median_filter(image, kernel_size):
	'''
	Median Filter for RGB and Gray Images
	Limited to Square Kernel of odd length
	'''
	outputImage = np.zeros_like(image)
	l = (kernel_size-1)//2
	kernel = np.ones((kernel_size,kernel_size))
	if len(image.shape) == 3:
		for k in range(3):
			timage = image[:,:,k]
			imagePadded = np.zeros((timage.shape[0]+2*l,timage.shape[1]+2*l))
			imagePadded[l:-l,l:-l] = timage
			for i in range(timage.shape[1]):
				for j in range(timage.shape[0]):
					(outputImage[:,:,k])[j,i] = np.median(kernel*imagePadded[j:j+2*l+1,i:i+2*l+1])
	else:
		imagePadded = np.zeros((image.shape[0]+2*l,image.shape[1]+2*l))
		imagePadded[l:-l,l:-l] = image
		for i in range(image.shape[1]):
			for j in range(image.shape[0]):
				outputImage[j,i] = np.median(kernel*imagePadded[j:j+2*l+1,i:i+2*l+1])
	return outputImage


if __name__ == '__main__':

	# Reading the lena image
	im1 = cv2.imread("lena.png")

	# Reading the art image as Gray Scale
	im2 = cv2.imread("art.png",cv2.IMREAD_GRAYSCALE)
			
	#Below are mean kernels with different sizes
	meanKernel3 = (1/9)*np.ones((3,3))
	meanKernel5 = (1/25)*np.ones((5,5))
	meanKernel7 = (1/49)*np.ones((7,7))
	meanKernel11 = (1/121)*np.ones((11,11))
	meanKernel17 = (1/289)*np.ones((17,17))

	#Different Sharpening Kernels
	sharpeningKernel = (1/9)*np.array([[-1,-1,-1],[-1,17,-1],[-1,-1,-1]])

	#Gausian Kernel with sigma 0.25 and size (3,3)
	gaussianKernel1 = np.array([[0.000518,0.021715,0.000518],[0.021715,0.91107,0.021715],[0.000518,0.021715,0.000518]])

	#Gaussian Kernel with sigma 0.5 and size (3,3)
	gaussianKernel2 = np.array([[0.024879,0.107973,0.024879],[0.107973,0.468592,0.107973],[0.024879,0.107973,0.024879]])

	#Gaussian Kernel with sigma 0.75 and size (3,3)
	gaussianKernel3 = np.array([[0.057934,0.124827,0.057934],[0.124827,0.268958,0.124827],[0.057934,0.124827,0.057934]])

	#Gaussian Kernel with sigma 1 and size (3,3)
	gaussianKernel4 = np.array([[0.077847,0.123317,0.077847],[0.123317,0.195346,0.123317],[0.077847,0.123317,0.077847]])

	#Gaussian Kernel with sigma 0.33 and size (5,5)
	gaussianKernel5 = np.array([[0,0,0.000003,0,0],[0,0.004207,0.056449,0.004207,0],[0.000003,0.056449,0.757363,0.056449,0.000003],[0,0.004207,0.056449,0.004207,0],[0,0,0.000003,0,0]])

	#Gaussian Kernel with sigma 0.81 and size (7,7)
	gaussianKernel6 = np.array([[0.000001,0.000031,0.000238,0.000465,0.000238,0.000031,0.000001],
	[0.000031,0.000962,0.007334,0.014357,0.007334,0.000962,0.000031],
	[0.000238,0.007334,0.055934,0.109492,0.055934,0.007334,0.000238],
	[0.000465,0.014357,0.109492,0.214332,0.109492,0.014357,0.000465],
	[0.000238,0.007334,0.055934,0.109492,0.055934,0.007334,0.000238],
	[0.000031,0.000962,0.007334,0.014357,0.007334,0.000962,0.000031],
	[0.000001,0.000031,0.000238,0.000465,0.000238,0.000031,0.000001]])

	#Gaussian Kernel with sigma 0.96 and size (9,9)
	gaussianKernel7 = np.array([
	[0,0.000001,0.000007,0.000032,0.000052,0.000032,0.000007,0.000001,0],
	[0.000001,0.00002,0.000244,0.001083,0.001778,0.001083,0.000244,0.00002,0.000001],
	[0.000007,0.000244,0.002968,0.013193,0.021657,0.013193,0.002968,0.000244,0.000007],
	[0.000032,0.001083,0.013193,0.05864,0.096262,0.05864,0.013193,0.001083,0.000032],
	[0.000052,0.001778,0.021657,0.096262,0.158021,0.096262,0.021657,0.001778,0.000052],
	[0.000032,0.001083,0.013193,0.05864,0.096262,0.05864,0.013193,0.001083,0.000032],
	[0.000007,0.000244,0.002968,0.013193,0.021657,0.013193,0.002968,0.000244,0.000007],
	[0.000001,0.00002,0.000244,0.001083,0.001778,0.001083,0.000244,0.00002,0.000001],
	[0,0.000001,0.000007,0.000032,0.000052,0.000032,0.000007,0.000001,0]])
	

	#--------------------------------------------------------------------
	# #Please Uncomment the Block starting Here
	# #1 A
	# #Convolution of Image1 with different mean Kernels

	# output1 = convolution(im1,meanKernel3)
	# output2 = convolution(im1,meanKernel5)
	# output3 = convolution(im1,meanKernel7)
	# output4 = convolution(im1,meanKernel11)
	# output5 = convolution(im1, meanKernel17)

	# cv2.imshow('Original',im1)
	# cv2.imshow('mean3',output1)
	# cv2.imshow('mean5',output2)
	# cv2.imshow('mean7',output3)
	# cv2.imshow('mean11',output4)
	# cv2.imshow('mean17',output5)
	# #The question 1 A ends Here
	#--------------------------------------------------------------------
	# #Please Uncomment the Block starting Here
	# #1 B
	# #Convolution of Image1 with different gaussian Kernels


	# output1 = convolution(im1,gaussianKernel1)
	# output2 = convolution(im1,gaussianKernel2)
	# output3 = convolution(im1,gaussianKernel3)
	# output4 = convolution(im1,gaussianKernel4)
	# output5 = convolution(im1,gaussianKernel5)
	# output6 = convolution(im1,gaussianKernel6)
	# output7 = convolution(im1,gaussianKernel7)
	
	# cv2.imshow('gaussian1',output1)
	# cv2.imshow('gaussian2',output2)
	# cv2.imshow('gaussian3',output3)
	# cv2.imshow('gaussian4',output4)
	# cv2.imshow('gaussian5',output5)
	# cv2.imshow('gaussian6',output6)
	# cv2.imshow('gaussian7',output7)
	# #The question 1 B ends Here
	#--------------------------------------------------------------------
	# #Please Uncomment the Block starting Here
	# #1 C
	# #Sharpened Image1 
	# #Sharpening Kernel used : (1/9)*np.array([[-1,-1,-1],[-1,17,-1],[-1,-1,-1]])

	# output1 = convolution(im1,sharpeningKernel)
	# #Here we are clipping the output image to avoid the image bleeding green at edges
	# output1 = np.clip(output1, 0, 255).astype(np.uint8)
	# cv2.imshow('Original',im1)
	# cv2.imshow('Sharpened',output1)
	# #The question 1 C ends Here
	#--------------------------------------------------------------------
	# #Please Uncomment the Block starting Here
	# #2 A
	# #Convolution of Image2 with mean filters

	# output1 = convolution(im2,meanKernel3)
	# output2 = convolution(im2,meanKernel5)
	# output3 = convolution(im2,meanKernel7)
	# output4 = convolution(im2,meanKernel11)
	# output5 = convolution(im2, meanKernel17)

	# cv2.imshow('Original',im2)
	# cv2.imshow('mean3',output1)
	# cv2.imshow('mean5',output2)
	# cv2.imshow('mean7',output3)
	# cv2.imshow('mean11',output4)
	# cv2.imshow('mean17',output5)
	# #The question 2 A ends Here
	#--------------------------------------------------------------------
	# #Please Uncomment the Block starting Here
	# #2 B
	# #Correlation of Image2 with mean filters

	# output1 = correlation(im2,meanKernel3)
	# output2 = correlation(im2,meanKernel5)
	# output3 = correlation(im2,meanKernel7)
	# output4 = correlation(im2,meanKernel11)
	# output5 = correlation(im2, meanKernel17)

	# cv2.imshow('Original',im2)
	# cv2.imshow('mean3',output1)
	# cv2.imshow('mean5',output2)
	# cv2.imshow('mean7',output3)
	# cv2.imshow('mean11',output4)
	# cv2.imshow('mean17',output5)
	# #The question 2 B ends Here
	#---------------------------------------------------------------------
	# #Please Uncomment the Block starting Here
	# #2 B
	# #Median Filter on Image2
	# output1 = median_filter(im2,5)
	# output2 = median_filter(im2,7)
	# output3 = median_filter(im2,11)
	# output4 = median_filter(im2,19)
	# cv2.imshow('Original',im2)
	# cv2.imshow('five',output1)
	# cv2.imshow('seven',output2)
	# cv2.imshow('eleven', output3)
	# cv2.imshow('nineteen', output4)
	# #The question 2 C ends here
	#---------------------------------------------------------------------
    # Don't Comment this part of the Code. All the Images can be closed at once by presing the key zero	
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# -----------------------------------------------



