# -*- coding: utf-8 -*-
import os
import sys
import cv2
import caffe
import argparse
import scipy.io
import numpy as np
import time, datetime
from os.path import join

def bbreg(boundingbox, reg):
    reg = reg.T 
    
    # calibrate bouding boxes
    if reg.shape[1] == 1:
        print("reshape of reg")
        pass # reshape of reg
    w = boundingbox[:,2] - boundingbox[:,0] + 1
    h = boundingbox[:,3] - boundingbox[:,1] + 1

    bb0 = boundingbox[:,0] + reg[:,0]*w
    bb1 = boundingbox[:,1] + reg[:,1]*h
    bb2 = boundingbox[:,2] + reg[:,2]*w
    bb3 = boundingbox[:,3] + reg[:,3]*h
    
    boundingbox[:,0:4] = np.array([bb0, bb1, bb2, bb3]).T
    #print "bb", boundingbox
    return boundingbox

def pad(boxesA, w, h):
    boxes = boxesA.copy() # shit, value parameter!!!
    #print '#################'
    #print 'boxes', boxes
    #print 'w,h', w, h
    
    tmph = boxes[:,3] - boxes[:,1] + 1
    tmpw = boxes[:,2] - boxes[:,0] + 1
    numbox = boxes.shape[0]

    #print 'tmph', tmph
    #print 'tmpw', tmpw

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw 
    edy = tmph

    x = boxes[:,0:1][:,0]
    y = boxes[:,1:2][:,0]
    ex = boxes[:,2:3][:,0]
    ey = boxes[:,3:4][:,0]
   
   
    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]
        ex[tmp] = w-1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
        ey[tmp] = h-1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])
    
    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy-1)
    dx = np.maximum(0, dx-1)
    y = np.maximum(0, y-1)
    x = np.maximum(0, x-1)
    edy = np.maximum(0, edy-1)
    edx = np.maximum(0, edx-1)
    ey = np.maximum(0, ey-1)
    ex = np.maximum(0, ex-1)
    
    #print "dy"  ,dy 
    #print "dx"  ,dx 
    #print "y "  ,y 
    #print "x "  ,x 
    #print "edy" ,edy
    #print "edx" ,edx
    #print "ey"  ,ey 
    #print "ex"  ,ex 


    #print 'boxes', boxes
    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]

def rerec(bboxA):
    # convert bboxA to square
    w = bboxA[:,2] - bboxA[:,0]
    h = bboxA[:,3] - bboxA[:,1]
    l = np.maximum(w,h).T
    
    #print 'bboxA', bboxA
    #print 'w', w
    #print 'h', h
    #print 'l', l
    bboxA[:,0] = bboxA[:,0] + w*0.5 - l*0.5
    bboxA[:,1] = bboxA[:,1] + h*0.5 - l*0.5 
    bboxA[:,2:4] = bboxA[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return bboxA

def nms(boxes, threshold, type):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: TODO
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort()) # read s using I
    
    pick = [];
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where( o <= threshold)[0]]
    return pick

def generateBoundingBox(map, reg, scale, t):
    stride = 2
    cellsize = 12
    map = map.T
    dx1 = reg[0,:,:].T
    dy1 = reg[1,:,:].T
    dx2 = reg[2,:,:].T
    dy2 = reg[3,:,:].T
    (x, y) = np.where(map >= t)

    yy = y
    xx = x
    
    '''
    if y.shape[0] == 1: # only one point exceed threshold
        y = y.T
        x = x.T
        score = map[x,y].T
        dx1 = dx1.T
        dy1 = dy1.T
        dx2 = dx2.T
        dy2 = dy2.T
        # a little stange, when there is only one bb created by PNet
        
        #print "1: x,y", x,y
        a = (x*map.shape[1]) + (y+1)
        x = a/map.shape[0]
        y = a%map.shape[0] - 1
        #print "2: x,y", x,y
    else:
        score = map[x,y]
    '''
    #print "dx1.shape", dx1.shape
    #print 'map.shape', map.shape
   

    score = map[x,y]
    reg = np.array([dx1[x,y], dy1[x,y], dx2[x,y], dy2[x,y]])

    if reg.shape[0] == 0:
        pass
    boundingbox = np.array([yy, xx]).T

    bb1 = np.fix((stride * (boundingbox) + 1) / scale).T # matlab index from 1, so with "boundingbox-1"
    bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T # while python don't have to
    score = np.array([score])

    boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)

    #print '(x,y)',x,y
    #print 'score', score
    #print 'reg', reg

    return boundingbox_out.T

def drawBoxes(im, boxes):
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	for i in range(x1.shape[0]):
		cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0,255,0), 1)
	return im
	
def drawPoints(im, coord):
	if (coord == []):
		return im
		
	for i in range(coord.shape[0]):
		points = coord[i]
		pointCount = int(points.shape[0] / 2)
		for j in range(pointCount):
			cv2.circle(im, (int(points[j]), int(points[j + pointCount])), 1, (0,255,0), 1)
	return im

def getAllFilesInDirectory(directory, postfix = ""):
	fileNames = []
	for root, dirs, files in os.walk(directory):
		for filename in files:
			if (filename.lower().endswith(postfix)): 
				fileNames.append(filename)
					
	return fileNames
	
def getFilesPath(directory, postfix = ""):
	filePaths = [s for s in os.listdir(directory)];
	if not postfix or postfix == "":
		return filePaths;
	else:
		return [s for s in filePaths if s.lower().endswith(postfix)];
	
def detect_face(img, minsize, detectCNNs, threshold, fastresize, factor):
    img2 = img.copy()
	
    PNet = detectCNNs[0]
    RNet = detectCNNs[1]
    ONet = detectCNNs[2]

    factor_count = 0
    total_boxes = np.zeros((0,9), np.float)
    points = []
    h = img.shape[0]
    w = img.shape[1]
    minl = min(h, w)
    img = img.astype(float)
    m = 12.0/minsize
    minl = minl*m
    
    #total_boxes = np.load('total_boxes.npy')
    #total_boxes = np.load('total_boxes_242.npy')
    #total_boxes = np.load('total_boxes_101.npy')
   
    # create scale pyramid
    scales = []
    while minl >= 12:
        scales.append(m * pow(factor, factor_count))
        minl *= factor
        factor_count += 1

    # first stage
    for scale in scales:
        hs = int(np.ceil(h*scale))
        ws = int(np.ceil(w*scale))

        if fastresize:
            im_data = (img-127.5)*0.0078125 # [0,255] -> [-1,1]
            im_data = cv2.resize(im_data, (ws,hs)) # default is bilinear
        else: 
            im_data = cv2.resize(img, (ws,hs)) # default is bilinear
            im_data = (im_data-127.5)*0.0078125 # [0,255] -> [-1,1]
        #im_data = imResample(img, hs, ws); print "scale:", scale


        im_data = np.swapaxes(im_data, 0, 2)
        im_data = np.array([im_data], dtype = np.float)
        PNet.blobs['data'].reshape(1, 3, ws, hs)
        PNet.blobs['data'].data[...] = im_data
        out = PNet.forward()
    
        boxes = generateBoundingBox(out['prob1'][0,1,:,:], out['conv4-2'][0], scale, threshold[0])
        if boxes.shape[0] != 0:
            #print boxes[4:9]
            #print 'im_data', im_data[0:5, 0:5, 0], '\n'
            #print 'prob1', out['prob1'][0,0,0:3,0:3]

            pick = nms(boxes, 0.5, 'Union')

            if len(pick) > 0 :
                boxes = boxes[pick, :]

        if boxes.shape[0] != 0:
            total_boxes = np.concatenate((total_boxes, boxes), axis=0)
         
    #np.save('total_boxes_101.npy', total_boxes)

    #####
    # 1 #
    #####
    #print("[1]:",total_boxes.shape[0])
    #print total_boxes
    #return total_boxes, [] 


    numbox = total_boxes.shape[0]
    if numbox > 0:
        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        #print("[2]:",total_boxes.shape[0])
        
        # revise and convert to square
        regh = total_boxes[:,3] - total_boxes[:,1]
        regw = total_boxes[:,2] - total_boxes[:,0]
        t1 = total_boxes[:,0] + total_boxes[:,5]*regw
        t2 = total_boxes[:,1] + total_boxes[:,6]*regh
        t3 = total_boxes[:,2] + total_boxes[:,7]*regw
        t4 = total_boxes[:,3] + total_boxes[:,8]*regh
        t5 = total_boxes[:,4]
        total_boxes = np.array([t1,t2,t3,t4,t5]).T
        #print "[3]:",total_boxes.shape[0]
        #print regh
        #print regw
        #print 't1',t1
        #print total_boxes

        total_boxes = rerec(total_boxes) # convert box to square
        #print("[4]:",total_boxes.shape[0])
        
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])
        #print("[4.5]:",total_boxes.shape[0])
        #print total_boxes
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

    #print total_boxes.shape
    #print total_boxes

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage

        #print 'tmph', tmph
        #print 'tmpw', tmpw
        #print "y,ey,x,ex", y, ey, x, ex, 
        #print "edy", edy

        #tempimg = np.load('tempimg.npy')

        # construct input for RNet
        tempimg = np.zeros((numbox, 24, 24, 3)) # (24, 24, 3, numbox)
        for k in range(numbox):
            tmp = np.zeros((int(tmph[k]) +1, int(tmpw[k]) + 1,3))
          
            #print "dx[k], edx[k]:", dx[k], edx[k]
            #print "dy[k], edy[k]:", dy[k], edy[k]
            #print "img.shape", img[y[k]:ey[k]+1, x[k]:ex[k]+1].shape
            #print "tmp.shape", tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1].shape

            tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
            #print "y,ey,x,ex", y[k], ey[k], x[k], ex[k]
            #print "tmp", tmp.shape
            
            tempimg[k,:,:,:] = cv2.resize(tmp, (24, 24))
            #tempimg[k,:,:,:] = imResample(tmp, 24, 24)
            #print 'tempimg', tempimg[k,:,:,:].shape
            #print tempimg[k,0:5,0:5,0] 
            #print tempimg[k,0:5,0:5,1] 
            #print tempimg[k,0:5,0:5,2] 
            #print k
    
        #print tempimg.shape
        #print tempimg[0,0,0,:]
        tempimg = (tempimg-127.5)*0.0078125 # done in imResample function wrapped by python

        #np.save('tempimg.npy', tempimg)

        # RNet

        tempimg = np.swapaxes(tempimg, 1, 3)
        #print tempimg[0,:,0,0]
        
        RNet.blobs['data'].reshape(numbox, 3, 24, 24)
        RNet.blobs['data'].data[...] = tempimg
        out = RNet.forward()

        #print out['conv5-2'].shape
        #print out['prob1'].shape

        score = out['prob1'][:,1]
        #print 'score', score
        pass_t = np.where(score>threshold[1])[0]
        #print 'pass_t', pass_t
        
        score =  np.array([score[pass_t]]).T
        total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)
        #print("[5]:",total_boxes.shape[0])
        #print total_boxes

        #print "1.5:",total_boxes.shape
        
        mv = out['conv5-2'][pass_t, :].T
        #print "mv", mv
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            #print 'pick', pick
            if len(pick) > 0 :
                total_boxes = total_boxes[pick, :]
                #print( "[6]:",total_boxes.shape[0])
                total_boxes = bbreg(total_boxes, mv[:, pick])
                #print("[7]:",total_boxes.shape[0])
                total_boxes = rerec(total_boxes)
                #print("[8]:",total_boxes.shape[0])
            
        #####
        # 2 #
        #####
        #print("2:",total_boxes.shape)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage
            
            total_boxes = np.fix(total_boxes)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)
           
            #print 'tmpw', tmpw
            #print 'tmph', tmph
            #print 'y ', y
            #print 'ey', ey
            #print 'x ', x
            #print 'ex', ex
        

            tempimg = np.zeros((numbox, 48, 48, 3))
            for k in range(numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]),3))
                tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                tempimg[k,:,:,:] = cv2.resize(tmp, (48, 48))
            tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]
                
            # ONet
            tempimg = np.swapaxes(tempimg, 1, 3)
            ONet.blobs['data'].reshape(numbox, 3, 48, 48)
            ONet.blobs['data'].data[...] = tempimg
            out = ONet.forward()
            
            score = out['prob1'][:,1]
            points = out['conv6-3']
            pass_t = np.where(score>threshold[2])[0]
            points = points[pass_t, :]
            score = np.array([score[pass_t]]).T
            total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis=1)
            #print("[9]:",total_boxes.shape[0])
            
            mv = out['conv6-2'][pass_t, :].T
            w = total_boxes[:,3] - total_boxes[:,1] + 1
            h = total_boxes[:,2] - total_boxes[:,0] + 1

            points[:, 0:5] = np.tile(w, (5,1)).T * points[:, 0:5] + np.tile(total_boxes[:,0], (5,1)).T - 1 
            points[:, 5:10] = np.tile(h, (5,1)).T * points[:, 5:10] + np.tile(total_boxes[:,1], (5,1)).T -1

            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, mv[:,:])
                #print("[10]:",total_boxes.shape[0])
                pick = nms(total_boxes, 0.7, 'Min')
                
                #print pick
                if len(pick) > 0 :
                    total_boxes = total_boxes[pick, :]
                    #print("[11]:",total_boxes.shape[0])
                    points = points[pick, :]

    #####
    # 3 #
    #####
    #print("3:",total_boxes.shape)

    return total_boxes, points

def haveFace(img, facedetector):
    minsize = facedetector[0]
    PNet = facedetector[1]
    RNet = facedetector[2]
    ONet = facedetector[3]
    threshold = facedetector[4]
    factor = facedetector[5]
    
    if max(img.shape[0], img.shape[1]) < minsize:
        return False, []

    img_matlab = img.copy()
    tmp = img_matlab[:,:,2].copy()
    img_matlab[:,:,2] = img_matlab[:,:,0]
    img_matlab[:,:,0] = tmp
    
    #tic()
    boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
    #toc()
    containFace = (True, False)[boundingboxes.shape[0]==0]
    return containFace, boundingboxes

def detect_on_images():
	inpRoot = 'INP/';
	outRoot = 'OUT/';

	caffe_model_path = "./model"
	
	PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
	RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
	ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)
	
	minsize = 20
	threshold = [0.6, 0.7, 0.9]
	factor = 0.709
	
	dataPaths = getFilesPath(inpRoot);
	averTime = 0;
	for i in range(0, len(dataPaths)):
		imgName = dataPaths[i];
		imgpath = join(inpRoot, imgName)
		
		tstart = datetime.datetime.now();
	
		img = cv2.imread(imgpath)
		img_matlab = img.copy()
		tmp = img_matlab[:,:,2].copy()
		img_matlab[:,:,2] = img_matlab[:,:,0]
		img_matlab[:,:,0] = tmp
		boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
		
		averTime += (datetime.datetime.now() - tstart).total_seconds();
	
		img = drawBoxes(img, boundingboxes)
		img = drawPoints(img, points)
		cv2.imwrite(outRoot + imgName.split('.')[0] + '_1.png', img);

def imresize(img, scale, interpolation = cv2.INTER_LINEAR):
	return cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=interpolation)

def imWidthHeight(input):
	if type(input) is str: #or type(input) is unicode:
		width, height = Image.open(input).size # this does not load the full image
	else:
		width =  input.shape[1]
		height = input.shape[0]
	return width,height
	
def imresizeAndPasteCenter(img, width, height, size, pad_value=0):
    # resize image
    imgWidth, imgHeight = imWidthHeight(img)
    scale = min(float(width) / float(imgWidth), float(height) / float(imgHeight))
    imgResized = imresize(img, scale, interpolation=cv2.INTER_CUBIC)
    resizedWidth, resizedHeight = imWidthHeight(imgResized)

    imOut = pad_value * np.ones((size,size, 3), dtype = img.dtype)

	# pad image
    top  = int(max(0, np.round((size - resizedHeight) / 2)))
    left = int(max(0, np.round((size - resizedWidth) / 2)))
    bottom = size - (size - resizedHeight - top)
    right  = size - (size - resizedWidth - left)
    imOut[top:bottom, left:right, 0] = imgResized[:,:, 0]
    imOut[top:bottom, left:right, 1] = imgResized[:,:, 1]
    imOut[top:bottom, left:right, 2] = imgResized[:,:, 2]
	
    return imOut

def imresizeAndPad(img, width, height, pad_value=0):
    # resize image
    imgWidth, imgHeight = imWidthHeight(img)
    scale = min(float(width) / float(imgWidth), float(height) / float(imgHeight))
    imgResized = imresize(img, scale, interpolation=cv2.INTER_CUBIC)
    resizedWidth, resizedHeight = imWidthHeight(imgResized)

    # pad image
    top  = int(max(0, np.round((height - resizedHeight) / 2)))
    left = int(max(0, np.round((width - resizedWidth) / 2)))
    bottom = height - top - resizedHeight
    right  = width - left - resizedWidth
    return cv2.copyMakeBorder(imgResized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value]), scale, top, left

def getFaceAttribute(faceImage, attributeModel):
	image = faceImage.copy().astype("float32")		
	if (len(image.shape) == 4):
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
	
	if (len(image.shape) == 2):
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
		
	#scale image to fix size 
	# image = imresizeAndPad(image, 178, 218, 0)[0]	#???????????????????????
	image = cv2.resize(image, (178, 218), interpolation=cv2.INTER_CUBIC)
	# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
	data = np.expand_dims(image.transpose(2, 0, 1), 0)
	
	attributeModel.blobs['data'].data[...] = data / 255
	output = attributeModel.forward()
	out_signal = output['moon-fc'][0]
	out_signal /= np.max(np.abs(out_signal))
	# print(out_signal)
	allAttributes = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
        'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
        'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
        'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young',
    ]
	mask_for_face = [1,1,1,1,0,0,1,1,0,0,0,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,0,1,0,1,0,0,0,0,1,0,0,0] 
	
	outAttributes = ''
	for i in range(len(allAttributes)): 
		# if (mask_for_face[i] == 1):
			if (out_signal[i] > 0):
				outAttributes += allAttributes[i] + ' '
	# outAttributes += ' No('
	# for i in range(len(allAttributes)): 
		# if (mask_for_face[i] == 1):
			# if (out_signal[i] < -0.5):
				# outAttributes += allAttributes[i] + ' '
	# outAttributes += ')'
	
	# print(outAttributes)
	# input()
		
	return outAttributes
							  
def getFaceExpression(faceImage, expressModel):
	image = faceImage.copy().astype("float32")		
	if (len(image.shape) == 4):
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
	
	if (len(image.shape) == 2):
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

	#scale image to fix size 
	#image = imresizeAndPasteCenter(image, 224, 224, 224, 0)	#???????????????????????
	image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
	# image = (image - 127.5) / 128
	# image = np.transpose(image, (1, 0, 2))
		
	# for c_i in range(3):
		# image[:, :, c_i] -= np.min(image[:, :, c_i])
		# image[:, :, c_i] /= np.max(image[:, :, c_i])
		# image[:, :, c_i] -= 0.5
		
	image -= np.min(image)
	image /= np.max(image)
	image =  image * 255 - 127.5
	
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
	data = np.expand_dims(image.transpose(2, 0, 1), 0)
	
	# cv2.imwrite('rez.png', image);
	# input()
	
	express_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']	
	expressModel.blobs['data'].data[...] = data
	expressModel.forward()
	express = expressModel.blobs['us_crop_ck'].data[0]
	express = express_list[express.argmax()]
	
	return express
							  
def alignFace(image, poins):
	imgSize = (112, 96)

	#optimal points for recognition
	x_ = np.array([30.2946, 65.5318, 48.0252, 33.5493, 62.7299]).astype("float32")
	y_ = np.array([51.6963, 51.5014, 71.7366, 92.3655, 92.2041]).astype("float32")
	
	src = np.dstack((x_,y_))
	poins.shape = (np.size(poins))
	pset_x = poins[0:5]
	pset_y = poins[5:10]
	dst = np.dstack((pset_x, pset_y))
	
	transmat = cv2.estimateRigidTransform( dst, src, True)	#???????????????????
	alignedFace = cv2.warpAffine(image, transmat, (imgSize[1], imgSize[0]))

	return alignedFace
		
def getFaceGender(faceImage, genderModel, mean):
	image = faceImage.copy().astype("float32")		
	if (len(image.shape) == 4):
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
	
	if (len(image.shape) == 2):
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

	#scale image to fix size 
	image = imresizeAndPasteCenter(image, 224, 224, 224, 0)	#???????????????????????
	
	for c_i in range(3):
		image[:, :, c_i] -= mean[c_i]
	data = np.expand_dims(image.transpose(2, 0, 1), 0)
	
	gender_list = ['Female', 'Male']	
	genderModel.blobs['data'].data[...] = data
	genderModel.forward()
	gender = genderModel.blobs['fc8-2'].data[0]
	gender = gender_list[gender.argmax()]
	
	return gender
	
def getFaceAGE(faceImage, ageModel, mean):	
	image = faceImage.copy().astype("float32")
	if (len(image.shape) == 4):
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
	
	if (len(image.shape) == 2):
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

	#scale image to fix size 
	image = imresizeAndPasteCenter(image, 224, 224, 224, 0)	#???????????????????????
		
	for c_i in range(3):
		image[:, :, c_i] -= mean[c_i]
	data = np.expand_dims(image.transpose(2, 0, 1), 0)
	
	ageModel.blobs['data'].data[...] = data
	ageModel.forward()
	age = ageModel.blobs['fc8-101'].data[0]	
	age = age.argmax()

	return age
	
def getGaze(sceneImage, faceImage, eyesCoord, gazeModel, gazeMeans):	
	face = faceImage.copy()
	image = sceneImage.copy()
	if (len(image.shape) == 4):
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
	if (len(face.shape) == 4):
		face = cv2.cvtColor(face, cv2.COLOR_RGBA2RGB)
	if (len(image.shape) == 2):
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	if (len(face.shape) == 2):
		face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
	
	#???????????????????????
	alpha = max(faceImage.shape[0] / sceneImage.shape[0], faceImage.shape[1] / sceneImage.shape[1])
	
	if (image.shape[0] != 227 or image.shape[1] != 227):
		# [image, scale, top, left] = imresizeAndPad(image, 227, 227, 0)
		image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_CUBIC)

	image = image - gazeMeans[:,:,:,0]	
	image = np.transpose(image, (1, 0, 2))
	image = np.expand_dims(image.transpose(2,0,1), 0)
	
	# new_x = eyesCoord[0] * sceneImage.shape[1] * scale + left
	# new_y = eyesCoord[1] * sceneImage.shape[0] * scale + top
	# eyesCoord[0] = new_x / 227
	# eyesCoord[1] = new_y / 227
		
	if (face.shape[0] != 227 or face.shape[1] != 227):
		#face = imresizeAndPad(face, 227, 227, 0)[0]		
		face = cv2.resize(face, (227, 227), interpolation=cv2.INTER_CUBIC)
	face = face - gazeMeans[:,:,:,1]	
	face = np.transpose(face, (1, 0, 2))
	face = np.expand_dims(face.transpose(2, 0, 1), 0)
	
	f = np.zeros((1,169,1,1), np.float32)
	z = np.zeros((13,13), np.float32)
	x = min(int(np.floor(eyesCoord[0] * 13)), 12)
	y = min(int(np.floor(eyesCoord[1] * 13)), 12)
	z[x, y] = 1
	f[0,:,0,0] = z.T.flatten()
	
	gazeModel.blobs['data'].data[...] = image
	gazeModel.blobs['face'].data[...] = face
	gazeModel.blobs['eyes_grid'].data[...] = f
	
	output = gazeModel.forward()
	fc_0_0 = output['fc_0_0']
	fc_1_0 = output['fc_1_0']
	fc_m1_0 = output['fc_m1_0']
	fc_0_1 = output['fc_0_1']
	fc_0_m1 = output['fc_0_m1']
		
	# print(np.mean(fc_0_0))
	# print(np.mean(fc_1_0))
	# print(np.mean(fc_m1_0))
	# print(np.mean(fc_0_1))
	# print(np.mean(fc_0_m1))
	
	hm = np.zeros((15,15), np.float32)
	count_hm = np.zeros((15,15), np.float32)
	f_0_0 = fc_0_0.reshape([5,5], order='F')
	f_0_0 = np.exp(alpha * f_0_0) / np.sum(np.exp(alpha * f_0_0[:]))
	f_1_0 = fc_1_0.reshape([5,5], order='F')
	f_1_0 = np.exp(alpha * f_1_0) / np.sum(np.exp(alpha * f_1_0[:]))
	f_m1_0 = fc_m1_0.reshape([5,5], order='F')
	f_m1_0 = np.exp(alpha * f_m1_0) / np.sum(np.exp(alpha * f_m1_0[:]))
	f_0_1 = fc_0_1.reshape([5,5], order='F')
	f_0_1 = np.exp(alpha * f_0_1) / np.sum(np.exp(alpha * f_0_1[:]))
	f_0_m1 = fc_0_m1.reshape([5,5], order='F')
	f_0_m1 = np.exp(alpha * f_0_m1) / np.sum(np.exp(alpha * f_0_m1[:]))

	f_cell = (f_0_0,f_1_0,f_m1_0,f_0_m1,f_0_1)
	v_x = [0, 1, -1, 0, 0]
	v_y = [0, 0, 0, -1, 1]
	
	for k in range(0, 5):
		delta_x = v_x[k]
		delta_y = v_y[k]
		f = f_cell[k]
		for x in range(1, 6):
			for y in range(1, 6):
				i_x = 1 + 3*(x-1) - delta_x
				i_x = max(i_x, 1)
				if(x == 1):
					i_x = 1;
				
				i_y = 1 + 3*(y-1) - delta_y
				i_y = max(i_y, 1)
				if(y == 1):
					i_y = 1
				
				f_x = 3*x - delta_x 
				f_x = min(15, f_x)
				if(x == 5):
					f_x = 15

				f_y = 3*y - delta_y
				f_y = min(15, f_y)
				if(y == 5):
					f_y = 15
				
				i_x -= 1
				i_y -= 1				
				hm[i_x:f_x,i_y:f_y] = hm[i_x:f_x,i_y:f_y] + f[x-1,y-1]
				count_hm[i_x:f_x,i_y:f_y] = count_hm[i_x:f_x,i_y:f_y] + 1
				
	hm_base = np.divide(hm, count_hm)
	hm_results = cv2.resize(hm_base.T, (227, 227), interpolation=cv2.INTER_CUBIC)

	idx = np.argmax(hm_results)
	y_predict = int(idx / 227)
	x_predict = int(idx % 227)
	
	# x_predict = int((x_predict - left) / scale)
	# y_predict = int((y_predict - top) / scale)
	
	x_predict = int(np.floor(sceneImage.shape[0] * x_predict / 227))
	y_predict = int(np.floor(sceneImage.shape[1] * y_predict / 227))
	
	return [y_predict, x_predict]
	
def contrastNorm(img):
	lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
	l, a, b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
	cl = clahe.apply(l)
	# cl = cv2.equalizeHist(image)
	limg = cv2.merge((cl,a,b))
	final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
	
	return final

def getFaceFetatures(faceImage, model):	
	image = faceImage.copy()
	if (len(image.shape) == 4):
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
	
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)			#!!!!!!!!!!!for gray base
	
	if (len(image.shape) == 2):
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
		
	# cv2.imwrite('rez.png', image) 
	# cv2.imwrite('rez1.png', contrastNorm(image)) 
	# input()
	
	#scale image to fix size 
	imHeight = image.shape[0]
	imWidth  = image.shape[1]	
	if (imHeight != 112 or imWidth != 96):
		# image = imresizeAndPad(image, 96, 112, 0)[0]
		image = cv2.resize(image, (96, 112), interpolation=cv2.INTER_CUBIC)

	# image = contrastNorm(image)
	
	# image = np.transpose(image, (1, 0, 2))
	# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #???????????????????s
	
	image = contrastNorm(image)
	image = (image - 127.5) / 128
	image = np.expand_dims(image.transpose(2, 0, 1), 0)

	#run CNN model
	model.blobs['data'].data[...] = image
	model.forward()
	features = model.blobs['fc5'].data[0]
	
	#features = np.sign(features) * (abs(features) ** 0.25)
	#features = L2Norm(features)

	return features
	
def baseMatching(baseVects, testVects):	
	baseSize = baseVects.shape[1]
	if (len(testVects.shape) == 1):
		testVects = np.reshape(testVects, (1, testVects.shape[0]))
		testSize = 1
	else:
		testSize = testVects.shape[0]

	baseLen = np.power(baseVects, 2)
	baseLen = np.sum(baseLen, axis=0)
	baseLen = np.power(baseLen, 0.5)
	baseLen = np.reshape(baseLen, (1, baseSize))
	testLen = np.power(testVects, 2)
	testLen = np.sum(testLen, axis = 1)
	testLen = np.power(testLen, 0.5)
	testLen = np.reshape(testLen, (testSize, 1))
	lenMult = testLen.dot(baseLen)
	vecMult = testVects.dot(baseVects)
	matchDist = 1 - vecMult / lenMult
	matchInd = np.argsort(matchDist, axis=1)

	return matchInd[0][0]
	
def loadBase(basePath, model, load_base):
	vectorLen = 512
	# maxBaseSize = 30
	baseFiles = getAllFilesInDirectory(basePath)
	baseFiles.sort()
	
	baseNames = []
	baseSize = len(baseFiles)
	# baseSize = min(len(baseFiles), maxBaseSize)
	baseFeatures = np.zeros((vectorLen, baseSize), np.float32)
	for i in range(0, baseSize):
		fileName = baseFiles[i]
		imgPath = os.path.join(basePath + '/' + fileName)
		img = cv2.imread(imgPath)
		imgVector = getFaceFetatures(img, model)
		baseFeatures[:, i] = imgVector
		baseNames.append(fileName)

	return baseFeatures, baseNames

def videoProcessing(videoPath, detectModel, checkModels, basePath):
	outRoot = 'OUT/';
	caffe_model_path = "./model"

	#detect param
	minsize = 20
	factor = 0.709
	threshold = [0.75, 0.85, 0.95]
	
	videoCapture = cv2.VideoCapture(videoPath)
	videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH,640)  
	videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)  

	fNumber = 0	
	faceCount = 0
	frameCount = 0
	
	recognTime = 0
	detetctTime = 0
	allProcTime = 0
	
	if (checkModels[0] == 1):			
		recModel = initRecognitionModel()		
		print("Base loading ...")			
		[baseFeatures, baseNames] = loadBase(basePath, recModel, load_base = False)
		print("Done.")
	if (checkModels[1] == 1):	
		[genderModel, genderMean] = initGenderModel()
	if (checkModels[2] == 1):				
		[ageModel, ageMean] = initAgeModel()
	if (checkModels[3] == 1):				
		[gazeModel, gazeMean] = initGazeModel()		
	if (checkModels[4] == 1):				
		expressModel = initExpressionModel()	
	if (checkModels[5] == 1):				
		attributeModel = initAttributeModel()
		
	printMenu()
		
	tAllStart = datetime.datetime.now()
	while True :
		success, frame = videoCapture.read()	
		if (success == False):
			break
			
		fNumber += 1
		if (fNumber % 5 != 0):
			continue
	
		tstart = datetime.datetime.now()
		img_matlab = frame.copy()
		tmp = img_matlab[:,:,2].copy()
		img_matlab[:,:,2] = img_matlab[:,:,0]
		img_matlab[:,:,0] = tmp
		boundingboxes, points = detect_face(img_matlab, minsize, detectModel, threshold, False, factor)
		detetctTime += (datetime.datetime.now() - tstart).total_seconds()
		frameCount += 1
		
		imgDraw = frame.copy()
		imgDraw = drawBoxes(imgDraw, boundingboxes)
		imgDraw = drawPoints(imgDraw, points)		
		
		if (len(points) != 0 and len(boundingboxes) != 0):
			for i in range(0, points.shape[0]):
				x1 = int(boundingboxes[i,0])
				y1 = int(boundingboxes[i,1])
				x2 = int(boundingboxes[i,2])
				y2 = int(boundingboxes[i,3])
				if (x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0):
					continue
				
				faceCount += 1				
				facePoints = points[i, :]				
				faceHeight = y2 - y1 + 1
				faceWidth = x2 - x1 + 1	
				
				if (sum(checkModels) > -6):				
					faceImage = alignFace(frame, facePoints)
					
				if (checkModels[0] == 1):
					tstart = datetime.datetime.now()
					faceFeatures = getFaceFetatures(faceImage, recModel)
					ind = baseMatching(baseFeatures, faceFeatures)
					name = baseNames[ind].split('.')[0] + ' '
					recognTime += (datetime.datetime.now() - tstart).total_seconds()
					outPoz = (max(x1, 15), max(y1, 15))
					cv2.putText(imgDraw, name, outPoz, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
					
				if (checkModels[1] == 1):
					gender = getFaceGender(faceImage, genderModel, genderMean)					
					outPoz = (min(x2 + 2, frame.shape[1] - 15), int((y1 + y2) / 2))
					cv2.putText(imgDraw, gender, outPoz, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
							
				if (checkModels[2] == 1):
					age = getFaceAGE(faceImage, ageModel, ageMean)
					outPoz = (max(x1-30, 15), int((y1 + y2) / 2))
					cv2.putText(imgDraw, str(age), outPoz, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
								
				if (checkModels[3] == 1):
					faceImage = frame[y1:y2, x1:x2]
					pointCount = int(facePoints.size / 2)
					eyesCoord = [int((facePoints[0] + facePoints[1]) / 2), 
							int((facePoints[0 + pointCount] + facePoints[1 + pointCount]) / 2)]
					prop_eyesCoord = [eyesCoord[0] / frame.shape[1], eyesCoord[1] / frame.shape[0]]				
					gazeCoord = getGaze(frame, faceImage, prop_eyesCoord, gazeModel, gazeMean)
					cv2.line(imgDraw, (eyesCoord[0], eyesCoord[1]),
						(gazeCoord[0], gazeCoord[1]), (0,255,0), 5)	
					
				if (checkModels[4] == 1):
					express = getFaceExpression(faceImage, expressModel)					
					outPoz = (min(x1, frame.shape[1] - 15), min(y2 + 15, frame.shape[0] - 15))
					cv2.putText(imgDraw, express, outPoz, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
					
				if (checkModels[5] == 1):					
					y1 = max(0, int(y1 - 0.25*faceHeight))
					x1 = max(0, int(x1 - 0.3*faceWidth))
					y2 = min(frame.shape[0] - 1, int(y2 + 0.25*faceHeight))
					x2 = min(frame.shape[1] - 1, int(x2 + 0.3*faceWidth))
					faceImage = frame[y1:y2, x1:x2]
					attributes = getFaceAttribute(faceImage, attributeModel)
					if (checkModels[4] != 1):
						outPoz = (min(x1, frame.shape[1] - 15), min(y2 + 15, frame.shape[0] - 15))
					else:
						outPoz = (min(x1, frame.shape[1] - 15), min(y2 + 20, frame.shape[0] - 15))
					cv2.putText(imgDraw, attributes, outPoz, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
					
								
	
		cv2.imshow("POC", imgDraw)				
		key = cv2.waitKey(1)
		if (key == ord('q')):
			break
		if (key == 32):
			key = -1
			while(key != 32):
				key = cv2.waitKey(1)
				
		if (key == ord('1')):
			if (checkModels[0] == -1):
				recModel = initRecognitionModel()		
				print("Base loading ...")			
				[baseFeatures, baseNames] = loadBase(basePath, recModel, load_base = False)
				print("Done.")
				checkModels[0] = 1
				printMenu()
			else:
				checkModels[0] = ~checkModels[0]
		if (key == ord('2')):			
			if (checkModels[1] == -1):
				[genderModel, genderMean] = initGenderModel()
				checkModels[1] = 1
				printMenu()
			else:
				checkModels[1] = ~checkModels[1]
		if (key == ord('3')):
			if (checkModels[2] == -1):
				[ageModel, ageMean] = initAgeModel()
				checkModels[2] = 1
				printMenu()
			else:
				checkModels[2] = ~checkModels[2]
		if (key == ord('4')):
			if (checkModels[3] == -1):
				[gazeModel, gazeMean] = initGazeModel()
				checkModels[3] = 1
				printMenu()
			else:
				checkModels[3] = ~checkModels[3]
		if (key == ord('5')):	
			if (checkModels[4] == -1):
				expressModel = initExpressionModel()
				checkModels[4] = 1
				printMenu()
			else:
				checkModels[4] = ~checkModels[4]
		if (key == ord('6')):
			if (checkModels[5] == -1):
				attributeModel = initAttributeModel()
				checkModels[5] = 1
				printMenu()
			else:
				checkModels[5] = ~checkModels[5]
		
	allProcTime += (datetime.datetime.now() - tAllStart).total_seconds()	
	
	if (frameCount != 0):
		detetctTime /= frameCount
		allProcTime /= frameCount
		
	if (faceCount != 0):
		recognTime /= faceCount
		
	# print('Average  detection  time: ' + str(detetctTime))
	# print('Average recognition time: ' + str(recognTime))
	print('Average processing  time: ' + str(allProcTime))
	
def printMenu():
	print('\nPress button [1]-[6] to turn on / off one of the functions:\n[1]. Recognition\n' + 
			'[2]. Gender\n[3]. Age\n[4]. Gaze\n[5]. Expression\n[6]. Attributes\n[q]. Exit\n[Space]. Pause')		
	
caffe_model_path = "./MODELS"	
	
def initDetectionModel():
	#detection CNNs
	PNet = caffe.Net(caffe_model_path + "/DETECTION/det1.prototxt", caffe_model_path + 
				"/DETECTION/det1.caffemodel", caffe.TEST)
	RNet = caffe.Net(caffe_model_path + "/DETECTION/det2.prototxt", caffe_model_path + 
				"/DETECTION/det2.caffemodel", caffe.TEST)
	ONet = caffe.Net(caffe_model_path + "/DETECTION/det3.prototxt", caffe_model_path + 
				"/DETECTION/det3.caffemodel", caffe.TEST)
	detectCNNs = [PNet, RNet, ONet]
	
	return detectCNNs

def initRecognitionModel():
	#Recognition CNN
	recCNN = caffe.Net(caffe_model_path + "/RECOGNITION/face_deploy.prototxt",
						caffe_model_path + "/RECOGNITION/face_model.caffemodel", caffe.TEST)
	return recCNN
	
def initGenderModel():	
	# Gender CNN base on VGG
	meanGender = [103.939, 116.779, 123.68]	
	gender_net_pretrained = caffe_model_path + '/AGEGENDER/gender.caffemodel'
	gender_net_model_file = caffe_model_path + '/AGEGENDER/gender_deploy.prototxt'
	genderCNN = caffe.Classifier(gender_net_model_file, gender_net_pretrained, caffe.TEST)
	
	return [genderCNN, meanGender]
	
def initAgeModel():
	#Age CNN base on VGG
	meanAge = [103.939, 116.779, 123.68]	
	age_net_pretrained = caffe_model_path + '/AGEGENDER/dex_chalearn_iccv2015.caffemodel'
	age_net_model_file = caffe_model_path + '/AGEGENDER/age_deploy.prototxt'
	ageCNN = caffe.Classifier(age_net_model_file, age_net_pretrained, caffe.TEST)  
	
	return [ageCNN, meanAge]

def initGazeModel():		
	#Attention and gaze CNN
	gaze_net_pretrained = caffe_model_path + '/ATTENTION/binary_w.caffemodel'
	gaze_net_model_file = caffe_model_path + '/ATTENTION/deploy_demo.prototxt'	
	gazeCNN = caffe.Net(gaze_net_model_file, gaze_net_pretrained, caffe.TEST)		
	gazeMeans = baseFeatures = np.zeros((227, 227, 3, 2), np.float32)
	gazeMeans[:,:,:,0] = scipy.io.loadmat(caffe_model_path + '/ATTENTION/places_mean_resize.mat')['image_mean']
	gazeMeans[:,:,:,1] = scipy.io.loadmat(caffe_model_path + '/ATTENTION/imagenet_mean_resize.mat')['image_mean']
	
	return [gazeCNN, gazeMeans]

def initExpressionModel():	
	express_net_pretrained = caffe_model_path + '/EXPRESSION/ck.caffemodel'
	express_model_file = caffe_model_path + '/EXPRESSION/deploy.prototxt'	
	expressCNN = caffe.Net(express_model_file, express_net_pretrained, caffe.TEST)		
	
	return expressCNN
	
def initAttributeModel():	
	attribute_net_pretrained = caffe_model_path + '/ATTRIBUTE/moon_tiny_iter_1000000.caffemodel'
	attribute_net_model_file = caffe_model_path + '/ATTRIBUTE/deploy.prototxt'
	attributeCNN = caffe.Net(attribute_net_model_file,	attribute_net_pretrained, caffe.TEST)				   
	attributeCNN.blobs['data'].reshape(1, 3, 218, 178)
	
	return attributeCNN
	
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--inputFilePath', default = 'VIDEO/test4.mp4')
	parser.add_argument('-b', '--baseFolderPath', default = 'BASE')
	parser.add_argument('-f', '--functionString', default = '111110')
	return parser.parse_args()	

if __name__ == "__main__":	
	args = parse_args()
	
	caffe_path = 'C:/Caffe/python'	#change path!!!!!!!
	if caffe_path not in sys.path:
		sys.path.insert(0, caffe_path)

	caffe.set_mode_gpu()

	detectModel= initDetectionModel()
	
	checkModels = np.array([int(i) for i in args.functionString])
	checkModels = 2 * checkModels - 1
	
	videoProcessing(args.inputFilePath, detectModel, checkModels, args.baseFolderPath)	
	
