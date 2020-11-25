# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
import math as m
import numpy as np
from PIL import Image
#from pose_engine import PoseEngine
from pose_engine_bp import PoseEngine, EDGES, BODYPIX_PARTS #project-bodypix "cp pose_engine.py pose_engine_bp.py"
import cv2
stt=200 #start frame
maxfnum=200 #number of frames
kernel = np.ones((3,3), np.uint8) #delute kernel
kernel2 = np.ones((9, 9), np.uint8) #kernel 2 for delute or
wid0=480
hei0=480
hwid0=wid0//2
hhei0=hei0//2
isz=24
wsz=32
scl=2
print("intsize=",isz*2,"search window=",wsz*2,file=sys.stderr)
ofnm='cut3'
model = 'models/bodypix_resnet_50_640_480_16_quant_edgetpu_decoder.tflite' #resnet model 640x480 very slow 
#model = 'models/bodypix_mobilenet_v1_075_640_480_16_quant_edgetpu_decoder.tflite' #fast model
model = 'models/bodypix_resnet_50_416_288_16_quant_edgetpu_decoder.tflite' # resnet model slow
ofnm2=(ofnm+str(stt)+str(maxfnum))
engine = PoseEngine(model)
cap=cv2.VideoCapture(ofnm+'.mp4')# change your own mp4
ret,frame=cap.read()
av=np.zeros(frame.shape,dtype=float) 
sd=np.zeros(frame.shape,dtype=float)
dif=np.zeros(frame.shape,dtype=float) 
flength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sd0=0
av0=0
if os.path.exists(ofnm2+'av'+'.npy') and os.path.exists(ofnm2+'sd'+'.npy') and os.path.exists(ofnm2+'sd0'+'.npy'):
    av=np.load(ofnm2+'av'+'.npy')
    sd=np.load(ofnm2+'sd'+'.npy')
    sd0=np.load(ofnm2+'sd0'+'.npy')
    print(av.shape,sd.shape,sd0.shape,file=sys.stderr)
else:
    if flength<stt+maxfnum: 
        print("exceed max frame %d",flength)
        exit()
    cap.set(cv2.CAP_PROP_POS_FRAMES,stt)
    for i in range(maxfnum) :
        ret,oframe=cap.read()
        of=oframe.astype('float')
        av0=av0+np.average(of)/maxfnum
        sd0=sd0+np.average(of)*np.average(of)/maxfnum
        for j in range(3):
            av[:,:,j]=av[:,:,j]+of[:,:,j]/maxfnum
            sd[:,:,j]=sd[:,:,j]+of[:,:,j]*of[:,:,j]/maxfnum
        if i%100 == 0 : print(i,file=sys.stderr)
    sd=sd-av*av
    np.where(sd<0,0,sd)
    sd0=sd0-av0*av0
    sd=np.sqrt(sd)
    sd0=m.sqrt(sd0)
    sd=np.nan_to_num(sd)
    np.save(ofnm2+'av'+'.npy',av)
    np.save(ofnm2+'sd'+'.npy',sd)
    np.save(ofnm2+'sd0'+'.npy',sd0)
#print(sd0,np.max(sd),np.count_nonzero(np.isnan(sd)))

timpos=0
bdp=np.zeros((maxfnum,6,17,3,))
tbdp=np.zeros((17,2))
rada=-0.0
rada2=m.radians(-0)
cgv=np.zeros(2)
bcen=np.array([hhei0,hwid0,1])
rta=np.array([[m.cos(rada),-m.sin(rada)],[m.sin(rada),m.cos(rada)]])
cap.set(cv2.CAP_PROP_POS_FRAMES,stt)
org=np.zeros((wsz*2,wsz*2))
dstt=np.zeros((wsz*2,wsz*2))
while(timpos<maxfnum):
    ret,frame =cap.read()
    dif=frame-av
    dmax=np.max(dif)
    dmin=np.min(dif)
    mask=np.where(np.abs(dif)>sd*1.5,1,0)
    maskk=np.where(sd<sd0*5,0,1)
    mask=mask*maskk
    mask2=cv2.bitwise_or(cv2.bitwise_or(mask[:,:,0],mask[:,:,1]),mask[:,:,2])
    dif=(dif-dmin)/(dmax-dmin)*255
    dif2=mask
    mask2=mask2.astype('uint8')
    mask2=cv2.erode(mask2,kernel)
    mask2=cv2.dilate(mask2,kernel2)
    mu=cv2.moments(mask2,False)
    cgv=[int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])]
    mask2=np.where(mask2==0,0.4,mask2)
    for j in range(3):
        dif2[:,:,j]=mask2*frame[:,:,j]
    dif2=dif2.astype('uint8')
    difg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.circle(dif2,(cgv[0],cgv[1]),10,(0,128,int(255)),-1)
    difh2=cv2.resize(dif2,(int(frame.shape[1]//3),int(frame.shape[0]//3)))
    #cv2.imshow('mask2',difh2)
    cgv[0]=cgv[0]//2
    cgv[1]=cgv[1]//2
    fram2=cv2.resize(frame,(int(frame.shape[1]//scl),int(frame.shape[0]//scl)))
    difgh=cv2.resize(difg,(int(frame.shape[1]//scl),int(frame.shape[0]//scl)))
    #print(difgh.shape,file=sys.stderr)
    if timpos==0:orgim=np.copy(fram2)
    if cgv[0]>hwid0: 
        x0=int(cgv[0]-hwid0)
        cgx=int(cgv[0])
        if cgv[0]+hwid0<frame.shape[1]//2:
            x1=int(cgv[0]+hwid0)
        else: x1=int(frame.shape[1]//2)
    else:
        x0=0
        cgx=hwid0
        x1=wid0
    if cgv[1]>hhei0:
        y0=int(cgv[1]-hhei0)
        cgy=int(cgv[1])
        if cgv[1]+hhei0-1<frame.shape[0]//2:
            y1=int(cgv[1]+hhei0)
        else: y1=int(frame.shape[0]//2)
    else:
        y0=0
        cgy=hhei0-1
        y1=hei0-1
    sft=np.array([y0,x0])
    dst=fram2[y0:y1,x0:x1,:] #nomalcaes 0 crop area of image
    #dst=cv2.rotate(dst,cv2.ROTATE_180) #upsidedown case 1
    #dst=cv2.rotate(dst,cv2.ROTATE_90_COUNTERCLOCKWISE) #ac90 case 3
    #dst=cv2.rotate(dst,cv2.ROTATE_90_CLOCKWISE) #c90 case 4
    #print(hhei0,hwid0)
    rm=cv2.getRotationMatrix2D((dst.shape[1]//2,dst.shape[0]//2),-m.degrees(rada2),1) #free rotation case 5
    dst2=cv2.warpAffine(dst,rm,(dst.shape[1],dst.shape[0])) #free rotation case 5

    #pilim=Image.fromarray(difh2) #background subtract image
    pilim=Image.fromarray(dst2)
    cvim=np.asarray(pilim)
    #poses, inference_time = engine.DetectPosesInImage(np.uint8(pilim))
    inference_time, poses, heatmap, bodyparts = engine.DetectPosesInImage(np.uint8(pilim))
    #poses2, inference_time2 = engine.DetectPosesInImage(np.uint8(pilim2))
    i=0
    if len(poses)==0:
        rada=-0.0
        #bcen=np.array([0,0,1])
        #rta=np.array([[m.cos(rada),-m.sin(rada)],[m.sin(rada),m.cos(rada)]])
    for pose in poses:
        if i==1 :continue
        if pose.score <0.0: continue
            #rada=-0.0
            #bcen=np.array([0,0,1])    
        j=0
        for label, keypoint in pose.keypoints.items():
            k=keypoint.score
            tbdp[j,1]=keypoint.yx[1]
            tbdp[j,0]=keypoint.yx[0]
            #tbdp[j,1]=wid0-keypoint.yx[1]+x0
            #tbdp[j,0]=hei0-keypoint.yx[0]+y0
            #tbdp[j,1]=wid0-keypoint.yx[0]+x0
            #tbdp[j,0]=keypoint.yx[1]+y0
            #tbdp[j,1]=keypoint.yx[0]+x0
            #tbdp[j,0]=hei0-keypoint.yx[1]+y0

            bdp[timpos,i,j,2]=keypoint.score
            cv2.circle(cvim,(keypoint.yx[1],keypoint.yx[0]),4,(int(k*i*255),int(k*j*255/16),int(255*k)),-1)
            j+=1
        tbdp=tbdp-bcen[0:2]
        rta[0,0]=m.cos(rada2)
        rta[1,1]=m.cos(rada2)
        rta[0,1]=m.sin(rada2)
        rta[1,0]=-m.sin(rada2)
        tbdp=np.dot(tbdp,rta)+bcen[0:2]
        tbdp=tbdp+sft
        #print(bdp[timpos,i,1,2],tbdp[1,1])
        bdp[timpos,i,:,0:2]=tbdp
        #print(bdp[timpos,i,1,2],tbdp[1,1])
        shld=(tbdp[5,:]+tbdp[6,:])/2
        hp=(tbdp[11,:]+tbdp[12,:])/2
        bcen2=(shld+hp)/2
        ang=hp-shld
        #if ang[0]>0:rada=m.atan(ang[1]/ang[0])
        #else:rada=m.atan(ang[1]/ang[0])-m.pi
        rada=m.atan2(ang[1],ang[0])
        print(timpos,pose.score,m.degrees(rada))
        
        cv2.line(fram2,(int(shld[1]),int(shld[0])),(int(hp[1]),int(hp[0])),(255,0,0))
        cv2.circle(fram2,(int(bcen2[1]),int(bcen2[0])),10,(0,0,255),-1)
        cv2.circle(orgim,(int(bcen2[1]),int(bcen2[0])),3,(0,0,255),-1)
        cv2.circle(orgim,(int(cgv[0]),int(cgv[1])),3,(0,255,255),-1)

        for j in {15,16}:
            k=bdp[timpos,i,j,2]
            cv2.circle(orgim,(int(tbdp[j,1]),int(tbdp[j,0])),2,(0,128,int(255*k)),-1)
            cv2.circle(fram2,(int(tbdp[j,1]),int(tbdp[j,0])),2,(0,128,int(255*k)),-1)
        cx=int(tbdp[16,1])
        cy=int(tbdp[16,0])
        #print(cx,cy,file=sys.stderr)
        
        i+=1
    if timpos==0:
            org=org.astype('uint8')
            dstt=dstt.astype('uint8')
            org[wsz-isz:wsz+isz,wsz-isz:wsz+isz]=difgh[cy-isz:cy+isz,cx-isz:cx+isz]
            oy=cy
            ox=cx
            cv2.rectangle(fram2,(cx-isz,cy-isz),(cx+isz,cy+isz),(255,0,0),3)
    else :
            if oy-wsz<0 or ox-wsz<0 or oy+wsz>difgh.shape[0] or ox+wsz>difgh.shape[1] :break
            dstt[:,:]=difgh[oy-wsz:oy+wsz,ox-wsz:ox+wsz]
            #cv2.imshow('dist',dstt)
            
            
            org2=org[wsz-isz:wsz+isz,wsz-isz:wsz+isz]
            orggg=org.astype('float')
            dstt2=dstt.astype('float')
            #d,e=cv2.phaseCorrelate(orggg,dstt2)
            #d=(0,0)
            #dx1,dy1=d
            #print(dx1,dy1,file=sys.stderr)
            dstt3=np.uint8(dstt)
            orgg=np.uint8(org2)
            #cv2.imshow('dist2',dstt3)
            #cv2.imshow('diff',orgg)
            res=cv2.matchTemplate(dstt3,orgg,cv2.TM_CCOEFF_NORMED)
            t1,t2,lmin,lmax=cv2.minMaxLoc(res)
            dx2=lmax[0]-wsz+isz
            dy2=lmax[1]-wsz+isz
            #print(ox,oy,dx2,dy2,file=sys.stderr)
            #cv2.arrowedLine(orgim,(ox,oy),(ox+int(dx1),oy+int(dy1)),(0,255,255),thickness=2,line_type=4)
            #cv2.arrowedLine(fram2,(ox,oy),(ox+int(dx1),oy+int(dy1)),(0,255,255),thickness=2,line_type=4)
            cv2.arrowedLine(orgim,(ox,oy),(ox+dx2,oy+dy2),(255,255,0),thickness=2,line_type=4)
            cv2.arrowedLine(fram2,(ox,oy),(ox+dx2,oy+dy2),(255,255,0),thickness=2,line_type=4)
            cv2.rectangle(fram2,(ox-isz,oy-isz),(ox+isz,oy+isz),(255,0,0),3)
            ox=ox+dx2
            oy=oy+dy2
            org[wsz-isz:wsz+isz,wsz-isz:wsz+isz]=difgh[oy-isz:oy+isz,ox-isz:ox+isz]
    rada2=rada
    cv2.imshow('soukan',dst)
    cv2.imshow('fram2',fram2)
    cv2.imshow('inference',cvim)
    cv2.imshow('trajectory',orgim)
    timpos+=1
    if cv2.waitKey(1)&0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()