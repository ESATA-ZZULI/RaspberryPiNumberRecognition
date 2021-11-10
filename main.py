import cv2 as cv
import math
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from PIL import Image
from functools import reduce

cap = cv.VideoCapture(0)

font=cv.FONT_HERSHEY_SIMPLEX
interpreter = Interpreter(model_path="./tflite_float_model.tflite")

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    squares = []
    img = cv.GaussianBlur(img, (3, 3), 0)   
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bin = cv.Canny(gray, 30, 100, apertureSize=3)
    _, RedThresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    img2, contours, hierarchy  = cv.findContours(RedThresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    #print("轮廓数量：%d" % len(contours))
    index = 0
    # 轮廓遍历
    def doOrder(rec, center):
        try:
            lt = list(filter(lambda x:x[0]<center[0] and x[1]<center[1], rec))[0]
            rt = list(filter(lambda x:x[0]>center[0] and x[1]<center[1], rec))[0]
            lb = list(filter(lambda x:x[0]<center[0] and x[1]>center[1], rec))[0]
            rb = list(filter(lambda x:x[0]>center[0] and x[1]>center[1], rec))[0]
        except:
            return rec
        
        return np.array([rb, lb, lt, rt])
    
    def reduceRes(sqrs):
        new_res = []
        skip = False
        for i in range(1, len(sqrs)):
            if skip:
                skip = False
                continue
            
            rec0 = sqrs[i-1]
            rec1 = sqrs[i]
            
            if abs(rec0[0][0] - rec1[0][0]) < 10 and abs(rec0[0][1] - rec1[0][1]) < 10:
                new_res.append(rec0)
                skip = True
        return new_res
    
    for cnt in contours:
        cnt_len = cv.arcLength(cnt, True) #计算轮廓周长
        cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True) #多边形逼近
        # 条件判断逼近边的数量是否为4，轮廓面积是否大于1000，检测轮廓是否为凸的
        if len(cnt) == 4 and 50000 > cv.contourArea(cnt) > 4000 and cv.isContourConvex(cnt):
            M = cv.moments(cnt) #计算轮廓的矩
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])#轮廓重心
            
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            # 只检测矩形（cos90° = 0）
            if max_cos < 0.5:
            # 检测四边形（不限定角度范围）
            #if True:
                index = index + 1
                #cv.putText(img,("#%d"%index),(cx,cy),font,0.7,(255,0,255),2)
                cnt = doOrder(cnt, (cx, cy))
                squares.append(cnt)
    
    squares.sort(key=lambda s: s[0][0])
    squares = reduceRes(squares)
    
    
    return squares, img, RedThresh

def makePerspect(p, frame):    
    orignal_W = math.ceil(np.sqrt((p[3][1] - p[2][1])**2 + (p[3][0] - p[2][0])**2))
    orignal_H = math.ceil(np.sqrt((p[3][1] - p[0][1])**2 + (p[3][0] - p[0][0])**2))

    # 原图中的四个顶点,与变换矩阵
    pts1 = np.float32([p[0], p[1], p[2], p[3]])
    pts2 = np.float32([[int(orignal_W+1),int(orignal_H+1)], [0, int(orignal_H+1)], [0, 0], [int(orignal_W+1), 0]])
    
    M = cv.getPerspectiveTransform(pts1,pts2)
    dstImage = cv.warpPerspective(frame,M, (int(orignal_W+3),int(orignal_H+1)))
    return dstImage

def convertTo2828(img):
    return cv.resize(img, (28, 28))

def predict(img):
    #cv.imwrite("8.bmp", img)
    #exit()
    img_arr = np.asarray(img, dtype=np.float32)
    
    img_arr = np.reshape(img_arr, (1, 784))
    img_arr /= 255
    interpreter.set_tensor(input_details[0]['index'],img_arr)
    interpreter.invoke()
    preres = interpreter.get_tensor(output_details[0]['index'])[0]
    label = zip(range(10), preres)

    return max(label, key=lambda x:x[1])

while(1):
    _,frame = cap.read()
    
    squares, img, binimg = find_squares(frame)
    cv.drawContours( img, squares, -1, (0, 0, 255), 2 )
    
    indexofs = 0
    rec_res = []
    for square in squares:
        if len(square) != 4:
            continue
        
        indexofs += 1
        dstimg = makePerspect(square, binimg)
        img28 = convertTo2828(dstimg)
        pres = predict(img28)
        rec_res.append(pres)
        cv.putText(img,("%d, %.2f%%"%(pres[0], pres[1]*100)),(square[2][0]+10, square[2][1]+40),font,0.7,(255,0,255),2)
        cv.imshow(f"persimg{pres[0]}", img28)
    print(rec_res)
        
    cv.imshow ('frame',img)

    k = cv.waitKey (5) & 0xFF
    if k ==27:
        break
cv.destroyAllWindows()