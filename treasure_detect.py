import cv2
import math
import time
# col_np=[[73,0,42,137,255,230],#蓝色
#         [0,14,78,34,197,255]#红色
# ]
import numpy as np

def constrain(x, a, b):
    if x >= a-b and x<=a+b: 
        return True
    else:
        return False

def treasure_detect(img,team_color):
    col_np=[[21,64,0,128,255,179],#蓝色 ,H-L<106, h_h>128  s_l=0 s_h>234   v_l>40 && vl<110 v_h>170
        
        [0,42,80,251,133,145]#红色, vl<170 and vl>40  h_l=0  h_h=24 s_L>30 and s_l<80  s_H>190  v_h>207
]
    if img is None:
        print("image is None")
        return None
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.medianBlur(img,5)
    img = cv2.resize(img, (350, 350))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_1 = cv2.inRange(hsv, np.array(col_np[0][0:3]), np.array(col_np[0][3:6]))#blue
    mask_2 = cv2.inRange(hsv, np.array(col_np[1][0:3]), np.array(col_np[1][3:6]))#red   最好修改为蓝色的hsv

    image_mask_1 = mask_1
    image_mask_2 = mask_2

    mask_reslut= 0.0
    mask_num1 = 0.0
    mask_num2 = 0.0
    # add_result = 0.0
    image_color=str()

    if team_color=="red":
        mask_num1 = np.sum(image_mask_1,dtype=np.float32)
        mask_num2 = np.sum(image_mask_2,dtype=np.float32)
        mask_reslut = mask_num2 - mask_num1

        if np.sum(mask_reslut) > 0:
            image_color = 'red'    
        else :
            image_color = 'blue'

    elif team_color=="blue":
        mask_num1 = np.sum(image_mask_1,dtype=np.float32)
        mask_num2 = np.sum(image_mask_2,dtype=np.float32)

        mask_reslut = mask_num1 - mask_num2
        if np.sum(mask_reslut) > 0:
            image_color = 'blue'    
        else :
            image_color = 'red'
        
    print(image_color)
    
    result_str = str()
    position = [0,0]
    

    if image_color == 'blue':
        mask = mask_1
        kernel = np.ones((3,3), np.uint8)  # 设置开运算所需核
        # kernel_1 = np.ones((1, 1), np.uint8)
        
        __, threshold = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        # threshold = cv2.erode(threshold, kernel, iterations=1)
       
        # threshold= cv2.Canny(threshold, 0, 255)#膨胀
        
        threshold= cv2.dilate(threshold, kernel, iterations=1)#膨胀三次
        threshold= cv2.erode(threshold, kernel, iterations=1)

       
        # cv2.imshow("img",threshold)
        # cv2.waitKey(0)
        contour,hierarch=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        # print(len(contour))
        cnt_num=0
        for bound in contour:
            approx = cv2.approxPolyDP(bound, 
                    0.05 * cv2.arcLength(bound, True), True)
            
            # print(len(approx))
            # image_test = np.zeros((350, 350), dtype=np.uint8)
            # 定义顶点坐标
             
            # 绘制多边形
            # cv2.polylines(image_test, approx, isClosed=True, color=(255, 255, 255), thickness=2)
            if len(approx) == 4:#判断矩形
                x, y, w, h = cv2.boundingRect(bound)
                if w <15 or h <15 :
                    continue
                _, row, _ = hierarch.shape
                cnt = 0
                for j in range(row):
                    cnt += 1
                    if hierarch[0][j][2] >= 0 :
                        child_contour_index = hierarch[0][j][2]
                        child_contour = contour[child_contour_index]
                        approx = cv2.approxPolyDP(child_contour, 
                            0.07 * cv2.arcLength(child_contour, True), True)
                        print(len(approx))
                        if len(approx) == 3:#判断三角形
                            cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
                            if team_color == image_color:
                                result_str = "this is my fake treasure"
                                position = [x+w/2,y+h/2]
                                print(result_str,position,image_color)

                                return result_str,position,image_color
                            else :
                                result_str = "this is oppnent fake treasure"
                                position =  [x+w/2,y+h/2]

                                print(result_str,position,image_color)
                                return result_str,position,image_color
                        else :
                            #圆形检测
                            print("circle")
                            # print(len(child_contour))
                            area = cv2.contourArea(child_contour)
                            # length = cv2.arcLength(child_contour, True)
                            # print(area/length)


                            perimeter = cv2.arcLength(child_contour, True)
                            (x, y), radius = cv2.minEnclosingCircle(child_contour)
                            aspect_ratio = float(perimeter) / (2 * radius * math.pi)
                            circularity = (4 * math.pi * area) / (perimeter * perimeter)

                            if 0.6 <= aspect_ratio <= 1.3 and 0.6 <= circularity <= 1.4:
                            # if constrain(area/length,10.7,2):
                                if  team_color == image_color:
                                    result_str = "this is my true treasure"
                                    position = [x+w/2,y+h/2]

                                    print(result_str,position,image_color)
                                    return result_str,position,image_color
                                else :
                                    result_str = "this is opponent true treasure"
                                    position = [x+w/2,y+h/2]

                                    print(result_str,position,image_color)
                                    return result_str,position,image_color
                            else :
                                if cnt == row-1:
                                    result_str = "error,not circle and triangle"
                                    # position = [0,0]
                                    print(result_str)

                                    return result_str
                                
            
            cnt_num+=1
            result_str = "error,not rectangle"
            if cnt_num == len(contour)-1:
                result_str = "error,not rectangle"
                # position = [0,0]
                print(result_str)
                return result_str
                
    elif image_color == 'red':
        
        mask = mask_2
        kernel = np.ones((3,3), np.uint8)  
        kernel_1 = np.ones((5,5), np.uint8)
        
        __, threshold = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
       
        # threshold = cv2.erode(threshold, kernel, iterations=1)
        # threshold= cv2.Canny(threshold, 0, 255)#膨胀
        
        threshold= cv2.dilate(threshold, kernel, iterations=1)#膨胀三次
        threshold= cv2.erode(threshold,kernel,iterations=1)
        
        # threshold= cv2.dilate(threshold, kernel, iterations=2)
       

        # cv2.imshow("img",threshold)
        # cv2.waitKey(0)
        contour,hierarch=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contour))
        cnt_num=0
        for bound in contour:
            approx = cv2.approxPolyDP(bound, 
                    0.05 * cv2.arcLength(bound, True), True)
            # print(len(approx))
            # image_test = np.zeros((350, 350), dtype=np.uint8)
            # 定义顶点坐标
            # 绘制多边形
            # cv2.polylines(image_test, approx, isClosed=True, color=(255, 255, 255), thickness=2)
            
            if len(approx) == 4:#判断矩形
                x, y, w, h = cv2.boundingRect(bound)
                if w <10 or h <10 :
                    continue
                _, row, _ = hierarch.shape
                cnt = 0
                for j in range(row):
                    cnt += 1
                    if hierarch[0][j][2] >= 0 :
                        child_contour_index = hierarch[0][j][2]
                        child_contour = contour[child_contour_index]
                        approx = cv2.approxPolyDP(child_contour, 
                            0.05 * cv2.arcLength(child_contour, True), True)
                        print(len(approx))
                        if len(approx) == 3:#判断三角形
                            cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
                            if team_color == image_color:
                                result_str = "this is my true treasure"
                                position = [x+w/2,y+h/2]
                                return result_str,position,image_color
                            else :
                                result_str = "this is oppnent ture treasure"
                                position =  [x+w/2,y+h/2]
                                return result_str,position,image_color
                        # elif len(approx) >=5:
                        else :
                            #圆形检测
                            print("circle")

                            area = cv2.contourArea(child_contour)
                            # length = cv2.arcLength(child_contour, True)
                            # print(area/length)
                            # if constrain(area/length,10.7,2):
                            perimeter = cv2.arcLength(child_contour, True)
                            (x, y), radius = cv2.minEnclosingCircle(child_contour)
                            aspect_ratio = float(perimeter) / (2 * radius * math.pi)
                            circularity = (4 * math.pi * area) / (perimeter * perimeter)

                            if 0.6 <= aspect_ratio <= 1.3 and 0.6 <= circularity <= 1.4:
                                if  team_color == image_color:
                                    result_str = "this is my fake treasure"
                                    position = [x+w/2,y+h/2]
                                    return result_str,position,image_color
                                else :
                                    result_str = "this is oppnent fake  treasure"
                                    position = [x+w/2,y+h/2]
                                    return result_str,position,image_color
                            else :
                                if cnt == row-1:
                                    result_str = "error,not circle and triangle"
                                    position = [0,0]
                                    return result_str,position,image_color
            
            cnt_num+=1
            result_str = "error,not rectangle"
            if cnt_num == len(contour)-1:
                result_str = "error,not rectangle"
                position = [0,0]
                return result_str,position,image_color

print("hello_ok")      
print(__name__)     
if __name__ == "__main__":
    print("hello")
    vedio = cv2.VideoCapture(0)
    last_time=time.time()
    

# def vision_detect(self,color):
    # vedio = cv2.VideoCapture(0)
    # last_time=time.time()
    ret, frame = vedio.read()
    flag=0
    result,_,_ = treasure_detect(frame,team_color="red")
    while True:
        now_time= time.time()
        if now_time - last_time >=0.3:
            last_time=now_time
            ret, frame = vedio.read()
            # print(treasure_detect(frame,"red"))     
            now_result,_,_=treasure_detect(frame,team_color="red")

            if now_result== result and flag>=0:
                flag+=1
                if flag==3:
                    print(now_result)
                    break
                    # return now_result
                    
            else :
                # flag=-1
                print("error")
                break
    
    # reslut=treasure_vision(image,team_color=color)
    # if reslut == None:
    #     return []
        # cnt=0
        # while True:
        #     now_time=time.time()
        #     if now_time-last_time==0.5:
        #         last_time=now_time
        #         image = self.img_processer.img_queue.get()
        #         now_result=treasure_vision(image,team_color=color)

        #         if now_result== reslut and flag>=0:
        #             flag+=1
        #             if flag==3:
        #                 return now_result
        #         else :
        #             flag=-1
                