import cv2 
import numpy as np 
import time 

def check_sum(data):
    sum = 0
    for i in range(len(data)):
        sum += data[i]
    # print(sum)
    return sum

def get_color_block(img):
    if len(img.shape==2):
        raise ValueError("Please input a bgr image") 
    masks=[]
    colos=[]
    return masks,colos 

destination_points = np.float32([[154,56], [340,56], [154, 423], [340, 425]])

# Specify the destination points - points in the output image
# Points are in order (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
original_points = np.float32([[156,45], [359,46], [53,429], [457,429]])

# Compute the perspective transform matrix
M = cv2.getPerspectiveTransform( original_points,destination_points)

def get_with_perspective(img):
    transformed_image = cv2.warpPerspective(img, M, img.shape[:2])
    return transformed_image 

def pre_process(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    img=cv2.GaussianBlur(img,(15,15),0)
    _, img = cv2.threshold(img, 100, 255,cv2.THRESH_BINARY_INV)#|cv2.THRESH_OTSU
    #自适应
    # img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,7,-2)
    # img=cv2.Canny(img,50,150)
    img=cv2.dilate(img,None,iterations=5)# 对图像进行膨胀操作
    img=cv2.erode(img,None,iterations=5)# 对图像进行腐蚀操作
    # img=cv2.dilate(img,None,iterations=4)# 对图像进行膨胀操作
 
    return img

def get_line_cross(img,draw=True):
    # need a gray image
    if len(img.shape)==3:
        # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=pre_process(img)
    img=get_with_perspective(img)
    rows,cols=img.shape
    right_pos=[]
    left_pos=[]
    up_pos=[]
    down_pos=[]
    #detect in a rectangle
    len_rect_r=int(rows*0.64)
    len_rect_c=int(cols*0.64)
    line_right=img[cols//2-len_rect_c//2,rows//2-len_rect_r//2:rows//2+len_rect_r//2]
    line_left=img[cols//2+len_rect_c//2,rows//2-len_rect_r//2:rows//2+len_rect_r//2]
    line_up=img[cols//2-len_rect_c//2:cols//2+len_rect_c//2,rows//2-len_rect_r//2]
    line_down=img[cols//2-len_rect_c//2:cols//2+len_rect_c//2,rows//2+len_rect_r//2]
    # print(line_right.shape)
    pos_righ=np.argwhere(line_right>0) 
    pos_left=np.argwhere(line_left>0) 
    pos_up=np.argwhere(line_up>0) 
    pos_down=np.argwhere(line_down>0) 
    if len(pos_righ)!=0:
        pos_righ=pos_righ.mean()+rows/2-len_rect_r/2
        pos_righ_xy=(int(pos_righ),int(cols/2-len_rect_c/2))
        up_pos.append(pos_righ_xy)
    if len(pos_left)!=0:
        pos_left=pos_left.mean()+rows/2-len_rect_r/2
        pos_left_xy=(int(pos_left),int(cols/2+len_rect_c/2))
        down_pos.append(pos_left_xy)
    if len(pos_up)!=0:
        pos_up=pos_up.mean()+cols/2-len_rect_c/2
        pos_up_xy=(int(rows/2-len_rect_r/2),int(pos_up))
        left_pos.append(pos_up_xy)
    if len(pos_down)!=0:
        pos_down=pos_down.mean()+cols/2-len_rect_c/2
        pos_down_xy=(int(rows/2+len_rect_r/2),int(pos_down))
        right_pos.append(pos_down_xy)
    
    len_rect_r=int(rows*0.5)
    len_rect_c=int(cols*0.5)
    line_right=img[cols//2-len_rect_c//2,rows//2-len_rect_r//2:rows//2+len_rect_r//2]
    line_left=img[cols//2+len_rect_c//2,rows//2-len_rect_r//2:rows//2+len_rect_r//2]
    line_up=img[cols//2-len_rect_c//2:cols//2+len_rect_c//2,rows//2-len_rect_r//2]
    line_down=img[cols//2-len_rect_c//2:cols//2+len_rect_c//2,rows//2+len_rect_r//2]
    pos_righ=np.argwhere(line_right>0) 
    pos_left=np.argwhere(line_left>0) 
    pos_up=np.argwhere(line_up>0) 
    pos_down=np.argwhere(line_down>0) 
    if len(pos_righ)!=0:
        pos_righ=pos_righ.mean()+rows/2-len_rect_r/2
        pos_righ_xy=(int(pos_righ),int(cols/2-len_rect_c/2))
        up_pos.append(pos_righ_xy)
    if len(pos_left)!=0:
        pos_left=pos_left.mean()+rows/2-len_rect_r/2
        pos_left_xy=(int(pos_left),int(cols/2+len_rect_c/2))
        down_pos.append(pos_left_xy)
    if len(pos_up)!=0:
        pos_up=pos_up.mean()+cols/2-len_rect_c/2
        pos_up_xy=(int(rows/2-len_rect_r/2),int(pos_up))
        left_pos.append(pos_up_xy)
    if len(pos_down)!=0:
        pos_down=pos_down.mean()+cols/2-len_rect_c/2
        pos_down_xy=(int(rows/2+len_rect_r/2),int(pos_down))
        right_pos.append(pos_down_xy)

    if draw:
 
        #draw circle in the image
        for i in range(len(right_pos)):

            cv2.circle(img,right_pos[i],5,(200,0,0),-1)
            cv2.putText(img,'right',(right_pos[i][0]+10,right_pos[i][1]+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,0,150),2)
        for i in range(len(left_pos)):
            cv2.circle(img,left_pos[i],5,(200,0,0),-1)
            cv2.putText(img,'left',(left_pos[i][0]+10,left_pos[i][1]+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,0,150),2)
        for i in range(len(up_pos)):
            cv2.circle(img,up_pos[i],5,(200,0,0),-1)
            cv2.putText(img,'up',(up_pos[i][0]+10,up_pos[i][1]+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,0,150),2)
        for i in range(len(down_pos)):
            cv2.circle(img,down_pos[i],5,(200,0,0),-1)
            cv2.putText(img,'down',(down_pos[i][0]+10,down_pos[i][1]+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,0,150),2)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    return right_pos ,left_pos ,up_pos ,down_pos ,img 


#只有位于节点位置时才会有两个以上的方向，处于巡线位置时只有一个方向，处于转弯位置时应有两个方向
#结合上一个函数可知，在一个方向上有两个以上的节点时，说明处于转弯位置，此时应该有两个方向
def get_availabledir(right_pos ,left_pos ,up_pos ,down_pos ):
    #get the available direction
    available_dir=[]
    if len(right_pos)==2:
        available_dir.append('right')
    if len(left_pos)==2:
        available_dir.append('left')
    if len(up_pos)==2:
        available_dir.append('up')
    if len(down_pos)==2:
        available_dir.append('down')
    return available_dir


#检测是否处于转弯位置,rx,ry,offx,offy 为与实际位置的缩放比例和偏移量
def get_cross_pos(right_pos ,left_pos ,up_pos ,down_pos,rx,ry,offx,offy ):
    idx=0
    lright_pos,lleft_pos,lup_pos,ldown_pos=len(right_pos),len(left_pos),len(up_pos),len(down_pos) 
    if lright_pos==2 and lleft_pos==2:
        print("发现 横线")
        idx+=1
    if lup_pos==2 and ldown_pos==2:
        print("发现 竖线")
        idx+=1
    if ldown_pos==2 and lright_pos==2:
        print("发现 右转线 ")
        # idx+=1
        # right_pos=np.array(right_pos)
        
        # if right_pos[0,0]==right_pos[1,0]:
    if ldown_pos==2 and lleft_pos==2:
        print("发现 左转线")
        idx+=1
    if ldown_pos==2 and lright_pos==2 and lleft_pos==2:
        print("发现 T型线")
        idx+=1
    if ldown_pos==2 and lright_pos==2 and lleft_pos==2 and lup_pos==2:
        print("发现十字线")
        idx+=1
    if idx==0:
        print("未发现节点")

    return idx

def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

def wether_to_turn(target_dir,poses,obj=None):
    thr=0.15
    if target_dir=='right':
        if len(poses[0])==2 and len(poses[3])==2:
            down_p1,down_p2=np.array(poses[3][0]),np.array(poses[3][1])
            right_p1,right_p2=np.array(poses[0][0]),np.array(poses[0][1])
            # print(f"diff1:{angle_cos(down_p2,down_p1,right_p1)-np.cos(np.pi/4)},diff2:{np.abs(angle_cos(down_p1,down_p2,right_p2)-np.cos(np.pi/4))}")
            if abs(angle_cos(down_p2,down_p1,right_p1)-np.cos(np.pi/4))<thr or abs(angle_cos(down_p1,down_p2,right_p2)-np.cos(np.pi/4))<thr:
                return True
            else :
                return False 
        else:
            return False
    if target_dir=='left':
        if len(poses[1])==2 and len(poses[3])==2:
            down_p1,down_p2=np.array(poses[3][0]),np.array(poses[3][1])
            left_p1,left_p2=np.array(poses[1][0]),np.array(poses[1][1])
            if abs(angle_cos(down_p2,down_p1,left_p1)-np.cos(np.pi/4))<thr or abs(angle_cos(down_p1,down_p2,left_p2)-np.cos(np.pi/4))<thr:
                return True
            else :
                return False
        else:
            return False
    if target_dir=="forward" or target_dir=="up":
        return obj.forward_dis<10
    if target_dir=="backword":
        return True 
         
    

def draw_arrow_based_on_direction(img, direction):
    # Get image dimensions
    height, width, _ = img.shape

    # Calculate the center point of the image
    center_point = (width//2, height//2)

    # Define arrow color (green) and thickness
    arrow_color = (0, 255, 0)
    arrow_thickness = 2

    # Define the end point based on the direction
    if direction == "forward":
        end_point = (width//3, 0)
    elif direction == "backward":
        end_point = (width//3, height)
    elif direction == "left":
        end_point = (0, height//3)
    elif direction == "right":
        end_point = (width, height//3)
    else:
        # If the direction is not recognized, don't draw anything
        return img

    # Draw the arrow
    img = cv2.arrowedLine(img, center_point, end_point, arrow_color, arrow_thickness, tipLength=0.3)

    return img

class PID:
    def __init__(self,  P=0, I=0.00, D=0.00, min_error=10, max_error=200, max_integral=200):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.max_integral = max_integral
        self.max_error = max_error
        self.min_error = min_error
        self.errors = np.array([0, 0, 0])
        self.reset()
        self.last_ti = time.time()
        self.output = 0
        self.last_error = 0.0
        
    def reset(self):
        self.setpoint = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0

    def update(self, feedback):
        error = self.setpoint - feedback
        if (np.abs(error) < self.min_error):
            error = 0
            
        error = max(min(error, self.max_error), -self.max_error)
        self.errors[1:] = self.errors[:-1]
        self.errors[-1] = error
        error = self.errors.mean()
        dt = time.time()-self.last_ti
        self.last_ti = time.time()
        delta_error = (error - self.last_error)/dt

        self.PTerm = self.Kp * error
        self.ITerm += error*dt*(error/self.max_error+1)
        self.DTerm = self.Kd * delta_error
        # 积分限幅
        self.ITerm = max(
            min(self.ITerm, self.max_integral), -self.max_integral)
        self.last_error = error
        self.output = self.PTerm + (self.Ki * self.ITerm) + self.DTerm
        return self.output
    
    def clear(self):
        self.errors=np.array([0, 0, 0])
        self.ITerm=0
        self.last_error=0
        

