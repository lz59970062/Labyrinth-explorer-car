import threading
# cap=cv2.VideoCapture(r'slow.mkv')
import time

import cv2
import numpy as np


def inv(f, y):
    x = (y-f[1])/f[0]
    return x


def draw_line(img, lines, color=[255, 0, 0], thickness=3):
    # 绘制所有的直线
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_angle(img, right_angles, color=[0, 255, 0], thickness=3):
    # 在每个直角交点处绘制一个圆
    for angle in right_angles:
        rho1, theta1, rho2, theta2 = angle
        a1 = np.cos(theta1)
        b1 = np.sin(theta1)
        a2 = np.cos(theta2)
        b2 = np.sin(theta2)
        x1 = a1 * rho1
        y1 = b1 * rho1
        x2 = a2 * rho2
        y2 = b2 * rho2
        if b1 != 0 and b2 != 0 and a1 / b1 != a2 / b2:
            intersection_x = int(
                (y2 - y1 + x1 * a1 / b1 - x2 * a2 / b2) / (a1 / b1 - a2 / b2))
            intersection_y = int((intersection_x - x1) * a1 / b1 + y1)
        else:
            intersection_x = None
            intersection_y = None
        if intersection_x is not None and intersection_y is not None:
            cv2.circle(img, (intersection_x, intersection_y),
                       5, color, thickness)


def dectect_corners(img, draw=True):
    if len(img.shape) > 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raws, cols = img.shape
    # img=cv2.GaussianBlur(img,(5,5),0)
    craw, ccols = raws//2, cols//2
    r_win_w = cols//2
    c_win_w = raws//2
    up_win = img[1, ccols-r_win_w:ccols+r_win_w]
    down_win = img[raws-1, ccols-r_win_w:ccols+r_win_w]
    left_win = img[craw-c_win_w:craw+c_win_w, 1]
    right_win = img[craw-c_win_w:craw+c_win_w, cols-1]
    up_list = np.argwhere(up_win > 0)
    down_list = np.argwhere(down_win > 0)
    left_list = np.argwhere(left_win > 0)
    right_list = np.argwhere(right_win > 0)
    pos = [-1, -1, -1, -1]
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if len(up_list) > 0:
        pos[0] = int(np.mean(up_list)+ccols-r_win_w)
        if draw == True and pos[0] > 0:
            cv2.circle(img, (int(pos[0]), 2), 5, (0, 205, 0), 2)
    if len(down_list) > 0:
        pos[1] = int(np.mean(down_list)+ccols-r_win_w)
        if draw == True and pos[1] > 0:
            cv2.circle(img, (int(pos[1]), raws-2), 5, (0, 205, 0), 2)
    if len(left_list) > 0:
        pos[2] = int(np.mean(left_list)+craw-c_win_w)
        if draw == True and pos[2] > 0:
            cv2.circle(img, (2, int(pos[2])), 5, (0, 205, 0), 2)
    if len(right_list) > 0:
        pos[3] = int(np.mean(right_list)+craw-c_win_w)
        if draw == True and pos[3] > 0:
            cv2.circle(img, (cols-2, int(pos[3])), 5, (0, 205, 0), 2)
    if draw:
        cv2.imshow('pos img', img)
    return pos


def low_pass_filter(img, filter_size_c=20, filter_size_r=80):

    # 获取图像的傅里叶变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 创建一个遮罩，中心正方形为1，其余全为0
    rows, cols = dft_shift.shape[:2]
    print(rows, cols)
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-filter_size_r:crow+filter_size_r,
         ccol-filter_size_c:ccol+filter_size_c] = 1

    # 应用遮罩并进行逆DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 1, cv2.NORM_MINMAX)

    return img_back


def filter_small_blobs(img, min_contour_area=50):
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 过滤小色块
    large_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    # 在原图像上绘制大色块
    mask = np.zeros_like(img)
    cv2.drawContours(mask, large_contours, -1,
                     (255, 255, 255), thickness=cv2.FILLED)
    # 应用掩膜
    result = cv2.bitwise_and(img, mask)
    return result


def pre_process(gray):
    if len(gray.shape) > 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (21, 21), 13)
    # cv2.imshow('blur', gray)
    gray = cv2.resize(gray, (0, 0), cv2.INTER_NEAREST, fx=0.2, fy=0.2)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 22)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_ERODE, kernel, iterations=2)
    kernel = np.ones((6, 6), np.uint8)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_DILATE, kernel, iterations=1)
    return thresh

detected=False 
def timer_cb():
    global detected
    detected=False

last_pos=0
ending_cnt=0
ending_full=25

def img_process(gray,forward_dis,cb):
    thr_up=0.4
    thr_low=thr_up-0.25
    global detected,last_pos,ending_cnt
    if detected:
        time.sleep(0.016)
        return (None,-1) 
             
    len_gray = len(gray)
    
    gray1 = gray[-int(thr_up *len_gray):-int(thr_low*len_gray), :]#0.25 
    gray2 = gray[-int(0.8*len_gray):-int(0.6 * len_gray), :]
    thresh = pre_process(gray1)
    forward_img = pre_process(gray2)
    cx=int(len(thresh[0])/2) 
    pos = dectect_corners(thresh, draw=False)
    thresh = cv2.Canny(thresh, 100, 200)
            # cv2.imshow('raw_canny', thresh)
    val_dir=-1
    
    if pos[1] != -1:
        last_pos=pos[1]
        # if cx//2<pos[1]<cx*1.5 :
        #     if forward_dis<20:
        #         cb()
        #         return None ,'back'
        img_left = thresh[:, :pos[1]]
        img_right = thresh[:, pos[1]:]
        dx_l = cv2.filter2D(img_left, -1, np.array([[-1, 1]]))
        dy_l = cv2.filter2D(img_left, -1, np.array([[-1], [1]]))
        dx_r = cv2.filter2D(img_right, -1, np.array([[-1, 1]]))
        dy_r = cv2.filter2D(img_right, -1, np.array([[-1], [1]]))
        sum_dx_l = np.sum(dx_l)
        sum_dy_l = np.sum(dy_l)
        sum_dx_r = np.sum(dx_r)
        sum_dy_r = np.sum(dy_r)
        left = sum_dx_l*2.8 < sum_dy_l
        right = sum_dx_r*2.8< sum_dy_r
        if left or right:
            if 8< pos[2] < 22 or 8 < pos[3] < 22: 
                if sum_dy_l > 12000 and sum_dy_r > 12000:
                    d2_x = cv2.filter2D(
                        forward_img, -1, np.array([[-1, 1]]))
                    sum_dx2= np.sum(d2_x)
                    if sum_dx2>3000:
                        print('all')
                        val_dir='all'
                        detected=True
                    else :
                        print('both') 
                        val_dir='both'
                        detected=True
                elif sum_dy_l > 12000:
                    d2_x = cv2.filter2D(
                        forward_img, -1, np.array([[-1, 1]]))
                    sum_dx2= np.sum(d2_x)
                    if sum_dx2>3000:
                        print('left_up')
                        val_dir='left_up'
                        detected=True
                    else :
                        print('left') 
                        val_dir='left'
                        detected=True

                elif sum_dy_r > 12000:
                    d2_x = cv2.filter2D(
                        forward_img, -1, np.array([[-1, 1]]))
                    sum_dx2= np.sum(d2_x)
                    if sum_dx2>3000:
                        print('right_up')
                        val_dir='right_up'
                        detected=True
                    else :
                        print('right') 
                        val_dir='right'
                        detected=True
                if detected:
                    timer=threading.Timer(0.25,timer_cb)
                    timer.start()
        
        
    else:
        pos[1]=last_pos 
    if pos[0]==-1:
            if ending_cnt<ending_full:
                ending_cnt+=1
            else:
                if val_dir==-1 :
                    if forward_dis<24:
                        val_dir='back'
    
    
    gray[-int(thr_up *len_gray):-int(thr_low*len_gray), :]=cv2.resize(thresh,(0,0),fx=5,fy=5) 
    return int(pos[1]/0.2),val_dir
  
            

        
 


if __name__ == "__main__":
    idx = 0
    # from skimage import morphology, draw
    cap = cv2.VideoCapture(0)
    count = 0
    ana_win = []
    while True:
        ret, frame = cap.read()
        if ret == True:
            idx += 1
            if detected:
                time.sleep(0.02)
                continue
            t1 = time.time()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            len_gray = len(gray)
            gray1 = gray[-int(0.46*len_gray):-int(0.21*len_gray), :]
            gray2 = gray[-int(0.8*len_gray):-int(0.6 * len_gray), :]
            thresh = pre_process(gray1)
            forward_img = pre_process(gray2)
            pos = dectect_corners(thresh, draw=False)
            thresh = cv2.Canny(thresh, 100, 200)
            cv2.imshow('raw_canny', thresh)

            if pos[1] != -1:
                img_left = thresh[:, :pos[1]]
                img_right = thresh[:, pos[1]:]
                dx_l = cv2.filter2D(img_left, -1, np.array([[-1, 1]]))
                dy_l = cv2.filter2D(img_left, -1, np.array([[-1], [1]]))
                dx_r = cv2.filter2D(img_right, -1, np.array([[-1, 1]]))
                dy_r = cv2.filter2D(img_right, -1, np.array([[-1], [1]]))
                sum_dx_l = np.sum(dx_l)
                sum_dy_l = np.sum(dy_l)
                sum_dx_r = np.sum(dx_r)
                sum_dy_r = np.sum(dy_r)
                left = sum_dx_l*2.8 < sum_dy_l
                right = sum_dx_r*2.8 < sum_dy_r
                if left or right:
                    if 10 < pos[2] < 19 or 10 < pos[3] < 19:
                        print(idx, left, sum_dx_l, sum_dy_l,
                              right, sum_dx_r, sum_dy_r)
                        if sum_dy_l > 13000 and sum_dy_r > 13000:
                            d2_x = cv2.filter2D(
                                forward_img, -1, np.array([[-1, 1]]))
                            sum_dx2= np.sum(d2_x)
                            if sum_dx2>3000:
                                print('all')
                                detected=True
                            else :
                                print('both') 
                                detected=True
                        elif sum_dy_l > 16000:
                            d2_x = cv2.filter2D(
                                forward_img, -1, np.array([[-1, 1]]))
                            sum_dx2= np.sum(d2_x)
                            if sum_dx2>3000:
                                print('left_up')
                                detected=True
                            else :
                                print('left') 
                                detected=True
                        elif sum_dy_r > 16000:
                            d2_x = cv2.filter2D(
                                forward_img, -1, np.array([[-1, 1]]))
                            sum_dx2= np.sum(d2_x)
                            if sum_dx2>3000:
                                print('right_up')
                                detected=True
                            else :
                                print('right') 
                                detected=True
                        if detected:
                            timer=threading.Timer(0.8,timer_cb)
                            timer.start()
  
            print(f'cost time {(t2-t1)*1000000:.2f}')
            cv2.imshow('forward', forward_img)
            cv2.putText(frame, str(idx), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('frame', frame)
            cv2.imshow('result', thresh)
            # key = cv2.waitKey(2) & 0xFF
            # if key == 27:
            #     break
            # # 如果按下空格键，就暂停
            # elif key == 32:
            #     cv2.waitKey(0)

        else:
            break

    cv2.destroyAllWindows()
