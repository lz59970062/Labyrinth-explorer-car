
import cv2 
import numpy as np 
import time
import threading
 
import sys
import queue

import os

camera_matrix = np.array([[528.6062, 0, 332.2342],
                          [0, 529.8388, 234.6237],
                          [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.array([0.0834, 0.1615, 0, 0, -0.5344], dtype=np.float32) # 更新畸变系数


class image_processer():
    def __init__(self,image_display_width,image_display_height ,show_img=0,cap_id=0)  :
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_display_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_display_height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.img_queue=queue.Queue(maxsize=10)
        self.cap_id=cap_id
        self.show_image=show_img
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.videoWriter = cv2.VideoWriter('{}/video_result.mp4'.format('.'), fourcc, 60,(640,480) )

        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (640, 480), 1, (image_display_width,image_display_height))
        self.start()
        # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        # self.cap.set(cv2.CAP_PROP_EXPOSURE, 0.01)
        # self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.01)
        # self.cap.set(cv2.CAP_PROP_CONTRAST, 0.01)
        # self.cap.set(cv2.CAP_PROP_SATURATION, 0.01)
        # self.cap.set(cv2.CAP_PROP_HUE, 0.01)
        # self.cap.set(cv2.CAP_PROP_GAIN, 0.01)
        # self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.01)
        # self.cap.set(cv2.CAP_PROP_AUTO_WB, 0.01)
        # self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 0.01)
        # self.cap.set(cv2.CAP_PROP_BACKLIGHT, 0.01)
        # self.cap.set(cv2.CAP_PROP_SHARPNESS, 0.01)
    def start(self):
        self.thread=threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                t1 = time.time() 
                if self.img_queue.full():
                    _ = self.img_queue.get()
                self.img_queue.put(frame)
                self.videoWriter.write(frame) 
                if self.show_image:
                    cv2.imshow("cap"+str(self.cap_id),frame)
                    cv2.waitKey(1)
            else:
                print("read image error")
                
            # time.sleep(0.0001)
    def get_image(self):
        return self.img_queue.get()
    
    def stop(self):
        self.is_running = False
        self.thread.join()  # 等待线程结束
        self.videoWriter.release() 
        self.cap.release()  # 关闭摄像头
        while not self.img_queue.empty():  # 清空队列
            self.img_queue.get()


# if __name__ == "__main__":
#     image_processer=image_processer(520,520,0,0)
#     while True:
#         time.sleep(1) 



        