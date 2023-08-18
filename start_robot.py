#!/home/orangepi/miniconda3/bin/python
import pygame
import cv2
import numpy as np
import sys
import os
from enum import Enum
import time
import communictate as com
import robot
import threading
import struct


# import baozang
# 初始化pygame和字体
pygame.init()
pygame.font.init()
rb = robot.Robot()
start_time=0 
start_count=False 

def process_fun(data):
         
        # for i in range(data) :
        #     print(i,en=',') 
        # print(' ')
        if(robot.check_sum(data[:-4]) != struct.unpack('I', data[-4:])[0]):
            print(time.time(),'check sum error')
            return
        # rb.control_node=node 
        data_line = struct.unpack('IhhffhhBBBBI', data[4:-4])# fhh
        
        data_line = list(data_line)
        rb.data_win.append(data_line)
        if(len(rb.data_win) > 10):
            rb.data_win.pop(0)  
        rb.update(rb.data_win)

node = com.Node('process_node', 'network', process_fun,ip="192.168.2.1", port=28288)
# node =com.Node ('process_node','serial',process_fun,com='/dev/ttyS0',baudrate=460800/4,timeout=0.5)
rb.control_node=node 

# node = com.Node('process_node', com.get_ip(), 28288, com.process_fun)
# node = com.Node('process_node','serial',process_fun )
message_dict = dict()
message_dict['postion'] = (-10, 20)

# 窗口尺寸和颜色
width, height = 1280, 720
screen = pygame.display.set_mode(
    (width, height), pygame.NOFRAME | pygame.HWSURFACE | pygame.FULLSCREEN)
# pygame.display.set_caption('UI Demo')
background_color = (30, 30, 30)

# 绘图区域尺寸和位置
drawing_area_width = height  # width // 2
drawing_area_height = height
drawing_area_position = (width-height, 0)

# 图像显示窗口尺寸和位置
image_display_width = width-height  # width // 2 - 10
image_display_height = image_display_width   # 保持图像比例
image_display_position = (0, height-image_display_height)


# 状态变量字体
font = pygame.font.Font(None, 32)

# 迷宫背景图全局变量
maze_image = None
map_seted = False


def draw_vehicle_trajectory(screen):
    draw_maze(screen)
    draw_trajectory(screen)

# 绘制迷宫背景图


def draw_maze(screen):
    global maze_image, map_seted
    if rb.map is None:
        screen.blit(maze_image, drawing_area_position)
    else:
        if not map_seted:
            map = rb.map
            map = cv2.rotate(map, cv2.ROTATE_90_CLOCKWISE)
            # map=cv2.flip(rb.map,0)
            map = cv2.resize(
                map, (int(drawing_area_width/480*400), drawing_area_width,))
            maze_img = 255-map
            maze_image = cv2.cvtColor(maze_img, cv2.COLOR_GRAY2BGR)
            maze_image = pygame.surfarray.make_surface(maze_image)
            map_seted = True
        else:
            screen.blit(maze_image, drawing_area_position)


def display_camera_image(screen):
    if rb.img_to_show is None:
        print("fail to get show_img")
        return
    # else :
        # print("img ok")
    
    frame = rb.img_to_show.copy()
    frame = cv2.resize(frame, (image_display_width, image_display_height))
    # frame = np.rot90(frame)
    frame = cv2.flip(frame, 0)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, image_display_position)


def display_status_variables(screen, messege_dict):
    i, j = 0, 0
    c = 0
    for key in messege_dict:
        i = 2+260*(c % 2)
        j += 25*((c+1) % 2)
        c += 1
        f = font.render(
            f'{str(key)}:{str(messege_dict[key])}', True, (255, 255, 255))
        screen.blit(f, (i, j))


def process_data():
    message_dict["has_data"] = com.has_data
    if node.has_data == True:
        screen.fill(background_color)
        rb.control_node=node 
        # rb.set_flag(2,1)
        if rb.flags[1] == 1:
            message_dict["state"] = "NO PID"
        else:
            message_dict["state"] = "NORMAL"

        message_dict['postion'] = (round(rb.odom_x, 1), round(rb.odom_y, 1))
        message_dict['yaw'] = round(rb.yaw*180/np.pi, 3)
        # message_dict['gyro_z'] = round(rb.wz, 3)
        message_dict['vx'] = round(rb.vx, 3)
        message_dict['dis']= (rb.forward_dis,rb.left_dis,rb.right_dis)
        if start_count:
            message_dict['time']=time.time()-start_time
 
        if rb.cur_sp is not None and rb.cur_sp.expect_pos is not None:
            message_dict['opt pos']=(round(rb.cur_sp.expect_pos[0],1),round(rb.cur_sp.expect_pos[1],1))
            message_dict['cur_node_id']=rb.cur_sp.current_node_id 

        # message_dict['left_wheel_speed']=round(rb.left_wheel_speed,4)
        # message_dict['right_wheel_speed']=round(rb.right_wheel_speed,4)
        # message_dict['left_wheel_distance']=round(rb.left_wheel_distance,3)
        # message_dict['right_wheel_distance']=round(rb.right_wheel_distance,3)

        # rb.set_flag(2,101)
        # rb.set_flag(3,99)
        # rb.set_flag(1,1)


fx,fy =1,1 

def pos_trans(x,y):
    if rb.map is not None :
        map_y = int(drawing_area_width/480*400)
        nx,ny =int(drawing_area_position[0]+(x+40)/fx),int(map_y-y/fy)
        return nx,ny 

# 绘制轨迹（您可以根据需要修改此函数）
def draw_trajectory(screen):
    global fx,fy 
    # 在此处添加绘制轨迹的代码
    # 1. 获取轨迹数据
    if rb.cur_sp is None :
        rb.trajectory.append([rb.odom_x, rb.odom_y, rb.yaw])
    else :
        if rb.cur_sp.expect_pos is not None:
            rb.trajectory.append([rb.cur_sp.expect_pos[0], rb.cur_sp.expect_pos[1], rb.yaw])
        else :
            rb.trajectory.append([rb.odom_x, rb.odom_y, rb.yaw])
    # print(rb.trajectory)
    if len(rb.trajectory) > 1500:
        rb.trajectory.pop(0)
    map_y = int(drawing_area_width/480*400)
    # 为显示正常，计算缩放系数
    if rb.map is None:
        fx, fy = 480/drawing_area_width, 400/map_y
    else:
        fx, fy = rb.map.shape[1]/drawing_area_width, rb.map.shape[0]/map_y
    trajectory = []
    for i in range(len(rb.trajectory)):
        trajectory.append([int(drawing_area_position[0]+(rb.trajectory[i]
                          [0]+40)/fx), int(map_y-rb.trajectory[i][1]/fy)])
    
    if rb.baozangs is not None :
        baozang_list=[]
        for i in rb.baozangs:
            baozang_poses = [drawing_area_position[0] +(i[0])/fx,map_y-i[1]/fy ]
            baozang_list.append(baozang_poses) 

        for bz in baozang_list:
            # pass
            pygame.draw.circle(screen,(0,255,0),(int(bz[0]),int(bz[1])),12)
      

    # 2. 绘制轨迹/
    if trajectory is not None:
        for i in range(1, len(trajectory)):
            pygame.draw.line(screen, (255, 0, 0),
                             trajectory[i-1], trajectory[i], 3)
    else:
        pass 
    # 3. 绘制当前位姿
    # 在此处添加绘制当前位姿的代码
    # 1. 获取当前位姿,坐标上进行垂直翻转，因为坐标系不同
    # current_pose = rb.odom_x, rb.odom_y, rb.yaw
    x,y,yaw=rb.trajectory[-1] 
    current_pose = drawing_area_position[0] +(x+40)/fx,  map_y-(y)/fy, yaw

    if rb.total_path!=None :
         
            path=rb.total_path
            # print(path) 
            for j in range(len(path)-1):
                x,y=rb.node_map[path[j]].pos
                x1,y1=pos_trans(x,y)
                x,y=rb.node_map[path[j+1]].pos
                x2,y2=pos_trans(x,y)
                pygame.draw.line(screen,(0,55,200),(x1,y1),(x2,y2),2)
                #写上节点编号
                f = font.render(f'{str(path[j])}', True, (0, 255, 255))
                screen.blit(f, (x1, y1)) 


    # 2. 绘制当前位姿
    if current_pose is not None:
        # 2.1 绘制小车
        # 2.2 绘制方向
        # 2.3 绘制方向角
        pygame.draw.circle(screen, (0, 0, 255), (int(
            current_pose[0]), int(current_pose[1])),6)
        
        pygame.draw.line(screen, (0, 255, 0), (int(current_pose[0]), int(current_pose[1])), (int(
            current_pose[0]+30*np.cos(current_pose[2])), int(current_pose[1]+30*np.sin(current_pose[2]))), 3)
        
        pygame.draw.line(screen, (255, 0, 0), (int(current_pose[0]), int(current_pose[1])), (int(
            current_pose[0]+30*np.cos(current_pose[2]+np.pi/2)), int(current_pose[1]+30*np.sin(current_pose[2]+np.pi/2))), 3)
    else:
        pass

 
class Button:
    def __init__(self, text, x, y, width, height, color, font_color):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.font_color = font_color

    def draw(self, screen, font):
        pygame.draw.rect(screen, self.color,
                         (self.x, self.y, self.width, self.height))
        text_surface = font.render(self.text, True, self.font_color)
        text_rect = text_surface.get_rect(
            center=(self.x + self.width // 2, self.y + self.height // 2))
        screen.blit(text_surface, text_rect)

    def is_clicked(self, pos, event):
        if self.x <= pos[0] <= self.x + self.width and self.y <= pos[1] <= self.y + self.height:
            if event.type == pygame.MOUSEBUTTONDOWN:
                return True
        return False
    def change_color(self,color):
        if color =='red':
            self.color=(250, 50, 50)
        if color =='blue':
            self.color=(50,50,250) 


def main():
    global maze_image

    message_dict['state'] = "Running"
    clock = pygame.time.Clock()
    exit_button = Button("Exit", width - 110, height - 70,
                         100, 70, (200, 50, 50), (255, 255, 255))
    user_button1 = Button("stop", width - 220, height -
                          70, 100, 70, (50, 200, 50), (255, 255, 255))
    user_button2 = Button("nopid", width - 330, height -
                          70, 100, 70, (50, 200, 50), (255, 255, 255))
    user_button3 = Button("User3", width - 440, height -
                          70, 100, 70, (50, 200, 50), (255, 255, 255))
    user_button4 = Button("User4", width - 550, height -
                          70, 100, 70, (50, 200, 50), (255, 255, 255))
    user_button5 = Button("User5", width - 660, height -
                          70, 100, 70, (50, 200, 50), (255, 255, 255))

    # 在main函数中加载迷宫背景图
    maze_image = pygame.image.load(os.path.join(
        '/home/orangepi/Desktop/temtes/maze.jpg'))

    maze_image = pygame.transform.scale(
        maze_image, (drawing_area_width, drawing_area_height))
 
    mileage = 0
    status = 'Running'
    idx = 0
    idx1=0 
    start_turn_time = time.time()
    if com.has_data == True:
        rb.set_flag(2,1) 
    while True:

        t1 = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # rb.img_processer.stop()
                cv2.destroyAllWindows()
                pygame.quit()
                sys.exit()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if user_button1.is_clicked(mouse_pos, event):
                    # cv2.imwrite(f"/home/orangepi/Desktop/temtes/img/img{idx}.jpg",rb.img_to_show)
                    # rb.set_speed(40)
                    rb.set_speed(0)
                    print("User1 button clicked")
                    rb.stopall(idx1)
                    start_count=True
                    start_time=time.time()  
                    idx1= not idx1
                    
                elif user_button2.is_clicked(mouse_pos, event):
                    print("User2 button clicked")
                    rb.img_processer.videoWriter.release() 
                    # rb.set_speed(0)
                    # cv2.imwrite(f"/home/orangepi/Desktop/temtes/img/img{idx}.jpg",rb.img_to_show)
                    rb.set_flag(2, idx % 2)
                    idx += 1
                    if (idx % 2) == 1:
                        rb.line_pid.ITerm = 0

                elif exit_button.is_clicked(mouse_pos, event):
                    # del rb
                    rb.set_speed(0)
                    rb.set_gyro(0)
                    cv2.destroyAllWindows()
                    pygame.quit()
                    sys.exit()

                elif user_button3.is_clicked(mouse_pos, event): 
                    rb.set_flag(4,4)
                    rb.self_restart()
                elif user_button4.is_clicked(mouse_pos, event):
                    if rb.map is None:
                        temp_thread = threading.Thread(target=rb.get_map)
                        temp_thread.start()
                elif user_button5.is_clicked(mouse_pos, event):
                    # print("user 5 is click")
                    # rb.turnback()
                    if rb.team_color is not None :
                        if rb.team_color=='red':
                            rb.team_color='blue'
                        else :
                            rb.team_color='red'
                    else :
                        rb.team_color='red'

                    # rb.set_flag(4,4)
                    # idx+=1
                    # cv2.imwrite(f"/home/orangepi/Desktop/temtes/img/img{idx}.jpg",rb.img_to_show)
                    pass
                    # rb.get_map()

        # screen.fill(background_color)
        process_data()
        draw_vehicle_trajectory(screen)
        display_camera_image(screen)
        display_status_variables(screen, message_dict)
        user_button1.draw(screen, font)
        user_button2.draw(screen, font)
        user_button3.draw(screen, font)
        user_button4.draw(screen, font)
        user_button5.draw(screen, font)
        exit_button.draw(screen, font)
        if rb.team_color is not None:
            user_button5.change_color(rb.team_color) 

        pygame.display.flip()
        
        clock.tick(15)

if __name__ == '__main__':

    main()
