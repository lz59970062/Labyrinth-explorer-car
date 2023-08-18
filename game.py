#/usr/bin/env python3
import pygame
import cv2
import numpy as np
import sys
import os
from enum import Enum
import time 

# 初始化pygame和字体
pygame.init()
pygame.font.init()

# 窗口尺寸和颜色
width, height =1280,720
screen = pygame.display.set_mode((width, height), pygame.NOFRAME|pygame.HWSURFACE)#|pygame.FULLSCREEN
# pygame.display.set_caption('UI Demo')
background_color = (30, 30, 30)

# 绘图区域尺寸和位置
drawing_area_width = height #width // 2
drawing_area_height = height
drawing_area_position = (width-height, 0)

# 图像显示窗口尺寸和位置
image_display_width = width-height#width // 2 - 10
image_display_height = image_display_width   # 保持图像比例
image_display_position = (0,height-image_display_height)

# 摄像头设置
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, image_display_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, image_display_height)

# 状态变量字体
font = pygame.font.Font(None, 24)

# 迷宫背景图全局变量
maze_image = None

def draw_vehicle_trajectory(screen):
    draw_maze(screen)
    draw_trajectory(screen)

# 绘制迷宫背景图
def draw_maze(screen):
    screen.blit(maze_image, drawing_area_position)

def display_camera_image(screen, camera):
    ret, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (image_display_width, image_display_height))
    frame = np.rot90(frame)
    # frame = cv2.flip(frame,1)

    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, image_display_position)

def display_status_variables(screen,messege_dict):
    i,j=0,0
    for key in messege_dict:
         i=5
         j+=25
         f=font.render(f'{str(key)}:{str(messege_dict[key])}', True, (255, 255, 255))
         screen.blit(f, (i,j))
            

# 绘制轨迹（您可以根据需要修改此函数）
def draw_trajectory(screen):
    # 在此处添加绘制轨迹的代码，您可以根据需要自定义轨迹
    
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
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        text_surface = font.render(self.text, True, self.font_color)
        text_rect = text_surface.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        screen.blit(text_surface, text_rect)

    def is_clicked(self, pos, event):
        if self.x <= pos[0] <= self.x + self.width and self.y <= pos[1] <= self.y + self.height:
            if event.type == pygame.MOUSEBUTTONDOWN:
                return True
        return False
    
def main():
    global maze_image
    message_dict=dict()
    message_dict['postion']=(-20,20)
    message_dict['state']="Running"
    clock = pygame.time.Clock()
    user_button1 = Button("User1", width - 240, height - 50, 70, 40, (50, 200, 50), (255, 255, 255))
    user_button2 = Button("User2", width - 160, height - 50, 70, 40, (50, 200, 50), (255, 255, 255))
    exit_button = Button("Exit", width - 80, height - 40, 70, 30, (200, 50, 50), (255, 255, 255))

    # 在main函数中加载迷宫背景图
    maze_image = pygame.image.load(os.path.join('/home/orangepi/Desktop/temtes/text3.jpg'))

    maze_image = pygame.transform.scale(maze_image, (drawing_area_width, drawing_area_height))

    mileage = 0
    status = 'Running'

    while True:
        t1=time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                camera.release()
                cv2.destroyAllWindows()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if user_button1.is_clicked(mouse_pos, event):
                    print("User1 button clicked")
                elif user_button2.is_clicked(mouse_pos, event):
                    print("User2 button clicked")
                elif exit_button.is_clicked(mouse_pos, event):
                    camera.release()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    sys.exit()

        # screen.fill(background_color)

        draw_vehicle_trajectory(screen)
        display_camera_image(screen, camera)
        display_status_variables(screen, message_dict)
        user_button1.draw(screen, font)
        user_button2.draw(screen, font)
        exit_button.draw(screen, font)
        pygame.display.flip()
        clock.tick(30)

if __name__ == '__main__':
    main()