import time
import numpy as np
import image_process as ip
import cv2
import threading
from detectMap import *
from nav import *
from utils import *
from speedplaner import *
import struct

# from treasure_detect import *
import baozang as bz


def cb():
    pass


from image_utilis import *


class Robot:
    def __init__(self) -> None:
        self.time_tick = 0
        self.odom_x = 0
        self.odom_y = 0

        self.yaw = 0
        self.vx = 0
        # self.wz = 0
        # self.left_wheel_speed = 0
        # self.right_wheel_speed = 0
        self.left_wheel_distance = 0
        self.right_wheel_distance = 0
        self.forward_dis = 0
        self.left_dis = 0
        self.right_dis = 0
        self.back_dis = 0
        self.state = 0
        self.control_node = None
        self.getting_map = False
        self.line_pid = PID(P=0.03, I=0.0002, D=0.00, min_error=10)
        self.dis_pid = PID(1, 0.2, 0, 2, 20, 100)
        self.flags = [0, 0, 0, 0]
        self.map = None
        self.baozangs = None
        self.trajectory = []
        self.trajectory_node = []
        self.target_yaw = 0
        self.turing = False
        self.traveled_nodes = []
        self.now_node = None
        self.last_node = None
        self.next_node = None
        self.now_target_node = None
        self.tback = False
        self.node_map = None
        self.plan_path = None
        self.planner = None
        self.cur_sp = None
        self.stop_all = True
        self.data_win = []
        self.now_plan_speed = 40
        self.odom_dx, self.odom_dy = 0, 0
        self.use_planner = True
        self.total_path = None
        self.team_color = None
        self.last_turn_dir = "right"
        self.state_list = [
            self.time_tick,
            self.odom_x,
            self.odom_y,
            self.yaw,
            self.vx,  # self.wz, self.left_wheel_speed,self.right_wheel_speed,
            self.left_wheel_distance,
            self.right_wheel_distance,
            self.forward_dis,
            self.left_dis,
            self.right_dis,
            self.back_dis,
            self.state,
        ]
        self.state_name = [
            "time_tick",
            "odom_x",
            "odom_y",
            "yaw",
            "vx",  # "wz", "left_wheel_speed",  "right_wheel_speed",
            "left_wheel_distance",
            "right_wheel_distance",
            "forward_dis",
            "left_dis",
            "right_dis",
            "back_dis",
            "state",
        ]
        self.offset_yaw = None
        self.img_processer = ip.image_processer(544, 544, 0, 1)
        self.img_to_show = None
        self.lock = threading.Lock()
        self.rb_thread = threading.Thread(target=self.update_exp)
        self.rb_thread.start()

        # self.data_thread = threading.Thread(target=self.update)
        # self.rb_thread.start()

    def update(self, data_win):
        if data_win is []:
            return
        if self.offset_yaw is None:
            if len(data_win) >= 10:
                sum = 0
                for i in range(1, 11):
                    sum += data_win[-i][3]
                self.offset_yaw = sum / 10
        if len(data_win[-1]) != len(self.state_list):
            l1 = len(data_win[-1])
            l2 = len(self.state_list)
            raise ValueError(f"wrang data format one is {l1} one is {l2}")
        self.time_tick = data_win[-1][0]
        self.odom_x = data_win[-1][1]
        self.odom_y = data_win[-1][2]

        if self.offset_yaw is not None:
            self.yaw = (data_win[-1][3] - self.offset_yaw + 3 * np.pi) % (
                np.pi * 2
            ) - np.pi
        else:
            self.yaw = data_win[-1][3]
        self.vx = data_win[-1][4]
        # self.wz = data_win[-1][5]
        # self.left_wheel_speed = data_win[-1][6]
        # self.right_wheel_speed = data_win[-1][7]
        self.left_wheel_distance = data_win[-1][5]
        self.right_wheel_distance = data_win[-1][6]
        self.forward_dis = data_win[-1][7]
        self.left_dis = data_win[-1][8]
        self.right_dis = data_win[-1][9]
        self.back_dis = data_win[-1][10]
        self.state = data_win[-1][11]
        self.state_list = [
            self.time_tick,
            self.odom_x,
            self.odom_y,
            self.yaw,
            self.vx,  # self.wz, self.left_wheel_speed, self.right_wheel_speed,
            self.left_wheel_distance,
            self.right_wheel_distance,
            self.forward_dis,
            self.left_dis,
            self.right_dis,
            self.back_dis,
            self.state,
        ]
        # print(self.state_list)
        # if self.odom_x > 0 and self.odom_y > 0 and self.node_map is not None:
        #     temp_node = self.node_map[locate_node_py_pos(
        #         self.odom_x, self.odom_y)]
        #     if temp_node is not self.now_node:
        #         self.last_node = self.now_node
        #         self.now_node = temp_node
        #         if self.planner is not None:
        #             self.cur_sp = self.planner.get_cur_splanner()  # 路径，节点，速度
        #             self.cur_sp.update_node(temp_node.id)

    def get_data_analyize(self, data_win, item):
        data = np.array(data_win)
        if item not in self.state_name:
            raise ValueError("wrong item name")
        idx = self.state_name.index(item)
        idx_data = data[:, idx]
        print(
            f"get {item} data,info is :std:{np.std(idx_data)},mean:{np.mean(idx_data)},max:{np.max(idx_data)},min:{np.min(idx_data)}"
        )

    def send_cmd(self, cmd):
        if self.control_node is None:
            print("control node not init")
            return
        if self.control_node.mode == "serial":
            self.control_node.send_to_serial(cmd)
        elif self.control_node.mode == "network":
            if len(self.control_node.neighbors) != 0:
                self.control_node.send_to_node("esp_diff", cmd)
                time.sleep(0.002)
            else:
                print("no control node available")

    def set_speed(self, speed):
        cmd = struct.pack("BBi", 0x55, 0x01, int(speed))
        self.send_cmd(cmd)

    def set_gyro(self, angular_z):
        cmd = struct.pack("BBf", 0x55, 0x02, float(angular_z))
        self.send_cmd(cmd)

    def set_distance(self, dis):
        cmd = struct.pack("BBi", 0x55, 0x03, int(dis))
        self.send_cmd(cmd)

    def set_angle(self, angle):
        cmd = struct.pack("BBf", 0x55, 0x04, float(angle))
        self.send_cmd(cmd)

    def set_flag(self, flag, data):
        self.flags[flag - 1] = data
        cmd = struct.pack(
            "BBBBBBBB",
            0x55,
            0x05,
            0,
            0,
            self.flags[0],
            self.flags[1],
            self.flags[2],
            self.flags[3],
        )
        self.send_cmd(cmd)

    def __del__(self):
        self.set_speed(0)
        # self.set_angle(0)
        self.set_distance(0)
        self.set_gyro(0)
        # 关闭线程
        self.img_processer.stop()
        self.rb_thread.join()
        print("robot del")

    def turn_back(self):
        if self.last_turn_dir == "right":
            self.turn_right_back()

        elif self.last_turn_dir == "left":
            self.turn_leftback()

        # now = self.yaw
        # idx=0
        # full=5
        # self.set_gyro(0)
        # self.set_speed(-30)
        # time.sleep(0.3)
        # self.set_speed(1)
        # if now < 0:
        #     except_yaw = now+np.pi
        #     while self.yaw<except_yaw-0.2:
        #         if idx<full:
        #             idx+=1
        #         self.set_gyro(2.8*idx/full)
        #         time.sleep(0.01)

        # if now >= 0:
        #     except_yaw = now-np.pi
        #     while self.yaw >except_yaw+0.2:
        #         if idx<full:
        #             idx+=1
        #         self.set_gyro(-2.8*idx/full)
        #         time.sleep(0.01)
        # self.set_gyro(0)
        # self.set_flag(3,99)
        # if not self.use_planner:
        #     self.set_speed(40)

    def turn_back_s(self):
        self.set_speed(0)
        self.set_gyro(0)
        time.sleep(0.1)
        self.set_speed(0)
        self.set_gyro(0)
        self.turn_back()

    def turn_leftback(self):
        except_yaw = self.yaw + np.pi
        if except_yaw > np.pi:  # ensure yaw is within -pi to pi
            except_yaw -= 2 * np.pi
        idx = 0
        full = 5
        while abs(except_yaw - self.yaw) > 0.3:
            if idx < full:
                idx += 1
            if self.cur_sp is not None:
                self.set_gyro(4.6 * (idx / full) * (self.cur_sp.planned_sp / 50))
            else:
                self.set_gyro(3.5 * idx / full)
            time.sleep(0.01)
        self.set_gyro(0)
        self.set_flag(3, 99)
        self.last_turn_dir = "left"

    def turn_left(self):
        except_yaw = self.yaw + np.pi / 2
        if except_yaw > np.pi:  # ensure yaw is within -pi to pi
            except_yaw -= 2 * np.pi
        idx = 0
        full = 5
        while abs(except_yaw - self.yaw) > 0.3:
            if idx < full:
                idx += 1
            if self.cur_sp is not None:
                self.set_gyro(4.6 * (idx / full) * (self.cur_sp.planned_sp / 50))
            else:
                self.set_gyro(3.5 * idx / full)
            time.sleep(0.01)
        self.set_gyro(0)
        self.set_flag(3, 99)
        self.last_turn_dir = "left"

    def turn_right(self):
        except_yaw = self.yaw - np.pi / 2  # Subtract pi/2 for a left turn
        if except_yaw < -np.pi:  # ensure yaw is within -pi to pi
            except_yaw += 2 * np.pi
        idx = 0
        full = 5
        while abs(except_yaw - self.yaw) > 0.3:
            if idx < full:
                idx += 1

            if self.cur_sp is not None:
                # self.cur_sp.print_debug(f"here idx is {idx} w {-4.6*(idx/full)*(self.cur_sp.planned_sp/50)} :")
                self.set_gyro(-4.6 * (idx / full) * (self.cur_sp.planned_sp / 50))
            else:
                self.set_gyro(-3.5 * idx / full)  # Apply negative gyro to turn left
            time.sleep(0.01)
        self.set_gyro(0)
        self.set_flag(3, 99)
        self.last_turn_dir = "right"

    def turn_right_back(self):
        except_yaw = self.yaw - np.pi  # Subtract pi/2 for a left turn
        if except_yaw < -np.pi:  # ensure yaw is within -pi to pi
            except_yaw += 2 * np.pi
        idx = 0
        full = 5
        while abs(except_yaw - self.yaw) > 0.3:
            if idx < full:
                idx += 1

            if self.cur_sp is not None:
                # self.cur_sp.print_debug(f"here idx is {idx} w {-4.6*(idx/full)*(self.cur_sp.planned_sp/50)} :")
                self.set_gyro(-4.6 * (idx / full) * (self.cur_sp.planned_sp / 50))
            else:
                self.set_gyro(-3.5 * idx / full)  # Apply negative gyro to turn left
            time.sleep(0.01)
        self.set_gyro(0)
        self.set_flag(3, 99)
        self.last_turn_dir = "right"

    def drive_back(self, dis):
        except_dis = (self.right_wheel_distance + self.left_wheel_distance) / 2 - dis
        self.set_gyro(0)
        ti = time.time()
        while (self.right_wheel_distance + self.left_wheel_distance) / 2 > except_dis:
            time.sleep(0.01)
            self.set_speed(-20)
            if time.time() - ti > 2.5:
                break
        self.set_speed(0)
        self.set_flag(3, 99)

    def get_current_node_id(self):
        return self.now_node.id

    def update_exp(self):
        last_try_time = time.time()
        fps_list = []
        fail_idx = 0
        stop_idx = 0
        while True:
            try:
                if self.getting_map == True:  # or self.planner is None
                    time.sleep(1)
                    continue
                if self.stop_all:
                    self.set_speed(0)
                    img = self.img_processer.img_queue.get()
                    t1 = time.time()
                    ret = bz.baozang_tetect(img.copy(), "red")
                    t2 = time.time()
                    fps_list.append(1 / (t2 - t1))

                    if len(fps_list) > 32:
                        fps_list.pop(0)
                        ret += f"mean is {np.mean(fps_list)}"
                    # print(ret)
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    cv2.putText(img, ret, (40, 40), font, 2, (0, 0, 255), 2)

                    self.img_to_show = img

                    time.sleep(0.01)
                    self.set_gyro(0)
                    time.sleep(0.01)

                    continue
                # if not self.stop_all  and  self.self_check():
                #     # self.stop_all=not self.stop_all
                #     img = self.img_processer.img_queue.get()
                #     font=cv2.FONT_HERSHEY_SIMPLEX
                #     cv2.putText(img,"CHECK_NOT PASS",(40,40),font,2,(0,0,255),2)
                #     time.sleep(0.5)
                #     continue

                img = self.img_processer.img_queue.get()
                # *poses, grayimg = get_line_cross(img)
                self.img_to_show = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                pos, val_dir = img_process(self.img_to_show, self.forward_dis, cb)
                if pos is not None and val_dir == -1:
                    self.line_pid.update(pos - len(self.img_to_show[0]) / 2)
                    self.set_gyro(self.line_pid.output)
                if self.forward_dis < 12:
                    self.set_speed(0)
                    self.set_gyro(0)
                    time.sleep(1)
                    stop_idx += 1
                    print("obscte decteced !")
                    if stop_idx > 3:
                        self.drive_back(5)
                    continue

                if val_dir != -1:
                    # print(val_dir)
                    self.set_flag(4, 4)
                    if not self.use_planner:
                        self.turn_cb(val_dir)
                    else:
                        ret = self.cur_sp.before_turn(val_dir)
                        self.cur_sp.print_debug(f"before turn {ret}")
                        if ret:
                            fail_idx = 0
                            self.cur_sp.after_turn()
                        else:
                            fail_idx += 1
                            self.drive_back(2 * fail_idx + 2)
                            self.set_speed(40)
                            time.sleep(0.3)
                            self.set_gyro(0)
                            if fail_idx >= 5:
                                fail_idx = 10
                    # self.set_flag(4,4)

                if time.time() - last_try_time > 0.02:
                    last_try_time = time.time()

                    if pos is not None:
                        if (pos - len(self.img_to_show[0]) / 2) > 130:
                            self.set_speed(42)
                        else:
                            if not self.use_planner:
                                self.set_speed(40)
                            else:
                                if self.cur_sp is not None:
                                    self.set_speed(self.cur_sp.planned_sp)
                    if self.use_planner:
                        self.cur_sp.try_to_update_node_by_odom()
                if self.tback == True:
                    self.turn_back()
                    self.tback = False
                if (
                    self.left_dis == 255
                    and self.right_dis == 255
                    and self.forward_dis > 190
                ):
                    self.set_speed(0)
                    time.sleep(0.01)
                    self.set_gyro(0)
            except Exception as e:
                print(str(e))
                if self.cur_sp is not None:
                    self.cur_sp.print_debug(str(e))

    def stop(self):
        self.set_speed(0)
        time.sleep(0.01)
        self.set_gyro(0)
        time.sleep(5)
        self.set_speed(20)

    def get_map(self):
        while True:
            self.getting_map = True
            image = self.img_processer.img_queue.get()

            # Detect QR code locator markers
            locator_markers = detect_locator_markers(image)
            # img = image.copy()
            for locator_marker in locator_markers:
                cv2.drawContours(image, [locator_marker], -1, (0, 255, 0), 2)
            cv2.imshow("QR Code Locator Marker Detection", image)
            cv2.waitKey(3)
            if len(locator_markers) != 4:
                print(len(locator_markers))
                print("QR code not detected")
                continue

            # Extract and rescale the ROI
            roi = extract_and_rescale_roi(image, locator_markers)
            if isinstance(roi, type(None)):
                print("wrong marks")
                continue

            # cv2.imshow("roi 1 ", roi)

            # Detect circles in the ROI
            self.lock.acquire()
            roi, self.team_color = roi_proc(roi)
            if self.team_color is None:
                self.lock.release()
                continue

            cv2.imwrite("roi.jpg", roi)
            circles = detect_circles(roi)
            self.baozangs = [circle for circle in circles if 45 < circle[0] < 435]
            print(circles)
            if len(self.baozangs) < 8:
                continue
            # Draw circles on the ROI
            # for circle in circles :
            #  cv2.circle(roi, (circle[0], circle[1]), circle[2]+4, (208, 208, 208), -1)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            binary_image = cv2.adaptiveThreshold(
                roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 18
            )
            for circle in circles:
                cv2.circle(
                    binary_image, (circle[0], circle[1]), circle[2] + 4, (0, 0, 0), -1
                )
            # cv2.imshow("ROI with Circle Detection",binary_image)
            # bordered_img = cv2.copyMakeBorder(
            #     binary_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            binary = fanggetu(binary_image)

            # cv2.line(binary, (0, 20), (100, 20), (155, 155, 155))
            self.map = binary.copy()
            self.lock.release()
            # Show the ROI
            # cv2.imshow("final", bordered_img)
            # cv2.imshow("final2", binary)
            self.baozang_nodes = []
            for i in range(len(self.baozangs)):
                x, y = self.baozangs[i][0] - 40, self.baozangs[i][1]
                self.baozang_nodes.append(x // 40 + y // 40 * 10)

            self.baozang_nodes = set(self.baozang_nodes)
            self.baozang_nodes = list(self.baozang_nodes)
            cv2.imwrite("bin.jpg", binary)
            nodes = get_node(
                binary,
                L,
                num_rows,
                num_cols,
                offsetx=40,
                offsety=2,
                img_width=480,
                img_height=404,
            )
            nodes = [node(i, nodes[i]) for i in nodes.keys()]
            self.node_map = nodes
            self.plan_path = path_planning(0, 99, self.baozang_nodes, nodes)
            self.getting_map = False

            if self.plan_path is not None:
                total_path = []
                for i in range(len(self.plan_path) - 1):
                    path = shortest_path(
                        self.plan_path[i], self.plan_path[i + 1], self.node_map
                    )
                    # print(path)
                    for j in range(len(path) - 1):
                        # x,y=self.node_map[path[j]].pos
                        total_path.append(path[j])

                if total_path[-1] == 98:
                    total_path.append(99)
                print(total_path)
                self.total_path = total_path
            self.cur_sp = SpeedPlanner(
                total_path, self.node_map, 120, 38, self, self.use_planner != True
            )  #
            for i in self.baozang_nodes:
                self.cur_sp.add_special_node(i, self.reg_bz_and_replan)
            self.getting_map = False
            cv2.waitKey(6000)
            cv2.destroyAllWindows()
            return

    def turn_cb(self, dir):
        if dir == "right":
            self.turn_right()
        elif dir == "left":
            self.turn_left()
        elif dir == "back":
            self.turn_back()
        elif dir == "right_up":
            self.turn_right()
            pass
        elif dir == "left_up":
            # self.turn_left()
            pass
        elif dir == "both":
            self.turn_right()
        elif dir == "all":
            # self.turn_right()
            self.turn_right()
            pass

    def turnback(self):
        self.tback = True

    def stopall(self, stop):
        if stop:
            self.stop_all = True
        else:
            self.stop_all = False

    def reg_bz_and_replan(self):
        # self.set_speed(20)
        self.set_gyro(0)
        self.set_speed(0)
        img = self.img_processer.img_queue.get()
        ret = bz.baozang_tetect(img, self.team_color)
        if len(ret) > 1 and ret[0] == "i":
            pass
        else:
            for i in range(18):
                if self.forward_dis > 33:
                    self.set_speed(15)
                    time.sleep(0.1)
                elif self.forward_dis < 25:
                    self.set_speed(-18)
                    time.sleep(0.1)
                time.sleep(0.04)
                # self.dis_pid.setpoint=22
                # self.dis_pid.update(self.forward_dis)
                # self.set_speed(self.dis_pid.output)
                # time.sleep(0.05)
                # if abs(self.forward_dis-24)<2:
                #     self.set_speed(0)
                #     break

        self.set_speed(0)
        self.set_gyro(0)
        for idx in range(3):
            img = self.img_processer.img_queue.get()
            ret = bz.baozang_tetect(img, self.team_color)
            print(
                f"\n\nnow node id is {self.cur_sp.current_node_id},ret is                                 {ret}\n\n"
            )
            self.cur_sp.print_debug(
                f"\n\nfor the {idx} now node id is {self.cur_sp.current_node_id},ret is {ret}\n\n"
            )
            if len(ret) < 3:
                self.set_flag(4, 15)
                if len(self.baozang_nodes) > 3:
                    self.baozang_nodes = set(
                        reduce_node(
                            self.baozang_nodes,
                            self.node_map,
                            self.cur_sp.current_node_id,
                            ret,
                        )
                    )
                    with open('baozang.txt','w') as f :
                        f.write(str(self.baozang_nodes))
                    self.baozang_nodes = list(self.baozang_nodes)
                    self.plan_path = path_planning(
                        self.cur_sp.current_node_id,
                        99,
                        self.baozang_nodes,
                        self.node_map,
                    )
                    if self.plan_path is not None:
                        total_path = []
                        for i in range(len(self.plan_path) - 1):
                            path = shortest_path(
                                self.plan_path[i], self.plan_path[i + 1], self.node_map
                            )
                            # print(path)
                            for j in range(len(path) - 1):
                                # x,y=self.node_map[path[j]].pos
                                total_path.append(path[j])
                        if total_path[-1] == 98:
                            total_path.append(99)
                        print(total_path)
                        if ret == "sp":
                            idx = 0
                        else:
                            idx = 1
                        self.cur_sp.replan_path(total_path, idx)
                        self.total_path = total_path
                # change_dir = self.cur_sp.get_change_dir()
                # current_node = self.cur_sp.nodes[self.cur_sp.current_node_id]
                if ret == "sp":
                    # current_node = self.cur_sp.nodes[self.cur_sp.current_node_id]
                    # turn_error = 5
                    now_node = self.node_map[self.cur_sp.current_node_id]
                    if len(now_node.available_node) == 2:
                        for i in now_node.abailable_node:
                            pass
                    self.turn_back_s()
                    self.drive_back(25)

                    # if change_dir == 0:
                    #     self.cur_sp.expect_pos_org[0],self.cur_sp.expect_pos_org[1] = (
                    #         current_node.pos[0] + turn_error,
                    #         current_node.pos[1],
                    #     )
                    # if change_dir == 1:  # 后
                    #     self.cur_sp.expect_pos_org[0],self.cur_sp.expect_pos_org[1] = (
                    #         current_node.pos[0] - turn_error,
                    #         current_node.pos[1],
                    #     )
                    # if change_dir == 2:
                    #     self.cur_sp.expect_pos_org[0],self.cur_sp.expect_pos_org[1] = (
                    #         current_node.pos[0],
                    #         current_node.pos[1] + turn_error,
                    #     )
                    # if change_dir == 3:
                    #     self.cur_sp.expect_pos_org[0],self.cur_sp.expect_pos_org[1] = (
                    #         current_node.pos[0],
                    #         current_node.pos[1] - turn_error,
                    #     )
                else:
                    # current_node = self.cur_sp.nodes[self.cur_sp.path[self.cur_sp.current_node_index + 1]]
                    # self.cur_sp.update_node(self.cur_sp.path[self.current_node_index + 1])
                    if self.cur_sp is not None:
                        self.cur_sp.print_debug(
                            f"after turn now node is {self.cur_sp.path[self.cur_sp.current_node_index + 1]}"
                        )
                    else:
                        print(
                            f"after turn now node is {self.cur_sp.path[self.cur_sp.current_node_index + 1]}"
                        )
                    turn_error = 20
                    self.turn_back_s()
                    # if change_dir == 0:
                    #     self.cur_sp.expect_pos_org[0],self.cur_sp.expect_pos_org[1] = (
                    #         current_node.pos[0] + turn_error,
                    #         current_node.pos[1],
                    #     )
                    # if change_dir == 1:  # 后
                    #     self.cur_sp.expect_pos_org[0],self.cur_sp.expect_pos_org[1] = (
                    #         current_node.pos[0] - turn_error,
                    #         current_node.pos[1],
                    #     )
                    # if change_dir == 2:
                    #     self.cur_sp.expect_pos_org[0],self.cur_sp.expect_pos_org[1] = (
                    #         current_node.pos[0],
                    #         current_node.pos[1] + turn_error,
                    #     )
                    # if change_dir == 3:
                    #     self.cur_sp.expect_pos_org[0],self.cur_sp.expect_pos_org[1] = (
                    #         current_node.pos[0],
                    #         current_node.pos[1] - turn_error,
                    #     )

                return True
            else:
                cv2.imwrite("./error_" + str(time.time())[3:] + ".jpg", img)
                time.sleep(0.05)
        self.turn_back_s()

    def self_check(self):
        if np.abs(self.yaw) > 10:
            return False
        if self.control_node is None:
            return False
        if self.team_color is not None and (
            self.team_color == "red" or self.team_color == "blue"
        ):
            return False
        if self.node_map is None:
            return False
        if self.baozang_nodes is None:
            return False
        return True

    def self_restart(self):
        roi = cv2.imread("roi.jpg")
        self.lock.acquire() 
        circles = detect_circles(roi)
        self.baozangs = [circle for circle in circles if 45 < circle[0] < 435] 
        # Draw circles on the ROI
        # for circle in circles :
        #  cv2.circle(roi, (circle[0], circle[1]), circle[2]+4, (208, 208, 208), -1)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        binary_image = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 18
        )
        for circle in circles:
            cv2.circle(
                binary_image, (circle[0], circle[1]), circle[2] + 4, (0, 0, 0), -1
            )
        # cv2.imshow("ROI with Circle Detection",binary_image)
        # bordered_img = cv2.copyMakeBorder(
        #     binary_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        binary = fanggetu(binary_image)

        # cv2.line(binary, (0, 20), (100, 20), (155, 155, 155))
        self.map = binary.copy()
        self.lock.release()
        # Show the ROI
        # cv2.imshow("final", bordered_img)
        # cv2.imshow("final2", binary)
        self.baozang_nodes = []
        for i in range(len(self.baozangs)):
            x, y = self.baozangs[i][0] - 40, self.baozangs[i][1]
            self.baozang_nodes.append(x // 40 + y // 40 * 10)

        self.baozang_nodes = set(self.baozang_nodes)
        self.baozang_nodes = list(self.baozang_nodes)
        # cv2.imwrite("bin.jpg", binary)
        nodes = get_node(
            binary,
            L,
            num_rows,
            num_cols,
            offsetx=40,
            offsety=2,
            img_width=480,
            img_height=404,
        )
        nodes = [node(i, nodes[i]) for i in nodes.keys()]
        self.node_map = nodes
        with open('baozang.txt','r') as f :
            strs=f.read() 
            self.baozang_nodes=list(eval(strs)) 
        self.plan_path = path_planning(0, 99, self.baozang_nodes, nodes)
        self.getting_map = False

        if self.plan_path is not None:
            total_path = []
            for i in range(len(self.plan_path) - 1):
                path = shortest_path(
                    self.plan_path[i], self.plan_path[i + 1], self.node_map
                )
                # print(path)
                for j in range(len(path) - 1):
                    # x,y=self.node_map[path[j]].pos
                    total_path.append(path[j])

            if total_path[-1] == 98:
                total_path.append(99)
            print(total_path)
            self.total_path = total_path
        self.cur_sp = SpeedPlanner(
            total_path, self.node_map, 120, 38, self, self.use_planner != True
        )  #
        
        for i in self.baozang_nodes:
            self.cur_sp.add_special_node(i, self.reg_bz_and_replan)
        self.getting_map = False
        cv2.waitKey(6000) 
