from collections import deque
import time
import numpy as np


class SpeedPlanner:
    # 接受的参数有目前所有节点构成的路径，节点字典，最大速度，最小速度（不能超过40cm/m），机器人对象,now_node 是当前节点的索引,如果有路径从新规划的行为，主要需要传入该参数
    def __init__(
        self, path, nodes, max_speed, min_speed, robot, walk_only=False, now_node=0
    ):
        self.path = path  # path是一个列表，里面是节点的id
        self.nodes = nodes  # nodes是一个字典，里面是节点的id和节点对象的映射
        self.max_speed = max_speed  # 小车运行最大速度
        self.min_speed = min_speed  # 小车运行最小速度
        self.current_speed = min_speed  # 初始化当前速度为最小速度
        self.current_node_index = now_node  # 初始化当前节点索引为0
        self.current_node_id = path[0]  # 初始化当前节点id为路径的第一个节点
        self.current_dir = None  # 初始化当前方向为None
        self.rb = robot  # 初始化机器人对象
        self.special_node = {}  # 初始化特殊节点字典，特殊字典的意义在于，进入特殊字典时，执行特殊操作，操作保存在回调函数中，即字典中的值
        self.baozang_node = []  # 初始化宝藏节点列表
        self.lastx, self.lasty = 0, 0  # 上一次跟新节点时的位置坐标
        self.expect_pos_org = [10, 0]
        self.expect_pos = None
        self.turn_node_id = 0  # 上一次转弯完成后的节点id
        self.next_turn_node_id = 0
        self.walk_only = walk_only  # 是否只是走路，如果是走路，不需要使用里程计和转向信息来判断是否更新节点
        self.straight_dis = 0
        self.planned_sp = 40
        self.last_val_dir = "all"
        self.last_turn = " "
    def restart(self):
        self.expect_pos_org = [10, 0]
        # self.last_val_dir = "all"
        # # self.expect_pos = None
        # self.last_turn = " "
        # self.lastx,self.lasty=0,0 

    def add_special_node(self, node_id, action):
        if not self.walk_only:
            self.special_node[node_id] = action
            # 将特殊节点和回调函数添加到特殊节点字典中，回调的编写有一定要求，1.回调函数的参数必须是self ，self中包含rb，尽量设置成无参函数，和其他需要的参数，2.回调函数必须返回一个布尔值，True表示回调函数执行成功，False表示回调函数执行失败，3.在回调函数中，如果需要更新节点，必须调用self.update_node(node_id)函数，node_id是需要更新的节点id
            # 必须更新节点，以及预期小车在直接巡线状态下会经过的下一个节点，小车在退出特殊节点的回调函数之后会默认进入巡线状态，即特殊节点的执行必须使状态闭环。如果不更新节点，小车会在下一个节点处停止，因为小车会认为自己已经到达了下一个节点，但是实际上小车还没有到达下一个节点，所以需要更新节点，让小车知道自己已经到达下一个节点

    def remove_special_node(self, node_id):  # 删除特殊节点
        if not self.walk_only:
            if node_id in self.special_node:
                del self.special_node[node_id]

    def handle_special_node(self):  # 判断是否进入特殊节点，如果进入，执行回调函数
        if not self.walk_only:
            if self.current_node_id in self.special_node:
                self.special_node[self.current_node_id]()
                self.remove_special_node(self.current_node_id)  # 执行完回调函数后，删除特殊节点，

    def needs_to_turn(
        self, node_index
    ):  # 简单来说，就是判断的节点递进方向是否和当前节点的方向一致，如果一致，就不需要转弯，如果不一致，就需要转弯
        if node_index == len(self.path) - 1:  # last node
            return False

        last_node_id = self.path[node_index - 1]
        last_node = self.nodes[last_node_id]
        next_node_id = self.path[node_index + 1]
        current_node_id = self.path[node_index]
        current_node = self.nodes[current_node_id]
        print(last_node)
        for i in range(len(last_node.available_node)):
            # print(f"[needs_to_turn] last_node_available{last_node.available_node},cur node{current_node_id}")
            if last_node.available_node[i] == current_node_id:
                last_dir = i
                break
        # self.current_node_id = self.path[node_index]

        for i in range(len(current_node.available_node)):
            # print(f"[needs_to_turn] cur_node_available{current_node.available_node},next node{ next_node_id}")
            if current_node.available_node[i] == next_node_id:
                next_dir = i
                break

        if last_dir == next_dir:
            return False
        else:
            return True

    def get_straight_distance(self):
        distance = 0
        node_index = self.current_node_index
        while True:
            if node_index == len(self.path) - 1:  # last node
                break
            if node_index==0:
                continue 
            # print(f"Now path index {node_index} node{self.path[node_index]}")
            if self.needs_to_turn(node_index):  # need turn
                self.next_turn_node_id = self.path[node_index]
                break
            distance += 1
            node_index += 1
        return distance

    def update_speed(
        self,
    ):  # 由于转弯前的的速度目前要严格限制在40cm/s以下，所以，需要通过里程计判断，在进入转向节点后需要吧速度降低到最低速度，然后在转向节点后，再把速度提升到最大规划速度
        if not self.walk_only:
            straight_distance = self.get_straight_distance()  # 获取从当前节点走起可以走几个节点  d
            self.straight_dis = straight_distance  # 记录下来，用于后面的判断
            delta_speed = self.max_speed - self.min_speed  # 最大速度和最小速度的差值
            if straight_distance >= 1:  # Adjust this value as needed.
                self.current_speed = delta_speed * 1/ 4 + self.min_speed
            elif straight_distance >= 2:
                self.current_speed = delta_speed * 1/ 4 + self.min_speed
            elif straight_distance > 3:
                self.current_speed = delta_speed * 2 / 4 + self.min_speed 
            else:
                self.current_speed = self.min_speed
            return self.current_speed

    def update_node(self, node_id):
        if not self.walk_only:
            # 先判断节点关系是否合理
            if node_id not in self.nodes[self.current_node_id].get_available_node():
                print("update_node: error node")
                return
            # when updating the current node, also update the current_node_index
            self.current_node_index += 1
            self.current_node_id = self.path[self.current_node_index]
            if self.current_node_id == self.next_turn_node_id:
                self.rb.set_speed(self.min_speed)
                self.planned_sp = self.min_speed

            self.handle_special_node()

    # 按照上面关于方向的定义，光凭路径的前后左右并不能判断执行，如果直行，需要连续几个节点的递推关系都是同一个方向,
    # 还有，记住，这种方法必须确定初始时陀螺仪正确摆放，且数据正确初始化。
    def try_to_update_node_by_odom(
        self,
    ):  # 使用里程计和转向信息综合判断是否更新节点,循环调用，作为辅助判断，主要判断是否进入节点，用于提前减速，判断进入特殊函数等
        if not self.walk_only:
            dx, dy = self.rb.odom_x - self.lastx, self.rb.odom_y - self.lasty
            self.expect_pos = [self.expect_pos_org[0] + dx, self.expect_pos_org[1] + dy]
            # 为了防止误差累积，每次里程计数据是以上一次转弯完成从后的位置为基准，所以需要记录上一次转弯完成后的位置，由于转向完成是以新进入的节点为基准，位置分布为基本服从二位高斯分布，偏差的绝对值范围大概在左右偏差7厘米走远，前后偏差5厘米左右，考虑打滑，这个估计值可能还要根据实际情况修改。
            # 举个实际的例子，如果小车完成转向，再往前走10厘米左右，就可以判断小车进入了新的节点
            if (
                self.current_node_index == len(self.path) - 1
            ):  # If the current node is the last node in the path
                return
            if self.path[self.current_node_index] != self.current_node_id:
                print("try_to_update_node_by_odom:path wrong")
                self.rb.set_speed(0)
                self.rb.set_flag(4, 10)
                time.sleep(2)
            next_node_id = self.path[self.current_node_index + 1]
            next_node = self.nodes[next_node_id]
            current_node = self.nodes[self.current_node_id]

            dis_cur_node = self.distance2node(self.expect_pos, current_node)
            dis_next_node = self.distance2node(self.expect_pos, next_node)
            dis_cur_next = self.node_distance(current_node, next_node)

            # if dis_cur_next<dis_cur_node or dis_cur_node<dis_next_node: #小车在两个节点之间，距离肯定是两节点之间的距离最大，如果不是，报错
            #   if current_node.id!=0:
            #     print("node distance error : 小车在两个节点之间，距离肯定是两节点之间的距离最大")
            #     self.rb.set_speed(0)

            #     self.rb.set_flag(4,10)
            #     time.sleep(2)
            if dis_next_node < dis_cur_node:
                self.update_node(next_node_id)  # 此处应是唯一跟新节点的位置
                
                self.print_debug(f"node updated now node is {next_node} last node is {current_node} next node is{self.path[self.current_node_index + 1 if self.current_node_index != len(self.path) - 1 else -1]} ") 
                
                # print(
                #     "try_to_update_node_by_odom:now node is ",
                #     next_node,
                #     "last node is",
                #     current_node,
                # )
                self.rb.set_flag(4, 2)
            else:
                # print(f"now opt pos{self.expect_pos},pos_cur{dis_cur_node},next pos{dis_next_node}")
                pass
            # 判断当前节点到下一个节点，应该是哪个方向上数据增量，前后，左右，这个需要按照节点表来判断，因为节点表中记录了节点的位置信息，对应的方向信息 。
            # 这里需要注意的是，如果小车在直接巡线状态下，会经过的下一个节点，这个节点的方向信息是不准确的，因为小车在直接巡线状态下，会经过的下一个节点，小车会认为自己已经到达了下一个节点，但是实际上小车还没有到达下一个节点，所以需要更新节点，让小车知道自己已经到达下一个节点

            # 到上面就单节点情况下成功更新了节点的状态了，当时还有一种情况，就是小车需要不拐弯地连续走过多个节点，上面的代码只能判断小车成功走进了转弯之后的第一个节点
            # 那么在执行上面代码之前，就需要判断连续走的节点个数，为了准确判断，需要多种判据联合使用，可以包括，先使用dx，dy预测一个节点，然后根据当前节点进行判断，如果正确的话，预测出来的结点肯定是当前节点的下一个节点
            # 或者根据纯粹根据当前节点判断，当前节点的下个节点的位置应为哪个区间，然后判断dx，dy，的和这个区间的相符合的程度，如果太过于离谱就报警 。
            # If the robot is closer to the next node than the current node, update the current node

    def node_distance(self, node1, node2):
        return np.sqrt(
            (node1.pos[0] - node2.pos[0]) ** 2 + (node1.pos[1] - node2.pos[1]) ** 2
        )

    def distance2node(self, pos, node):
        return np.sqrt((pos[0] - node.pos[0]) ** 2 + (pos[1] - node.pos[1]) ** 2)
    def get_change_dir(self ) :
        next_node_id = self.path[self.current_node_index + 1]  
        current_node = self.nodes[self.current_node_id]
            # 判断当前节点到下一个节点，应该是哪个方向上数据增量，前后，左右，这个需要按照节点表来判断，因为节点表中记录了节点的位置信息，对应的方向信息 。
            # 这里需要注意的是，如果小车在直接巡线状态下，会经过的下一个节点，这个节点的方向信息是不准确的，因为小车在直接巡线状态下，会经过的下一个节点，小车会认为自己已经到达了下一个节点，但是实际上小车还没有到达下一个节点，所以需要更新节点，让小车知道自己已经到达下一个节点
        change_dir = None
        # dir_list=['forword','backword','left','right']
        for dir in range(len(current_node.available_node)):
            if current_node.available_node[dir] != -1:
                if current_node.available_node[dir] == next_node_id:
                    change_dir = dir
                    break
        return change_dir 

    def after_turn(
        self,
    ):  # 通过循迹小车 的转向来判断是否更新节点，一般弯道设置在节点中心，识别到弯道说明已经刚刚进入节点8里面左右,此函数在识别到节点之后调用
        if not self.walk_only:
            # 先总结从上一个节点运行过来时的运行信息，并跟新节点,现在往往是即将进入一个新的节点
            if (
                self.current_node_index == len(self.path) - 1
            ):  # If the current node is the last node in the path
                return
            # 先更新速度，根据到下一个转向节点的距离，来调整速度
            self.lastx, self.lasty = self.rb.odom_x, self.rb.odom_y  # 记录下来，用于后面的判断
            self.turn_node_id = self.path[self.current_node_index]  # 记录下来，用于后面的判断
            current_node = self.nodes[self.current_node_id]
            if (
                self.current_node_index == len(self.path) - 1
            ):  # If the current node is the last node in the path
                return
            next_node_id = self.path[self.current_node_index + 1]
            # 判断当前节点到下一个节点，应该是哪个方向上数据增量，前后，左右，这个需要按照节点表来判断，因为节点表中记录了节点的位置信息，对应的方向信息 。
            # 这里需要注意的是，如果小车在直接巡线状态下，会经过的下一个节点，这个节点的方向信息是不准确的，因为小车在直接巡线状态下，会经过的下一个节点，小车会认为自己已经到达了下一个节点，但是实际上小车还没有到达下一个节点，所以需要更新节点，让小车知道自己已经到达下一个节点
            change_dir = None
            # dir_list=['forword','backword','left','right']
            for dir in range(len(current_node.available_node)):
                if current_node.available_node[dir] != -1:
                    if current_node.available_node[dir] == next_node_id:
                        change_dir = dir
                        break
            if change_dir is None:
                print("error")  # 如果这样，就必须好好检查节点表的生成过程是否有问题
                return
            if self.last_val_dir == "back":
                turn_error = 10
            else:
                turn_error = 12
            if change_dir == 0:  # 前
                self.expect_pos_org[0],self.expect_pos_org[1] = (
                    current_node.pos[0] + turn_error,
                    current_node.pos[1],
                )
            if change_dir == 1:  # 后
                self.expect_pos_org[0],self.expect_pos_org[1] = (
                    current_node.pos[0] - turn_error,
                    current_node.pos[1],
                )
            if change_dir == 2:
                self.expect_pos_org[0],self.expect_pos_org[1] = (
                    current_node.pos[0],
                    current_node.pos[1] + turn_error,
                )
            if change_dir == 3:
                self.expect_pos_org[0],self.expect_pos_org[1] = (
                    current_node.pos[0],
                    current_node.pos[1] - turn_error,
                )
            # try:
            speed = self.update_speed()  # 这个函数执行后有个直线距离会被跟新，这个直线距离对使用里程计来跟新节点的函数有用
            self.rb.set_speed(speed)
            self.planned_sp = speed
            # except:
            #     self.rb.set_speed(40)
            #     self.planned_sp = 40
            


    def stop_and_report_error(self):
        self.rb.set_speed(0)
        time.sleep(0.001)
        self.rb.set_gyro(0)
        print("before_turn: error check the node table and no valid direction")
        time.sleep(0.01)
        self.rb.set_flag(4, 10)


    def global_to_local(self,yaw_num, change_dir, val_dir):
    # 根据小车yaw代表的旋转信息转换全局坐标系和局部坐标系的关系

      if yaw_num == 0:  # 同方向
        if change_dir == 0:  # 前
            if val_dir in ["all", "right_up", "left_up"]:
                return True
            elif val_dir in ["left", "right"]:
                self.stop_and_report_error()
                return False
        elif change_dir == 1:  # 后
            self.rb.turn_back()
            return True
        elif change_dir == 2:  # 左
            if val_dir in ["all", "left_up", "left", "both"]:
                self.rb.turn_left()
                return True
            elif val_dir in ["right", "right_up", "back"]:
                self.stop_and_report_error()
                return False
        elif change_dir == 3:  # 右
            if val_dir in ["all", "right_up", "right", "both"]:
                self.rb.turn_right()
                return True
            elif val_dir in ["left", "left_up", "back"]:
                self.stop_and_report_error()
                return False

      elif yaw_num == 1 or yaw_num == -3:  # 左方向
        if change_dir == 0:  # 前 -> 右
            if val_dir in ["all", "right_up", "right", "both"]:
                self.rb.turn_right()
                return True
            elif val_dir in ["left", "left_up", "back"]:
                self.stop_and_report_error()
                return False
        elif change_dir == 1:  # 后 -> 左
            if val_dir in ["all", "left_up", "left", "both"]:
                self.rb.turn_left()
                return True
            elif val_dir in ["right", "right_up", "back"]:
                self.stop_and_report_error()
                return False
        elif change_dir == 2:  # 左 -> 前
            if val_dir in ["all", "right_up", "left_up"]:
                return True
            elif val_dir in ["left", "right", "back"]:
                self.stop_and_report_error()
                return False
        elif change_dir == 3:  # 右 -> 后
            self.rb.turn_back()
            return True

      elif yaw_num == 2 or yaw_num == -2:  # 后方向
        if change_dir == 0:  # 前 -> 后
            self.rb.turn_back()
            return True
        elif change_dir == 1:  # 后 -> 前
            if val_dir in ["all", "right_up", "left_up"]:
                return True
            elif val_dir in ["left", "right", "back", "both"]:
                self.stop_and_report_error()
                return False
        elif change_dir == 2:  # 左 -> 右
            if val_dir in ["all", "right_up", "right", "both"]:
                self.rb.turn_right()
                return True
            elif val_dir in ["left", "left_up", "back"]:
                self.stop_and_report_error()
                return False
        elif change_dir == 3:  # 右 -> 左
            if val_dir in ["all", "left_up", "left", "both"]:
                self.rb.turn_left()
                return True
            elif val_dir in ["right", "right_up", "back"]:
                self.stop_and_report_error()
                return False

      elif yaw_num == 3 or yaw_num == -1:  # 右方向
        if change_dir == 0:  # 前 -> 左
            if val_dir in ["all", "left_up", "left", "both"]:
                self.rb.turn_left()
                return True
            elif val_dir in ["right", "right_up", "back"]:
                self.stop_and_report_error()
                return False
        elif change_dir == 1:  # 后 -> 右
            if val_dir in ["all", "right_up", "right", "both"]:
                self.rb.turn_right()
                return True
            elif val_dir in ["left", "left_up", "back"]:
                self.stop_and_report_error()
                return False
        elif change_dir == 2:  # 左 -> 后
            self.rb.turn_back()
            return True
        elif change_dir == 3:  # 右 -> 前
            if val_dir in ["all", "right_up", "left_up"]:
                return True
            elif val_dir in ["left", "right", "back", "both"]:
                self.stop_and_report_error()
                return False

      else:
        print("error yaw")
        self.stop_and_report_error()
        return False

    def before_turn(self, val_dir):
        # 这个函数被用来从节点图中取出下一个节点的方向信息，然后根据方向信息来调整小车的转向，这个函数在识别到节点之后调用
        if self.planned_sp > 50:
            self.rb.set_speed(40)
        if not self.walk_only:
            # if self.current_node_index == len(self.path) - 1:
            #     print("last node ")
            #     return#如果当前节点是最后一个节点，就不需要转向了
            current_node = self.nodes[self.current_node_id]
            next_node_id = self.path[self.current_node_index + 1]
            next_node = self.nodes[next_node_id]
            # dir_list=['forword','backword','left','right']
            change_dir=None 
            for dir in range(len(current_node.available_node)):
                if current_node.available_node[dir] != -1:
                    # print(
                    #     f"cur node {current_node},vai node{current_node.available_node},next id {next_node_id}"
                    # )
                    if current_node.available_node[dir] == next_node_id:
                        change_dir = dir
                        break
            if change_dir is None:
                print("before_turn : error check the node table and no valid direction")
                print(self.path) 
                self.rb.set_speed(0)
                self.rb.set_flag(4, 10)
                time.sleep(2)
                return False
            # 上面判断了下一个节点的方向，下面根据方向来调整小车的转向，但是上面判断的方向是对于整个地图的全局方向，而下面的转向是针对小车的局部方向，所以需要转换一下
            # 依据小车yaw代表着全局坐标和局部坐标的旋转信息，可以通过这个来转换
            # 全局坐标系的x方向是前后，y方向是左右，x正方向代表yaw是0左右，左方向是小车yaw角pi/2，右方向是小车yaw角-pi/2，后方向是小车yaw角+-pi左右
            # 如果小车yaw角是0，那么全局坐标系和局部坐标系是一致的，如果小车yaw角是pi/2，那么全局坐标系的x方向是小车局部坐标系的y方向，全局坐标系的y方向是小车局部坐标系的-x方向
            # 先根据yaw计算旋转方向
            yaw = self.rb.yaw
            # 对0，pi/2，-pi/2，+-pi取最接近的
            yaw_num = round(yaw / (np.pi / 2))
            print(f"{yaw*180/np.pi} turn",yaw_num,change_dir,val_dir)
            
            self.print_debug(f"now node is :{current_node}  yaw :{yaw*180/np.pi} turn, yaw_num :{yaw_num},change_dir: {change_dir},val_dir:{val_dir}")
            
            self.last_val_dir = val_dir
            ret=self.global_to_local(yaw_num, change_dir, val_dir)
            return ret
# 
            # if yaw_num==0 :# with same direction
            #     #然后根据旋转方向来计算全局坐标系和局部坐标系的转换关系
            #     if change_dir==0:#前
            #         if val_dir in ['all','right_up','left_up']:
            #             pass
            #             return True
            #         elif val_dir in ['left','right']:
            #             self.rb.set_speed(0)#如果是左右转弯，就停止
            #             time.sleep(0.001)
            #             self.rb.set_gyro(0)
            #             print("before_turn :error check the node table and no valid direction")
            #             print(f"before_turn :currunt node{current_node} next node is {next_node}")
            #             time.sleep(0.01)
            #             self.rb.set_flag(4,10)
            #             # time.sleep(2)
            #             return False
            #     elif change_dir==1:#后

            #         self.rb.turn_back()
            #         return True
            #     elif change_dir==2:#左
            #         if val_dir in ['all','left_up','left','both']:
            #             self.rb.turn_left()
            #             return True
            #         elif val_dir in ['right','right_up','back']:
            #             self.rb.set_speed(0)
            #             time.sleep(0.001)
            #             self.rb.set_gyro(0)
            #             print("before_turn :error check the node table and no valid direction")
            #             time.sleep(0.01)
            #             self.rb.set_flag(4,10)
            #             # time.sleep(2)
            #             return False
            #     elif change_dir==3:#右
            #         if val_dir in ['all','right_up','right','both']:
            #             self.rb.turn_right()
            #             return True
            #         elif val_dir in ['left','left_up','back']:
            #             self.rb.set_speed(0)
            #             time.sleep(0.001)
            #             self.rb.set_gyro(0)
            #             print("before_turn :error check the node table and no valid direction")
            #             time.sleep(0.01)
            #             self.rb.set_flag(4,10)
            #             # time.sleep(2)
            #             return False
            # elif yaw_num==1 or yaw_num==-3:# with left direction
            #     if change_dir==0:#向前就是向右
            #         if val_dir in ['all','right_up','right','both']:
            #             self.rb.turn_right()
            #             return True
            #         elif val_dir in ['left','left_up','back']:
            #             self.rb.set_speed(0)
            #             time.sleep(0.001)
            #             self.rb.set_gyro(0)
            #             print("before_turn :error check the node table and no valid direction")
            #             time.sleep(0.01)
            #             self.rb.set_flag(4,10)
            #             # time.sleep(2)
            #             return False
            #     elif change_dir==1:#向后就是向左
            #         if val_dir in ['all','left_up','left','both']:
            #             self.rb.turn_left()
            #             return True
            #         elif val_dir in ['right','right_up','back']:
            #             self.rb.set_speed(0)
            #             time.sleep(0.001)
            #             self.rb.set_gyro(0)
            #             print("before_turn :error check the node table and no valid direction")
            #             time.sleep(0.01)
            #             self.rb.set_flag(4,10)
            #             # time.sleep(2)
            #             return False
            #     elif change_dir==2:#向左就是向前
            #         if val_dir in ['all','right_up','left_up']:
            #             pass
            #             return True
            #         elif val_dir in ['left','right','back']:
            #             self.rb.set_speed(0)
            #             time.sleep(0.001)
            #             self.rb.set_gyro(0)
            #             print("before_turn :error check the node table and no valid direction")
            #             time.sleep(0.01)
            #             self.rb.set_flag(4,10)
            #             # time.sleep(2)
            #             return False
            #     elif change_dir==3:#向右就是向后
            #             self.rb.turn_back()
            #             return True
            # elif yaw_num==2 or yaw_num==-2:# with back direction
            #     if change_dir==0:#向前就是向后
            #         self.rb.turn_back()
            #         return True
            #     elif change_dir==1:#向后就是向前
            #         if val_dir in ['all','right_up','left_up']:
            #             pass
            #             return True
            #         elif val_dir in ['left','right','back','both']:
            #             self.rb.set_speed(0)
            #             time.sleep(0.001)
            #             self.rb.set_gyro(0)
            #             print("before_turn :error check the node table and no valid direction should hanve up ")
            #             time.sleep(0.01)
            #             self.rb.set_flag(4,10)
            #             # time.sleep(2)
            #             return False
            #     elif change_dir==2:#向左就是向右
            #         if val_dir in ['all','right_up','right','both']:#['all','right_up','right','both']
            #             self.rb.turn_right()
            #             return True
            #         elif val_dir in ['left','left_up','back']:
            #             self.rb.set_speed(0)
            #             time.sleep(0.001)
            #             self.rb.set_gyro(0)
            #             print(self.current_node_id,self.path[self.current_node_index+1])
            #             print(f"before_turn :error check the node table and no valid direction should have right val dir{val_dir}")
            #             time.sleep(0.01)
            #             self.rb.set_flag(4,10)
            #             # time.sleep(2)
            #             return False
            #     elif change_dir==3:#向右就是向左
            #         if val_dir in ['all','left_up','left','both']:
            #             self.rb.turn_left()
            #             return True
            #         elif val_dir in ['right','right_up','back']:
            #             self.rb.set_speed(0)
            #             time.sleep(0.001)
            #             self.rb.set_gyro(0)
            #             print(f"before_turn :error check the node table and no valid direction should have left val_dir{val_dir}")
            #             time.sleep(0.01)
            #             self.rb.set_flag(4,10)
            #             # time.sleep(2)
            #             return False
            # elif yaw_num==3 or yaw_num==-1:# with right direction
            #     if change_dir==0:#向前就是向左
            #         if val_dir in ['all','left_up','left','both']:
            #             self.rb.turn_left()
            #             return True
            #         elif val_dir in ['right','right_up','back']:
            #             self.rb.set_speed(0)
            #             time.sleep(0.001)
            #             self.rb.set_gyro(0)

            #             print("before_turn :error check the node table and no valid direction")
            #             time.sleep(0.01)
            #             self.rb.set_flag(4,10)
            #             # time.sleep(2)
            #             return False
            #     elif change_dir==1:#向后就是向右
            #         if val_dir in ['all','right_up','right','both']:
            #             self.rb.turn_right()
            #             return True
            #         elif val_dir in ['left','left_up','back']:
            #             self.rb.set_speed(0)
            #             print("before_turn :error check the node table and no valid direction")
            #             time.sleep(0.001)
            #             self.rb.set_gyro(0)
            #             time.sleep(0.001)
            #             self.rb.set_flag(4,10)
            #             # time.sleep(2)
            #             return False
            #     elif change_dir==2:#向左就是向后
            #         self.rb.turn_back()
            #         return True
            #     elif change_dir==3:#向右就是向前
            #         if val_dir in ['all','right_up','left_up']:
            #             pass
            #             return True
            #         elif val_dir in ['left','right','back','both']:
            #             self.rb.set_speed(0)
            #             time.sleep(0.001)
            #             self.rb.set_gyro(0)
            #             print("before_turn :error check the node table and no valid direction")
            #             time.sleep(0.01)
            #             self.rb.set_flag(4,10)
            #             # time.sleep(2)
            #             return False
            # else:
            #     print("error yaw")
            #     self.rb.set_speed(0)
            #     time.sleep(0.01)
            #     self.rb.set_flag(4,10)
            #     time.sleep(2)
            #     return False

        #            if dir =='right':
        #     self.turn_right()
        # elif dir =='left':
        #     self.turn_left()
        # elif dir=="back":
        #     self.turn_back()
        # elif dir=='right_up':
        #     self.turn_right()
        #     pass
        # elif dir=='left_up':
        #     # self.turn_left()
        #     pass
        # elif dir=='both':
        #     self.turn_right()
        # elif dir=='all':
        #     # self.turn_right()
        #     self.turn_right()
        #     pass

    def get_current_direction(self):
        if not self.walk_only:
            current_node = self.nodes[self.current_node_id]
            if self.current_node_id == self.path[-1]:
                return None
            next_node_id = self.path[self.current_node_index + 1]  # change here
            if next_node_id in current_node.get_available_node():
                if current_node.forword == next_node_id:
                    self.current_dir = "forword"
                    return "forward"
                elif current_node.right == next_node_id:
                    self.current_dir = "right"
                    return "right"
                elif current_node.left == next_node_id:
                    self.current_dir = "left"
                    return "left"
                elif current_node.backword == next_node_id:
                    self.current_dir = "backword"
                    return "backword"
            return None

    # def remove_special_node(self, node_id):
    #     if node_id in self.special_node:
    #         del self.special_node[node_id]

    def handle_obstacle(self, obstacle):
        if not self.walk_only:
            # Here you can add code to handle obstacles
            pass

    def replan_path(self, new_path,idx=0):
        if not self.walk_only:
            # Here you can add code to replan the path
            self.path = new_path
            self.current_node_index = idx
            self.current_node_id = self.path[idx]
    def print_debug(self,data):
        try:
            self.rb.control_node.send_to_node("debug_node",data.encode())
        except:
            self.rb.control_node.send_to_node("debug_node",data)
        finally:
            pass 

if __name__ == "__main__":
    # Create a dictionary of your nodes, keyed by ID.
    nodes = {node.id: node for node in your_nodes}

    # Create a speed planner with your path and nodes.
    speed_planner = SpeedPlanner(
        path=[1, 2, 5, 8], nodes=nodes, max_speed=70, min_speed=40
    )

    # Get the speed for the current node.
    current_node_id = 2
    speed = speed_planner.update_speed(current_node_id)
    print(f"Speed at node {current_node_id} is {speed}.")
