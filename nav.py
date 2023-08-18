# #!/home/orangepi/miniconda3/bin/python
# import cv2
# import numpy as np

# from sko.SA import SA_TSP
# from time import perf_counter
# from collections import deque


# # 定义方格纸的行数和列数
# L = 400
# num_rows = 10
# num_cols = 10


# def get_node(gray_img, L, num_rows, num_cols, offsetx=2, offsety=2, img_width=484, img_height=404):
#     if len(gray_img.shape) > 2:
#         gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)

#     gray_img2 = gray_img.copy()
#     # 创建一个字典来存储节点
#     nodes = {}

#     def constrain(x, low, max):
#         if x < low:
#             return low
#         elif x > max:
#             return max
#         else:
#             return x

#     def valid(x, low, max):
#         if x < low:
#             return -1
#         elif x > max:
#             return -1
#         else:
#             return x

#     def has_wall(map, p1, p2):  # p1,p2为两个位置坐标
#         # 坐标转换成数组索引
#         p1 += np.array([offsetx, offsety], dtype='int')
#         p2 += np.array([offsetx, offsety], dtype='int')
#         p1[0], p1[1] = constrain(
#             p1[0], 0, img_width), constrain(p1[1], 0, img_width)
#         p2[0], p2[1] = constrain(
#             p2[0], 0, img_height), constrain(p2[1], 0, img_height)
#         if p1[0] == p2[0] and p1[1] != p2[1]:
#             if p1[1] > p2[1]:
#                 temp = p1[1]
#                 p1[1] = p2[1]
#                 p2[1] = temp
#             area = map[p1[1]:p2[1], p1[0]]
#             sum = np.sum(area)
#             if sum > 1000:
#                 return True
#             else:
#                 return False

#         elif p1[0] != p2[0] and p1[1] == p2[1]:
#             if p1[0] > p2[0]:
#                 temp = p1[0]
#                 p1[0] = p2[0]
#                 p2[0] = temp
#             area = map[p1[1], p1[0]:p2[0]]
#             sum = np.sum(area)

#             if sum > 1000:
#                 return True
#             else:
#                 return False


# # 循环遍历所有的方格，以数字方式为每个节点命名，并将其存储到字典中
#     for row in range(num_rows):
#         for col in range(num_cols):
#             node_name = row*num_cols + col
#             cv2.putText(gray_img2, str(node_name), (col*L//10+15, row*L//10+23),
#                         (cv2.FONT_HERSHEY_SIMPLEX), 0.5, (200, 200, 200), 1, cv2.LINE_AA)
#             y = row * L//10 + L//20
#             x = col * L//10 + L//20

#             node_center = np.array([x, y])

#             qian = has_wall(gray_img, node_center.copy(),
#                             node_center+np.array([L//10, 0]))
#             hou = has_wall(gray_img, node_center.copy(),
#                            node_center+np.array([-L//10, 0]))
#             zuo = has_wall(gray_img, node_center.copy(),
#                            node_center+np.array([0, L//10]))
#             you = has_wall(gray_img, node_center.copy(),
#                            node_center+np.array([0, -L//10]))

#             if qian == 1:
#                 qianmian = -1
#             else:
#                 qianmian = row*num_cols + col+1
#             if hou == 1:
#                 houmian = -1
#             else:
#                 houmian = row*num_cols + col - 1
#             if zuo == 1:
#                 zuomian = -1
#             else:
#                 zuomian = (row+1)*num_cols + col
#             if you == 1:
#                 youmian = -1
#             else:
#                 youmian = (row-1)*num_cols + col

#             neibor = [qianmian, houmian, zuomian, youmian]
#             nei = [valid(i, 0, num_rows*num_cols-1) for i in neibor]
#             nodes[node_name] = node_center, [
#                 not qian, not hou, not zuo, not you], nei
#     # 打印出节点字典
#     cv2.imwrite("map_with_node.jpg", gray_img2)
#     return nodes


# class node():
#     def __init__(self, id, data):
#         pos, dir, available_node = data
#         self.id = id
#         self.pos = pos
#         self.up = dir[0]
#         self.forword = dir[1]
#         self.left = dir[2]
#         self.right = dir[3]
#         self.available_node = [i for i in available_node if i != -1]
#         if available_node[0] != -1:
#             self.forword_node = available_node[0]
#         else:
#             self.forword_node = None
#         if available_node[1] != -1:
#             self.backword_node = available_node[1]
#         else:
#             self.backword_node = None
#         if available_node[2] != -1:
#             self.left_node = available_node[2]
#         else:
#             self.left_node = None
#         if available_node[3] != -1:
#             self.right_node = available_node[3]
#         else:
#             self.right_node = None


#     def __str__(self):
#         return f"[id:{str(self.id)},pos:{str(self.pos)}]"

#     def __repr__(self):
#         return f"[id:{str(self.id)},pos:{str(self.pos)}]"

#     def __eq__(self, other):
#         return self.id == other.id

#     def get_available_node(self):
#         return self.available_node

#     def get_pos(self):
#         return self.pos


# np.random.seed(0)
# # for i,node in enumerate(nodes):
# #     print(i,node.get_available_node())


# def shortest_path(start, end, nodes):
#     queue = deque([(start, [start])])
#     visited = set([start])
#     while queue:
#         node, path = queue.popleft()
#         if node == end:
#             return path
#         # print(node,path)
#         for next_node in nodes[node].get_available_node():
#             if next_node not in visited:
#                 queue.append((next_node, path + [next_node]))
#                 visited.add(next_node)
#     return None

# # path = shortest_path(start_node, end_node, nodes)


# nodes = None


# def path_len(start_node, end_node, nodes):
#     path = shortest_path(start_node, end_node, nodes)
#     if path is None:
#         print("No path found")
#         return np.nan
#     else:

#         return len(path)


# def path_mat():
#     mat = np.zeros((len(nodes), len(nodes)))
#     for i in range(len(nodes)):
#         for j in range(len(nodes)):
#             mat[i, j] = path_len(i, j, nodes)
#     return mat


# def caculate_dis_mat(node_list, all_nodes):
#     l = len(node_list)
#     mat = np.zeros((l, l))
#     for i in range(l):
#         for j in range(l):
#             mat[i, j] = path_len(node_list[i], node_list[j], all_nodes)
#     return mat


# def cal_total_distance(routine, distance_matrix):
#     num_points, = routine.shape
#     sum1 = 0
#     sum1 += distance_matrix[-2, routine[0]]
#     sum1 += distance_matrix[routine[-1], -1]
#     return sum1+sum([distance_matrix[routine[i], routine[(i + 1)]] for i in range(0, num_points - 1)])

# # 退火算法TSP 问题


# def path_planning(start_node, end_node, through_nodes, nodes):
#     path_nodes = np.array(through_nodes+[start_node] + [end_node])
#     print("path nodes", path_nodes)
#     print(path_nodes)
#     distance_matrix = caculate_dis_mat(path_nodes, nodes)
#     points_coordinate = np.array([nodes[i].get_pos() for i in path_nodes])
#     num_points = points_coordinate.shape[0]
#     print("num_points", num_points)

#     start = perf_counter()
#     # 执行模拟退火(SA)算法
#     sa_tsp = SA_TSP(func=lambda x: cal_total_distance(x, distance_matrix), x0=range(
#         num_points-2), T_max=1, T_min=0.2, L=2 * num_points)  # 调用工具箱
#     # 结果输出1
#     best_points, best_distance = sa_tsp.run()
#     print("运行时间是: {:.5f}s".format(perf_counter()-start))  # 计时结束
#     result_cur_best = [path_nodes[i] for i in best_points]
#     print("最优路线：", result_cur_best)
#     print("最优值：", cal_total_distance(best_points, distance_matrix))  # 数据还原
#     return [start_node]+result_cur_best+[end_node]


# def locate_node_py_pos(x, y):
#     x /= 40
#     y /= 40
#     x = int(x)
#     y = int(y)
#     return y*10+x


# # path=path_planning(0, 19, [12,49,18,56,10,30,27,90], nodes)
# # print(path)
# def pos_trans(x, y):

#     return x+40, y+2


# if __name__ == '__main__':
#     # 读取二值图像
#     img = cv2.imread('bn.jpg')
#     nodes = get_node(img, L, num_rows, num_cols,
#                      offsetx=40, offsety=2, img_width=480, img_height=404)
#     nodes = [node(i, nodes[i]) for i in nodes.keys()]
#     node_map = nodes
#     baozang_nodes = [18, 24, 30, 48, 51, 69, 81, 85]
#     plan_path = path_planning(
#         0, 99, baozang_nodes, nodes)
#     if plan_path is not None:
#         for i in range(len(plan_path)-1):
#             path = shortest_path(plan_path[i], plan_path[i+1], node_map)
#             print(path)
#             for j in range(len(path)-1):
#                 x, y = node_map[path[j]].pos
#                 x1, y1 = pos_trans(x, y)
#                 x, y = node_map[path[j+1]].pos
#                 x2, y2 = pos_trans(x, y)
#                 cv2.line((x1, y1), (x2, y2)(0, 55, 200), 2)
#                 cv2.putText(img, str(path[j]))

#!/home/orangepi/miniconda3/bin/python
import cv2
import numpy as np

from sko.SA import SA_TSP
from time import perf_counter
from collections import deque


# 定义方格纸的行数和列数
L = 400
num_rows = 10
num_cols = 10


def get_node(gray_img, L, num_rows, num_cols, offsetx=2, offsety=2, img_width=484, img_height=404):
    if len(gray_img.shape) > 2:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)

    gray_img2 = gray_img.copy()
    # 创建一个字典来存储节点
    nodes = {}

    def constrain(x, low, max):
        if x < low:
            return low
        elif x > max:
            return max
        else:
            return x

    def valid(x, low, max):
        if x < low:
            return -1
        elif x > max:
            return -1
        else:
            return x

    def has_wall(map, p1, p2):  # p1,p2为两个位置坐标
        # 坐标转换成数组索引
        p1 += np.array([offsetx, offsety], dtype='int')
        p2 += np.array([offsetx, offsety], dtype='int')
        p1[0], p1[1] = constrain(
            p1[0], 0, img_width), constrain(p1[1], 0, img_height)
        p2[0], p2[1] = constrain(
            p2[0], 0, img_width), constrain(p2[1], 0, img_height)
        if p1[0] == p2[0] and p1[1] != p2[1]:
            if p1[1] > p2[1]:
                temp = p1[1]
                p1[1] = p2[1]
                p2[1] = temp
            area = map[p1[1]:p2[1], p1[0]]
            sum = np.sum(area)
            if sum > 1000:
                return True
            else:
                return False

        elif p1[0] != p2[0] and p1[1] == p2[1]:
            if p1[0] > p2[0]:
                temp = p1[0]
                p1[0] = p2[0]
                p2[0] = temp
            area = map[p1[1], p1[0]:p2[0]]
            sum = np.sum(area)

            if sum > 1000:
                return True
            else:
                return False


# 循环遍历所有的方格，以数字方式为每个节点命名，并将其存储到字典中
    for row in range(num_rows):
        for col in range(num_cols):
            node_name = row*num_cols + col
            cv2.putText(gray_img2, str(node_name), (col*L//10+15, row*L//10+23),
                        (cv2.FONT_HERSHEY_SIMPLEX), 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            y = row * L//10 + L//20
            x = col * L//10 + L//20
            node_center = np.array([x, y])

            qian = has_wall(gray_img, node_center.copy(),
                            node_center+np.array([L//10, 0]))
            hou = has_wall(gray_img, node_center.copy(),
                           node_center+np.array([-L//10, 0]))
            zuo = has_wall(gray_img, node_center.copy(),
                           node_center+np.array([0, L//10]))
            you = has_wall(gray_img, node_center.copy(),
                           node_center+np.array([0, -L//10]))

            if qian == 1:
                qianmian = -1
            else:
                qianmian = row*num_cols + col+1
            if hou == 1:
                houmian = -1
            else:
                houmian = row*num_cols + col-1
            if zuo == 1:
                zuomian = -1
            else:
                zuomian = (row+1)*num_cols + col
            if you == 1:
                youmian = -1
            else:
                youmian = (row-1)*num_cols + col

            neibor = [qianmian, houmian, zuomian, youmian]
            nei = [valid(i, 0, num_rows*num_cols-1) for i in neibor]
            nodes[node_name] = node_center, [
                not qian, not hou, not zuo, not you], nei

    # for m in range(len(nodes)):
    #     for n in range(4):
    #         if nodes[m][2][n] != -1:
    #             print(nodes[m][2][n])
    #             x, y = nodes[m][0]
    #             x1, y1 = pos_trans(x, y)
    #             b = nodes[m][2][n]
    #             x, y = nodes[b][0]
    #             x2, y2 = pos_trans(x, y)
    #             cv2.line(gray_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    # cv2.imshow("quanlian", gray_img)
    # cv2.waitKey(0)

    # # 打印出节点字典
    # print(nodes)
    # cv2.imwrite("map_with_node.jpg",gray_img2)
    return nodes

class node():
    def __init__(self, id, data):
        pos, dir, available_node = data
        self.id = id
        self.pos = pos
        # self.up = dir[0]
        # self.forword = dir[1]
        # self.left = dir[2]
        # self.right = dir[3]
        self.available_node = [i for i in available_node ]#if i!=-1
        if available_node[0] != -1:
            self.forword  = available_node[0]
        else:
            self.forword  = None
        if available_node[1] != -1:
            self.backword  = available_node[1]
        else:
            self.backword  = None
        if available_node[2] != -1:
            self.left  = available_node[2]
        else:
            self.left  = None
        if available_node[3] != -1:
            self.right  = available_node[3]
        else:
            self.right  = None


    def __str__(self):
        return f"[id:{str(self.id)},pos:{str(self.pos)}]"

    def __repr__(self):
        return f"[id:{str(self.id)},pos:{str(self.pos)}]"

    def __eq__(self, other):
        return self.id == other.id

    def get_available_node(self):
        return [i for i in self.available_node if i!=-1] 

    def get_pos(self):
        return self.pos




np.random.seed(0)
# for i,node in enumerate(nodes):
#     print(i,node.get_available_node())


def shortest_path(start, end, nodes):
    queue = deque([(start, [start])])
    visited = set([start])
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        # print(node,path)
        for next_node in nodes[node].get_available_node():
            if next_node not in visited:
                queue.append((next_node, path + [next_node]))
                visited.add(next_node)
    return None

# path = shortest_path(start_node, end_node, nodes)


nodes = None


def path_len(start_node, end_node, nodes):
    path = shortest_path(start_node, end_node, nodes)
    if path is None:
        print("No path found")
        return np.nan
    else:

        return len(path)


def path_mat():
    mat = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            mat[i, j] = path_len(i, j, nodes)
    return mat


def caculate_dis_mat(node_list, all_nodes):
    l = len(node_list)
    mat = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            mat[i, j] = path_len(node_list[i], node_list[j], all_nodes)
    return mat


def cal_total_distance(routine, distance_matrix):
    num_points, = routine.shape
    sum1 = 0
    sum1 += distance_matrix[-2, routine[0]]
    sum1 += distance_matrix[routine[-1], -1]
    return sum1+sum([distance_matrix[routine[i], routine[(i + 1)]] for i in range(0, num_points - 1)])

# 退火算法TSP 问题


def path_planning(start_node, end_node, through_nodes, nodes):
    path_nodes = np.array(through_nodes+[start_node] + [end_node])
    print("path nodes", path_nodes)
    print(path_nodes)
    try:
        distance_matrix = caculate_dis_mat(path_nodes, nodes)
        points_coordinate = np.array([nodes[i].get_pos() for i in path_nodes])
        num_points = points_coordinate.shape[0]
        print("num_points", num_points)

        start = perf_counter()
    # 执行模拟退火(SA)算法
        sa_tsp = SA_TSP(func=lambda x: cal_total_distance(x, distance_matrix), x0=range(
        num_points-2), T_max=5, T_min=0.1, L=3 * num_points)  # 调用工具箱
    # 结果输出1
        best_points, best_distance = sa_tsp.run()
        print("运行时间是: {:.5f}s".format(perf_counter()-start))  # 计时结束
    
            
        result_cur_best = [path_nodes[i] for i in best_points]
        print("最优路线：", result_cur_best)
        print("最优值：", cal_total_distance(best_points, distance_matrix))  # 数据还原
        return [start_node]+result_cur_best+[end_node]
    except Exception as e:
        print(e) 
        with open("./errer_log.txt",'a+') as f :
            f.write(str(e)) 
    


def locate_node_py_pos(x, y):
    x /= 40
    y /= 40
    x = int(x)
    y = int(y)
    return y*10+x

def nodes_with_quadrant(nodes, node, node_list):
    # 找出同象限及对称象限的节点
    def calculate_quadrant(pos):
        # 计算象限ID
        return [pos[0] > 0, pos[1] > 0]

    # 获取节点的位置，并移到中心
    pos = np.array(node.get_pos()) - np.array([200,200])
    quadrant_node = calculate_quadrant(pos)
    
    # 初始化象限数组
    quadrant = np.zeros((len(node_list),2), dtype=bool)
    
    # 计算node_list中每个节点的象限ID
    for i in range(len(node_list)):
        pos1 = np.array(nodes[node_list[i]].get_pos()) - np.array([200,200])
        quadrant[i] = calculate_quadrant(pos1)

    # 找到在同一个象限的节点及其索引
    same_quadrant_mask = np.all(quadrant == quadrant_node, axis=1)
    same_quadrant_node_index = np.where(same_quadrant_mask)[0]
    same_quadrant_node = [node_list[i] for i in same_quadrant_node_index]

    # 找到在对称象限的节点及其索引
    opposite_quadrant_mask = np.all(quadrant != quadrant_node, axis=1)
    opposite_quadrant_node_index = np.where(opposite_quadrant_mask)[0]
    opposite_quadrant_node = [node_list[i] for i in opposite_quadrant_node_index]

    return same_quadrant_node, same_quadrant_node_index, opposite_quadrant_node, opposite_quadrant_node_index


def symmetric_node(node):
    pos=np.array(node.get_pos())
    pos1=np.array([400,400])-pos
    return pos1[0]//40+pos1[1]//40*10
    

def reduce_node(through_nodes,nodes,know_node,know_node_type):#两个挑选规则，对称象限，相同象限
    nn_duicen=1
    #先找出节点中同象限和中心对称象限的节点，这两个象限的节点分布服从对称规则的约束，其他象限的节点服从宝藏数目的约束
    #通过对称规则挑选节点
    # print("through_nodes",through_nodes)
    same_quadrant_node, same_quadrant_node_index, opposite_quadrant_node, opposite_quadrant_node_index=nodes_with_quadrant(nodes, nodes[know_node], through_nodes)
    if know_node_type=='sp':
        #移除同象限节点，因为已知其是对手节点
        for i in same_quadrant_node:
            through_nodes.remove(i)#因为自己也是同象限节点的内容，所以，这里会把自己也移除掉
         
        #移除对称象限节点，因为已知其是对手节点
        for i in opposite_quadrant_node:
            # print(i,symmetric_node(nodes[know_node]))
            if i==symmetric_node(nodes[know_node]):
                # print(1) 
                through_nodes.remove(i)
    if know_node_type=="op":
        #不能移除同象限节点，因为已知其是宝藏节点
        if know_node in  through_nodes:
            through_nodes.remove(know_node)#移除本身
        #移除对称象限的非对称节点，因为可以推断其是对手宝藏节点
        for i in opposite_quadrant_node:
            if i!=symmetric_node(nodes[know_node]):
                through_nodes.remove(i)
    if know_node_type=="sn":
        #移除同象限节点，因为已知其是对手节点
        for i in same_quadrant_node:
            through_nodes.remove(i)
        #移除对称位置节点，因为可以确定其是对手节点
        for i in opposite_quadrant_node:
            if i==symmetric_node(nodes[know_node]):
                through_nodes.remove(i)
    if know_node_type=="on":
        #不能移除同象限节点，因为已知其是宝藏节点
        if know_node in  through_nodes:
            through_nodes.remove(know_node)#移除本身
        #移除对称位置节点，因为可以确定其是宝藏节点
        for i in opposite_quadrant_node:
            if not nn_duicen:
              if i!=symmetric_node(nodes[know_node]):#移除对称象限的非对称节点，因为已知其是己方节点
                through_nodes.remove(i)
            else:
                through_nodes.remove(i) #如果真假宝藏对称，其对称节点是我方的假宝藏，所以移除

    # print("through_nodes",through_nodes)
    if know_node_type=="is":
        #移除同象限节点，因为已知其是对手节点
        for i in same_quadrant_node:
            through_nodes.remove(i)
        #移除对称位置节点，因为可以确定其是对手节点
        for i in opposite_quadrant_node:
            if i==symmetric_node(nodes[know_node]):
                through_nodes.remove(i)
    if know_node_type=="io":
        #不能移除同象限节点，因为已知其是宝藏节点
        through_nodes.remove(know_node)
        #移除对称位置节点，因为可以确定其是宝藏节点
        for i in opposite_quadrant_node:
            if i!=symmetric_node(nodes[know_node]):
                through_nodes.remove(i)
    # print("through_nodes",through_nodes)
    return through_nodes
         
        
        
 


    


# path=path_planning(0, 19, [12,49,18,56,10,30,27,90], nodes)
# print(path)
# def pos_trans(x, y):
#         return x+40, y+2

if __name__ == '__main__':
    import random
    img = cv2.imread('bn.jpg')
    nodes = get_node(img, L, num_rows, num_cols,
                     offsetx=40, offsety=2, img_width=480, img_height=404)
    nodes = [node(i, nodes[i]) for i in nodes.keys()]
    #print(nodes)
    node_map = nodes
    baozang_nodes = [81,18 ,77,22,48,51, 30,69]
    step_list=[]
    for i in range(100000):
      step=0
      baozang_nodes = [81,18 ,77,22,48,51, 30,69]
      random_state = ["sp"]*3 + ["op"]*3 + ["sn"] + ["on"]
      random.shuffle(random_state)
      while len(baozang_nodes):
        step+=1
        l=len(baozang_nodes)
        idx=random.randint(0,l-1)
        state=random_state[idx]
        baozang_nodes=reduce_node(baozang_nodes,nodes,baozang_nodes[idx],state)
        random_state.pop(idx)
        # print(baozang_nodes)
      step_list.append(step)
    step_list=np.array(step_list)
    print(step_list.mean(),step_list.std())
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.distplot(step_list, bins=20, kde=False, rug=True)
    plt.show()

    # plan_path = path_planning(
    #     0, 99, baozang_nodes, node)
    # # print(len(plan_path))
    # # print("plan_path",plan_path)
    # if plan_path is not None:
    #     for i in range(len(plan_path)-1):
    #         path = shortest_path(plan_path[i], plan_path[i+1], node_map)
    #         # print("path",path)
    #         # print(len(path))
    #         for j in range(len(path)-1):
    #             x, y = node_map[path[j]].pos
    #             x1, y1 = pos_trans(x, y)
    #             # print(x,y)
    #             # print(x1,y1)
    #             x, y = node_map[path[j+1]].pos
    #             x2, y2 = pos_trans(x, y)
    #             cv2.line(img, (x1, y1), (x2, y2), (0, 55, 200), 2)
    #             cv2.putText(
    #                 img, str(path[j]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    # cv2.imshow("jiedian", img)
    # cv2.waitKey(0)
    