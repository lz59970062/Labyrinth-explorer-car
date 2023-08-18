import cv2
import numpy as np
import math

# import matplotlib.pyplot as plt 

low_threshold = np.array([90,87,113])
high_threshold = np.array([120, 255, 255])
def main():
    # Read the image file
    # image_path ="2.png"#"12.jpeg" #
    # image = cv2.imread(image_path)
  cap=cv2.VideoCapture(0)
  while True:
    ret,image=cap.read()
    # Detect QR code locator markers
    locator_markers = detect_locator_markers(image)
    img=image.copy()
    for locator_marker in locator_markers:
       cv2.drawContours(img, [locator_marker], -1, (0,255,0), 2)
    cv2.imshow("QR Code Locator Marker Detection", img)
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
    roi=roi_proc(roi)
    # Detect circles in the ROI
    circles = detect_circles(roi)
    print(circles)
    # Draw circles on the ROI
    #for circle in circles :
    #  cv2.circle(roi, (circle[0], circle[1]), circle[2]+4, (208, 208, 208), -1)
    roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51, 18)
    for circle in circles :
       cv2.circle(binary_image, (circle[0], circle[1]), circle[2]+4, (0, 0, 0), -1)
    # bordered_img = cv2.copyMakeBorder(binary_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    binary=fanggetu(binary_image)
    # Show the ROI
    cv2.imshow("ROI with Circle Detection", roi)
    
    # cv2.imshow("final",bordered_img)

    cv2.imshow("final2",binary)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def fanggetu(binary_image):
    kernel = np.array([[0,0,1,0,0],
                       [0,1,1,1,0],
                       [1,1,1,1,1],
                       [0,1,1,1,0],
                       [0,0,1,0,0]],np.uint8)
   
    # kernel = np.ones((4,4),np.uint8)#定义核
    binary = cv2.dilate(binary_image,kernel,iterations = 1)# 腐蚀
    return binary


def roi_proc(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_threshold, high_threshold)
    # cv2.imshow("mask",mask)
    # cv2.imshow("roi",roi)
    # cv2.waitKey(0)
    blue_area=np.argwhere(mask>0)
    team_color=None 
    if len(blue_area)>0:
        blue_area_mean=blue_area.mean(axis=0)
        blue_pos=blue_area_mean//np.array([mask.shape[0]/2,mask.shape[1]/2])  
        blue_pos=blue_pos.astype('int') 
        # cv2.circle(roi,(int(blue_area_mean[1]),int(blue_area_mean[0])),5,(255,255,255),-1 )
        if blue_pos[0]==0 and blue_pos[1]==1:
            roi=cv2.flip(roi,1)
            team_color='red'
        elif blue_pos[0]==1 and blue_pos[1]==0:
            roi=cv2.flip(roi,1)
            team_color='blue' 
    gray = roi[:,:,1]
    gray = 255-gray
    # cv2.imshow("roi",roi)
    # cv2.waitKey(0)
    # gray=roi
    x_array=gray.sum(axis=0)
    y_array=gray.sum(axis=1)
    x_2=int(len(x_array)/2)
    y_2=int(len(y_array)/2)
    left_xmax=np.argmax(x_array[:x_2])-30
    right_xmax=np.argmax(x_array[x_2:])+x_2+30
    left_ymax=np.argmax(y_array[:y_2])-2
    right_ymax=np.argmax(y_array[y_2:])+y_2+2
    roi=roi[left_ymax :right_ymax ,left_xmax :right_xmax ]
    roi=cv2.resize(roi,(480,404),cv2.INTER_LINEAR)
    # cv2.imshow("roi2",roi)
    # cv2.waitKey(0)
    return roi ,team_color 
    # plt.plot(y_array)
    # plt.plot(x_array)
    # plt.show()
    

def iou(box1, box2):
    x1, y1, x1_max, y1_max = box1
    x2, y2, x2_max, y2_max = box2

    intersection_x1 = max(x1, x2)
    intersection_y1 = max(y1, y2)
    intersection_x2 = min(x1_max, x2_max)
    intersection_y2 = min(y1_max, y2_max)

    intersection_area = max(intersection_x2 - intersection_x1, 0) * max(intersection_y2 - intersection_y1, 0)
 
    box1_area = (x1_max - x1) * (y1_max - y1)
    box2_area = (x2_max - x2) * (y2_max - y2)

    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

def non_maximum_suppression(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]
    keep_indices = []

    while len(sorted_indices) > 0:
        current_index = sorted_indices[0]
        keep_indices.append(current_index)

        other_indices = sorted_indices[1:]
        other_boxes = boxes[other_indices]

        ious = np.array([iou(boxes[current_index], other_box) for other_box in other_boxes])
        sorted_indices = sorted_indices[np.where(ious <= iou_threshold)[0] + 1]

    return keep_indices
 

def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


# def detect_locator_markers(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   #灰度化
#     # _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)  #二值化
#     # cv2.imshow("binary" , binary_image)
#     # cv2.waitKey(0)
#     # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     # # eroded_image = cv2.erode(binary_image, kernel, iterations=1)
#     # dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
#     contour,hierarch=cv2.findContours(binary_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  #contour检测到的轮廓的列表  hierarch轮廓的层级信息
#     quadrilaterals = []  #存储检测到的四边形轮廓
#     scores = []     #存储每个四边形轮廓的得分（面积）
#     for bound in contour:
#         area = cv2.contourArea(bound)  #计算当前轮廓信息
#         if area < 300 or area > 10000:
#             continue
#         #对当前轮廓进行多边形逼近，通过指定逼近精度来减少轮廓中的点数。
#         perimeter = cv2.arcLength(bound, True)
#         approx = cv2.approxPolyDP(bound, 0.02 * perimeter, True)
 
#         if len(approx) == 4:#判断矩形
#             x, y, w, h = cv2.boundingRect(bound) #轮廓的最小外接矩形的坐标和尺寸
#             if w <10 or h <10 :
#                 continue
#             _, row, _ = hierarch.shape#获取层级信息 hierarch 的行数
#             cnt = 0 #子轮廓数量
#             for j in range(row):
#                 #cnt += 1
#                 if hierarch[0][j][2] >= 0 :
#                     child_contour_index = hierarch[0][j][2]#将当前行的子轮廓索引赋值给变量 child_contour_index
#                     child_contour = contour[child_contour_index]#根据子轮廓索引从轮廓列表 contour 中获取对应的子轮廓
#                     #对子轮廓进行多边形逼近，得到逼近后的多边形点集列表 approx
#                     approx_1 = cv2.approxPolyDP(child_contour, 0.04 * cv2.arcLength(child_contour, True), True)
#                     if len(approx_1) == 4:#判断矩形
#                         x, y, w, h = cv2.boundingRect(child_contour) #轮廓的最小外接矩形的坐标和尺寸
#                         if w <10 or h <10 :
#                             continue
#                         cnt += 1
#                         #在图像 binary_image 上绘制当前子轮廓的轮廓线，颜色为绿色，线宽为2
#                         cv2.drawContours(image, [approx_1], 0, (0, 255, 0), 2)
#                     if cnt == 2:
#                         #found += 1
#                         cnt = 0
#                         cos_angles = [angle_cos(approx[i][0], approx[(i + 1) % 4][0], approx[(i + 2) % 4][0]) for i in range(4)] #计算当前四边形的四个角的余弦值，并将结果存储在列表 cos_angles 中
#                         max_cos_angle = np.max(cos_angles)  #计算列表 cos_angles 中的最大余弦值

#                         if 0.005 < max_cos_angle < 0.25:    #筛选出近似为矩形的四边形  
#                             quadrilaterals.append(approx.reshape(4, 2)) #将符合条件的四边形轮廓添加到 quadrilaterals 列表中，并将其形状重塑为4行2列
#                             scores.append(area) #将当前四边形轮廓的面积添加到 scores 列表中
            

#     quadrilaterals = np.array(quadrilaterals) 
#     scores = np.array(scores)

#     bounding_boxes = [cv2.boundingRect(q) for q in quadrilaterals] #根据四边形轮廓计算最小外接矩形的坐标和尺寸，得到一个包含多个最小外接矩形的列表 bounding_boxes
#     bounding_boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in bounding_boxes]) #将每个最小外接矩形的坐标和尺寸转换为左上角和右下角坐标的形式，并存储在 numpy 数组 bounding_boxes 中
#     keep_indices = non_maximum_suppression(bounding_boxes, scores, iou_threshold=0.3) #调用 non_maximum_suppression() 函数对最小外接矩形进行非最大值抑制，筛选出重叠程度较小的最佳候选框的索引，存储在列表 keep_indices 中

#     locator_markers = [quadrilaterals[i] for i in keep_indices] #根据最佳候选框的索引从四边形轮廓列表 quadrilaterals 中获取对应的定位器标记

#     return locator_markers


def detect_locator_markers(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   #灰度化
    # _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)  #二值化
    # cv2.imshow("binary" , binary_image)
    # cv2.waitKey(0)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # # eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    # dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    contour,hierarchy=cv2.findContours(binary_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  #contour检测到的轮廓的列表  hierarch轮廓的层级信息
    quadrilaterals = []  #存储检测到的四边形轮廓
    scores = []     #存储每个四边形轮廓的得分（面积）
    hierarchy = hierarchy[0]
    mark=0
    area_parent_contours = []        
    for i in range(len(contour)):
        k = i
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c = c + 1
        if hierarchy[k][2] != -1:
            c = c+1
        if c >= 2:
            area = cv2.contourArea(contour[k-2])
            if area < 300 or area > 10000:
                continue
            perimeter = cv2.arcLength(contour[k-2], True)
            approx = cv2.approxPolyDP(contour[k-2], 0.02 * perimeter, True)
            if len(approx) == 4:
                cos_angles = [angle_cos(approx[i][0], approx[(i + 1) % 4][0], approx[(i + 2) % 4][0]) for i in range(4)]
                max_cos_angle = np.max(cos_angles)
                if 0.005 < max_cos_angle < 0.25:
                    quadrilaterals.append(approx.reshape(4, 2))
                    scores.append(area)

    quadrilaterals = np.array(quadrilaterals)
    scores = np.array(scores)

    bounding_boxes = [cv2.boundingRect(q) for q in quadrilaterals]
    bounding_boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in bounding_boxes])
    keep_indices = non_maximum_suppression(bounding_boxes, scores, iou_threshold=0.3)

    locator_markers = [quadrilaterals[i] for i in keep_indices]

    return locator_markers
# def detect_locator_markers(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#     # cv2.imshow("binary" , binary_image)
#     # cv2.waitKey(0)
#     # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     # # eroded_image = cv2.erode(binary_image, kernel, iterations=1)
#     # dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
#     contours, hierarchy = cv2.findContours( binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     #print(hierarchy)
    
#     quadrilaterals = []
#     scores = []
#     ic=0
#     parentIdx = -1
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area < 300 or area > 10000:
#             continue
        
#         perimeter = cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

#         if len(approx) == 4:
#             cos_angles = [angle_cos(approx[i][0], approx[(i + 1) % 4][0], approx[(i + 2) % 4][0]) for i in range(4)]
#             max_cos_angle = np.max(cos_angles)

#             if 0.005 < max_cos_angle < 0.25:
#                 quadrilaterals.append(approx.reshape(4, 2))
#                 scores.append(area)

#     quadrilaterals = np.array(quadrilaterals)
#     scores = np.array(scores)

#     bounding_boxes = [cv2.boundingRect(q) for q in quadrilaterals]
#     bounding_boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in bounding_boxes])
#     keep_indices = non_maximum_suppression(bounding_boxes, scores, iou_threshold=0.3)

#     locator_markers = [quadrilaterals[i] for i in keep_indices]

#     return locator_markers


def extract_and_rescale_roi(image, locator_markers):

    # print(locator_markers)
    # print(len(locator_markers))
    
    # Sort the locator markers by their x and y coordinates
    locator_markers = np.array(locator_markers)
    sums = np.array([pt[:, 0].sum() + pt[:, 1].sum() for pt in locator_markers])
    sorted_indices = np.argsort(sums)
    locator_markers = locator_markers[sorted_indices]
    # print(locator_markers)
    # Determine the four corner points of the ROI
    top_left, top_right, bottom_left, bottom_right = locator_markers
    top_left =top_left.mean(axis=0)
    top_right =top_right.mean(axis=0)
    bottom_left =bottom_left.mean(axis=0)
    bottom_right =bottom_right.mean(axis=0)
    # Sort top_left and top_right
    if top_right[0] < top_left[0]:
        top_left, top_right = top_right, top_left

    # Sort bottom_left and bottom_right
    if bottom_right[0] < bottom_left[0]:
        bottom_left, bottom_right = bottom_right, bottom_left

    # Calculate the width and height of the ROI
    width = int(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))
    height = int(max(np.linalg.norm(top_left - bottom_left), np.linalg.norm(top_right - bottom_right)))

    # Apply a perspective transform to extract the ROI
    src_points = np.float32([top_left, top_right, bottom_left, bottom_right])
    p1=top_left-top_right
    p2=bottom_left-bottom_right
    p3=top_left-bottom_left
    p4=top_right-bottom_right
    cos_12=np.dot(p1,p2)/(np.linalg.norm(p1)*np.linalg.norm(p2))
    cos_34=np.dot(p3,p4)/(np.linalg.norm(p3)*np.linalg.norm(p4))
    print(cos_12,cos_34)

    if np.abs(cos_12)<0.95 or np.abs(cos_34)<0.95:
        return None
    dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    roi = cv2.warpPerspective(image, transformation_matrix, (width, height))
    if roi.shape[0]>roi.shape[1]:
        roi =cv2.rotate(roi,cv2.ROTATE_90_CLOCKWISE)
    # cv2.imshow("roi_scal",roi)
    # cv2.waitKey(0)
    return roi

def detect_circles(image):
    # Convert the image to grayscale and apply binary thresholding
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 8)
    # Apply erosion to the binary image
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    eroded_image = binary_image
    # cv2.imshow("binary" , eroded_image)
    # Find contours
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter circles
    circle_candidates = []
    
    scores = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20 or area > 1000:
            continue
        print(area)
        perimeter = cv2.arcLength(contour, True)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        # print(x,y)
        # y=400-y
        aspect_ratio = float(perimeter) / (2 * radius * math.pi)
        circularity = (4 * math.pi * area) / (perimeter * perimeter)

        if 0.6 <= aspect_ratio <= 1.3 and 0.6 <= circularity <= 1.4:
            circle_candidates.append((int(x), int(y), int(radius)))
            scores.append(area)
    # for contour in contours:
    #   if len(contour) >= 5:  # cv2.fitEllipse() 要求轮廓至少有5个点
    #     area = cv2.contourArea(contour)
    #     perimeter = cv2.arcLength(contour, True)
        
    #     # 使用轮廓拟合椭圆
    #     (center, axes, angle) = cv2.fitEllipse(contour)
    #     (x, y), (major_axis, minor_axis) = center, axes

    #     # 计算椭圆的长短轴比
    #     aspect_ratio = float(minor_axis) / float(major_axis)

    #     # 计算轮廓的圆形度
    #     circularity = (4 * math.pi * area) / (perimeter * perimeter)

    #     # 根据长短轴比和圆形度来判断轮廓是否接近椭圆
    #     if 0.8 <= aspect_ratio <= 1.2 and 0.8 <= circularity <= 1.2:
    #         circle_candidates.append((int(x), int(y), int(major_axis / 2+minor_axis / 2)))
    #         scores.append(area)

    # Apply non-maximum suppression
    circle_candidates = np.array(circle_candidates)
    scores = np.array(scores)
    bounding_boxes = np.array([[x - radius, y - radius, x + radius, y + radius] for x, y, radius in circle_candidates])
    keep_indices = non_maximum_suppression(bounding_boxes, scores, iou_threshold=0.3)

    circles = [circle_candidates[i].tolist() for i in keep_indices]

    return circles 



if __name__ == "__main__":
    main()
