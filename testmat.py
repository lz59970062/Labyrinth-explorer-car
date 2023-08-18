import cv2
import time

def process_image_using_mat(image):
    for i in range(1000):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 50, 150)
    return edges

def process_image_using_umat(umat_image):
    for i in range(1000):
        gray_image = cv2.cvtColor(umat_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 50, 150)
    return edges

# 读取输入图像
input_image_path = "maze.jpg"
input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

if input_image is None:
    print("Could not open or find the image.")
    exit(-1)

# 使用 Mat 处理图像并计时
start_time_mat = time.time()
edges_mat = process_image_using_mat(input_image)
end_time_mat = time.time()
umat_image = cv2.UMat(input_image)
# 使用 UMat 处理图像并计时
start_time_umat = time.time()
edges_umat = process_image_using_umat(umat_image)
end_time_umat = time.time()
edges_umat=edges_umat.get()
# 计算并显示执行时间
elapsed_time_mat = end_time_mat - start_time_mat
elapsed_time_umat = end_time_umat - start_time_umat
print("Elapsed time using Mat: {:.6f} seconds".format(elapsed_time_mat))
print("Elapsed time using UMat: {:.6f} seconds".format(elapsed_time_umat))

# 显示结果
cv2.imshow("Input Image", input_image)
cv2.imshow("Output Image (Mat)", edges_mat)
cv2.imshow("Output Image (UMat)", edges_umat)

# 等待按键，然后退出
cv2.waitKey(0)
cv2.destroyAllWindows()
