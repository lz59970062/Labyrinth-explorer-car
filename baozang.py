import cv2
import numpy as np
import platform
from rknnlite.api import RKNNLite
import time 
import os
import glob

INPUT_SIZE = 224

# RK356X_RKNN_MODEL = 'resnet18_for_rk356x.rknn'

# RK3588_RKNN_MODEL = '/home/orangepi/Desktop/temtes/mobilenet_fcn.rknn'

RK3588_RKNN_MODEL4 = '/home/orangepi/Desktop/temtes/effnet.rknn'
RK3588_RKNN_MODEL1 = '/home/orangepi/Desktop/temtes/ResNet_fcn.rknn'
RK3588_RKNN_MODEL2 = '/home/orangepi/Desktop/temtes/ResNet_dark.rknn'
RK3588_RKNN_MODEL3 = '/home/orangepi/Desktop/temtes/resnet_fcn.rknn'
def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def init_model(rknn_model,core):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
        return None
    ret = rknn_lite.init_runtime(core_mask=core)#|RKNNLite.NPU_CORE_2|RKNNLite.NPU_CORE_1
    if ret != 0:
        print('Init runtime failed')
        exit(ret)
        return None
    return rknn_lite

rknn_lite1=init_model(RK3588_RKNN_MODEL3,RKNNLite.NPU_CORE_0)
# rknn_lite2=init_model(RK3588_RKNN_MODEL2,RKNNLite.NPU_CORE_1) 
def baozang_tetect(img,team_color="none"):
    my_class_dir=['blue_cir', 'blue_down', 'blue_tan', 'other', 'red_cir', 'red_down', 'red_tan']
    # my_class_dir=['red_down', 'red_cir', 'red_tan', 'blue_cir', 'blue_down', 'blue_tan', 'other']//‘effnet’
    #my_class_dir=['blue_cir', 'blue_tan','down','other',  'red_cir', 'red_tan']
    # print('done')
    target_size=(224,224)
    
    # class_names = os.listdir(data_dir)
    # my_class_dir = ['blue_cir', 'blue_tan', 'other', 'red_cir', 'red_tan']
    # img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)

    outputs1=rknn_lite1.inference([img])
    soft_max1=softmax(np.array(outputs1[0][0]))
    # outputs2=rknn_lite2.inference([img]) 
    # soft_max2=softmax(np.array(outputs2[0][0]))
    
    predicted_class1 = np.argmax(soft_max1)
    # predicted_class2 = np.argmax(soft_max2) 
    # if predicted_class1!=predicted_class2:
    #     return "detect error"

    # print(np.round(soft_max2,2)) 
    soft_max2=soft_max1
    predicted_class2=predicted_class1
    # print(f"predicted type is {my_class_dir[predicted_class1]}")
    if soft_max2[predicted_class2]>0.2:
    
      if team_color=="red":
        if my_class_dir[predicted_class2] == "red_tan" :
            result_str = "sp"
            return result_str

        elif my_class_dir[predicted_class2] == "red_cir" :
            result_str = "sn"
            return result_str
        
        elif my_class_dir[predicted_class2] == "blue_tan" :
            result_str = "on"
            return result_str
        
        elif my_class_dir[predicted_class2] == "blue_cir" :
            result_str = "op"
            return result_str
        elif my_class_dir[predicted_class2] == 'blue_down' :
            result_str = "io"
            return result_str
        elif my_class_dir[predicted_class2] == 'red_down' :
            result_str = "is"
            return result_str
        else :
            result_str = "detect error"
            return result_str
        

      elif team_color=="blue":
        if my_class_dir[predicted_class2] == "red_tan" :
            result_str = "op"
            return result_str

        elif my_class_dir[predicted_class2] == "red_cir" :
            result_str = "on"
            return result_str
        
        elif my_class_dir[predicted_class2] == "blue_tan" :
            result_str = "sn"
            return result_str
        
        elif my_class_dir[predicted_class2] == "blue_cir" :
            result_str = "sp"
            return result_str
        elif my_class_dir[predicted_class2] == 'blue_down' :
            result_str = "is"
            return result_str
        elif my_class_dir[predicted_class2] == 'red_down' :
            result_str = "io"
            return result_str
        else :
            result_str = "detect error"
            return result_str
      else :
        result_str = "team_color error"
        return result_str
    else :
        return ""

# if  __name__ == "__main__":
#     img_path= '/home/orangepi/Desktop/rknn_toolkit_lite2/examples/inference_with_lite/my_data/15.jpg'
#     my_str=baozang_tetect(img_path,"red")
#     print(my_str)




# class_data=[ 'blue_cir', 'blue_tan','down','other',  'red_cir', 'red_tan']
# # print(class_dir[1])
# def my_fc():
    
#     target_size=(224,224)
#     data_dir="/home/orangepi/Desktop/temtes/data"
#     class_names=os.listdir(data_dir)
#     correct_count = 0
#     total_count = 0
#     idx = 0

#     rknn_model=RK3588_RKNN_MODEL
#     rknn_lite = RKNNLite()
#     ret = rknn_lite.load_rknn(rknn_model)
#     if ret != 0:
#         print('Load RKNN model failed')
#         exit(ret)
#         return None
#     print('done')
#     ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
#     target_size=(224,224)
#     # print(class_names)
#     for class_name in class_names:
#         class_dir= os.path.join(data_dir, class_name)
#         file_ls = os.listdir(class_dir)

#         # print(class_name)

#         for file_name in file_ls:
#             # 读取图片，并进行预处理
#             img_path = os.path.join(class_dir, file_name)
#             print(img_path)
#             img = cv2.imread(img_path)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img, target_size)

#             outputs=rknn_lite.inference([img])
#             predicted_class = np.argmax(softmax(np.array(outputs[0][0])))
           

#             idx += 1
#             print(predicted_class)
#             print(class_data[predicted_class])
#             print(class_name)
#             # print(class_names)
#                 # 检查预测标签是否正确
#             if class_data[predicted_class] == class_name:
#                 correct_count += 1
#             total_count += 1

# # 打印模型的准确率
#     accuracy = correct_count / total_count
#     print('Model accuracy: {:.2%}'.format(accuracy))
    
# if __name__=="__main__":
#     my_fc()