import cv2
import os
import csv
import glob
import base64
import random
import shutil
from time import localtime
from time import strftime



def get_image(path: str):
    name = [['image_name','scale','center_w','center_h','original_0_x','original_0_y','original_1_x','original_1_y','original_2_x','original_2_y','original_3_x','original_3_y','original_4_x','original_4_y','original_5_x','original_5_y','original_6_x','original_6_y','original_7_x','original_7_y','original_8_x','original_8_y'
]]
    print(" ======= Parsing Video data is : ", path)
    filepath = path
    video = cv2.VideoCapture(f'{filepath}')

    if not video.isOpened():
        print("Could not Open :", filepath)
        exit(0)

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))


    print("length :", length)
    print("width :", width)
    print("height :", height)
    print("fps :", fps, "\n")

    currunt_time = localtime()
    try:
        os.makedirs("data/images/")
    except OSError:
        pass

    count = 0

    while (video.isOpened()):
        ret, img = video.read()
        time = int(video.get(1))
        if ((time) % (fps//2) == 0):  # 앞서 불러온 fps 값을 사용하여 1초마다 추출
            tm = localtime()
            capturedtime = strftime('%Y%m%d_%H%M%S_', tm)
            try:
                cv2.imwrite(f'data/images/{capturedtime}{str(time)}.jpg', img)
            except:
                i
                pass
            print('Saved frame number :', str(time))
            name.append([f'{capturedtime}{str(time)}.jpg',f'{max(width,height)/200}',f'{width/2}',f'{height/2}'])
            count += 1
        if(ret == False):
            break
    with open("data/video.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(name)
    print(name)

    video.release()

if __name__ == '__main__':
    img_list = []
    #get_image(os.path.abspath('IMG_9799.MOV'))

    # path = 'D:\\FND\\data\\Raw Video Files\\'
    # for i, f in enumerate(glob.glob(path+"**",recursive=True)):
    #     extend = os.path.splitext(f)[1]
    #     # if (extend == '.jpg'):
    #     #     img_list.append(f)
    #     if (extend == '.mp4' or extend == '.MOV'):
    #         get_image(f)

    path = 'D:\\FND\\test\\data\\'
    img_list = os.listdir(path+'images')

    valid_lsit = random.sample(img_list,int(len(img_list)*0.2))
    for i in valid_lsit:
        shutil.copyfile(os.path.join(path+'images',i),os.path.join(path+'valid',i))
        # print(os.path.join(path + 'images', i), os.path.join(path + 'label', i))
    train_list = list(set(img_list) - set(valid_lsit))
    labeld_list = random.sample(train_list, int(len(img_list) * 0.2))
    for i in labeld_list:
        shutil.copyfile(os.path.join(path+'images',i),os.path.join(path+'label',i))
        # print(os.path.join(path + 'images', i), os.path.join(path + 'label', i))
    unlabel_list = list(set(train_list) - set(labeld_list))
    for i in unlabel_list:
        shutil.copyfile(os.path.join(path+'images',i),os.path.join(path+'unlabel',i))
        # print(os.path.join(path + 'images', i), os.path.join(path + 'unlabel', i))
