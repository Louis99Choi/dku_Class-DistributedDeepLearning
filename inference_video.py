import os
import cv2
import csv
import math
import torch
import natsort
import shutil
from tqdm import tqdm
import pandas as pd
from analysis import postprogress_pandas
from HRNet.tools import test
from PIL import Image
import numpy as np

if os.name == 'posix':
    import pathlib
    temp = pathlib.PosixPath
    pathlib.WindowsPath = pathlib.PosixPath

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

path =  '.\\data'

file_list = []
inference_list = []
def PIL2OpenCV(pil_image):
    numpy_image= np.array(pil_image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image

def OpenCV2PIL(opencv_image):
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)
    return pil_image

def main():
    if os.path.exists('./data'):
        shutil.rmtree('./data')
    os.mkdir('./data')
    # # Model
    model = torch.hub.load('./yolo/yolov5', 'custom', path='./yolo/yolov5/best.pt',source='local', force_reload=True)  # local model
    video_path = os.listdir('./video/inference')

    inference_list = [os.path.join('./video/inference',i) for i in video_path]
    cap = cv2.VideoCapture(inference_list[0])
    v1_fps = round(cap.get(cv2.CAP_PROP_FPS))
    cap = cv2.VideoCapture(inference_list[1])
    v2_fps = round(cap.get(cv2.CAP_PROP_FPS))
    gl_fps = math.gcd(v1_fps,v2_fps)

    for i in tqdm(inference_list):
        video_path = i
        cap = cv2.VideoCapture(video_path)
        fps = int(round(cap.get(cv2.CAP_PROP_FPS))//gl_fps)
        # fps = 1
        word = [word for word in ['Top', 'Front','top','front'] if word in i][0]
        word = word.capitalize() if word.islower() else word
        os.mkdir(f'./data/{word}')
        while (cap.isOpened()):
            ret, img = cap.read()
            if not ret:
                break
            if (int(cap.get(1)) % fps == 0):
                (h, w) = img.shape[:2]
                if w > h:
                    if w !=1920 and h !=1080:
                        img = cv2.resize(img,(1920,1080))
                elif w < h:
                    if w !=1080 and h !=1920:
                        img = cv2.resize(img,(1080,1920))
                    img = OpenCV2PIL(img)
                    img = img.rotate(90,expand=True)
                    img = PIL2OpenCV(img)

                cv2.imwrite(f'./data/{word}/{str(int(cap.get(1)/fps))}.jpg',img)
        cap.release()

    for i in os.listdir('data'):
        video_name = []
        scales = []
        center_w = []
        center_h = []
        data_path = os.path.join('./data',i)
        data_list = natsort.natsorted(os.listdir(data_path))
        for name in data_list:
            img = cv2.imread(os.path.join(data_path,name))
            results = model(img[..., ::-1], size=640) # batch of images
            try:
                j = results.pandas().xyxy[0]
                img_h, img_w, _ = img.shape
                xmin = round(j.iloc[0,0],6)
                ymin = round(j.iloc[0,1],6)
                xmax = round(j.iloc[0,2],6)
                ymax = round(j.iloc[0,3],6)
                xmin = math.floor(xmin)
                xmax = math.ceil(xmax)
                ymin = math.floor(ymin)
                ymax = math.ceil(ymax)
                pred_cx = (xmax+xmin)/2
                pred_cy = (ymax+ymin)/2
                scale = (max((xmax-xmin),(ymax-ymin))/200)
                video_name.append(name)
                scales.append(scale)
                center_w.append(pred_cx)
                center_h.append(pred_cy)
            except:
                pass

        word = [word for word in ['Top', 'Front','top','front'] if word in i][0]
        word = word.capitalize() if word.islower() else word
        data = pd.DataFrame(zip(video_name, scales, center_w, center_h, ))
        data.columns=['image_name','scale','center_w','center_h']
        data = data.reindex(columns = data.columns.tolist() + ['original_0_x','original_0_y','original_1_x','original_1_y','original_2_x','original_2_y','original_3_x','original_3_y','original_4_x','original_4_y','original_5_x','original_5_y','original_6_x','original_6_y','original_7_x','original_7_y','original_8_x','original_8_y'])

        data.to_csv(f'./inference.csv',index=False)

        inference = pd.DataFrame(test.alignment(cfg='./HRNet/experiments/animal/inference.yaml', model_file='./HRNet/model_best_forth.pth',data_path=word))
        total = pd.concat([data[['image_name','scale','center_w','center_h']],inference],axis=1)
        total.columns=['image_name','scale','center_w','center_h','original_0_x','original_0_y','original_1_x','original_1_y','original_2_x','original_2_y','original_3_x','original_3_y','original_4_x','original_4_y','original_5_x','original_5_y','original_6_x','original_6_y','original_7_x','original_7_y','original_8_x','original_8_y']
        total.to_csv(f'./inference_{word}.csv',index=False)
        os.remove('./inference.csv')
    postprogress_pandas()
if __name__ == '__main__':
    main()