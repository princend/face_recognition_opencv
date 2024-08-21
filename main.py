import cv2
import numpy as np
import os
import pickle
import time
import json
import face_recognition 
from PIL import Image
# 設定一個 tolerance，當最短距離超過此值時，我們認定無法辨識，回傳 unknown。
tolerance = 0.6    

def preprocessing_encode():
    # 读取JSON文件
    with open('singer-map.json', 'r',encoding='utf-8') as file:
        known_face_list = json.load(file)

    # 将encode字段改为None
    for face in known_face_list:
        face['encode'] = None

    # load image data by large model of face landmarks
    for data in known_face_list:
        try:
            img = cv2.imread('known_singer_image/'+data['filename'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data['encode'] = face_recognition.face_encodings(img, model='small')[0]  # use small model of face landmarks
        except Exception as error:
            print('data is ',data,'error is ',error )

    known_face_encodes = [data['encode'] for data in known_face_list]
    return known_face_list,known_face_encodes
    
#辨認    
def recognize(img_input,known_face_list,known_face_encodes,tolerance = 0.6)->str:
    
        # img = cv2.imread(img_input)
        img = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)

        cur_face_locs = face_recognition.face_locations(img)
        cur_face_encodes = face_recognition.face_encodings(img, cur_face_locs, model='small')  # use small model of face landmarks

        for cur_face_encode in cur_face_encodes:
            face_distance_list = face_recognition.face_distance(known_face_encodes, cur_face_encode)
            min_distance_index = np.argmin(face_distance_list)
            if face_distance_list[min_distance_index] < tolerance:
                result = known_face_list[min_distance_index]['name']
            else:
                result = 'unknown'
            # distance_with_name_list = [(face_data['name'], round(distance, 4)) for face_data, distance in zip(known_face_list, face_distance_list)]
            # print(f'辨識檔案: {img}, 辨識結果: {result}, 特徵距離: {distance_with_name_list}, 相差: {round(abs(distance_with_name_list[0][1] - distance_with_name_list[1][1]), 4)}')    
            return result            