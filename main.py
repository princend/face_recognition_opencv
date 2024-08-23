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
        print(data)
        try:
            img = cv2.imread('known_singer_image/'+data['filename'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(img, model='small')[0]  # use small model of face landmarks
            data['encode'] = encoding.tolist()
            # data['encode'] = face_recognition.face_encodings(img, model='small')[0]  # use small model of face landmarks
        except Exception as error:
            print('data is ',data,'error is ',error )

    # known_face_encodes = [data['encode'] for data in known_face_list]
      # Save to JSON file
    with open('singer-map-with-encodings.json', 'w', encoding='utf-8') as file:
        json.dump(known_face_list, file, ensure_ascii=False, indent=4)

    return known_face_list, [data['encode'] for data in known_face_list]
    # return known_face_list,known_face_encodes
    
#辨認    
def recognize(img_input, known_face_list, known_face_encodes, tolerance=0.6) -> str:
    img = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    cur_face_locs = face_recognition.face_locations(img)
    cur_face_encodes = face_recognition.face_encodings(img, cur_face_locs, model='small')

    if not cur_face_encodes:
        # No faces found in the image
        return 'unknown'
    
    results = []
    for cur_face_encode in cur_face_encodes:
        face_distance_list = face_recognition.face_distance(known_face_encodes, cur_face_encode)
        min_distance_index = np.argmin(face_distance_list)
        if face_distance_list[min_distance_index] < tolerance:
            result = known_face_list[min_distance_index]['name']
        else:
            result = 'unknown'
        results.append(result)    
    
    if len(results)>=1:
       return results[0]
    elif len(results)==0:
       return 'unknown'
    
        
def load_encodings(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Convert lists back to numpy arrays
    for face in data:
        face['encode'] = np.array(face['encode'])
    
    known_face_list = data
    known_face_encodes = [face['encode'] for face in data]
    return known_face_list, known_face_encodes                   


# json_file= 'singer-map-with-encodings.json'
# known_face_list, known_face_encodes = load_encodings(json_file)