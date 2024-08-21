import cv2
import numpy as np
import face_recognition
import os
import pickle
import time


# known_face_list = [
#     {
#         'name': 'Hyun Bin',
#         'filename': 'shanbin.jpeg',
#         'encode': None,
#     },
#     {
#         'name': 'Son Ye Jin',
#         'filename': 'yijen.jpeg',
#         'encode': None,        
#     },
# ]


# # load image data
# for data in known_face_list:
#     image_path=data['filename']
#     if not os.path.isfile(image_path):
#         print(f"文件不存在: {image_path}")
#     else:    
#         img = cv2.imread(image_path)
#         if img is None:
#             print(f"无法加载图像: {image_path}")
#         else:    
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             data['encode'] = face_recognition.face_encodings(img)[0]
    
# known_face_encodes = [data['encode'] for data in known_face_list]
# tolerance = 0.6



# test_fn_list = ['yijen-t1.jpeg', 'yijen-t2.jpeg', 'yijen-t3.jpeg', 'shanbin_and_yijen.jpeg']



# # save known_face_list to dat file
# with open('faces.dat', 'wb') as f:
#     pickle.dump(known_face_list, f)
    
    



# for fn in test_fn_list:
#     image_path=fn
#     if not os.path.isfile(image_path):
#          print(f"test文件不存在: {image_path}")
#     else:    
#         img = cv2.imread(fn)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         _t = time.time()

#         face_recognition.face_locations(img) # use HOG model to detect face locations

#         _t1 = time.time()

#         face_recognition.face_locations(img, model='cnn') # use CNN model to detect face locations

#         print(f'HOG: {round(_t1 - _t, 2)} secs, CNN: {round(time.time() - _t1, 2)} secs') 
    
tolerance = 0.6    
known_face_list = [
    {
        'name': 'Lee',
        'filename': 'lee.jpg',
        'encode': None,
    },
    {
        'name': 'Pan',
        'filename': 'pan.jpg',
        'encode': None,        
    },
]

test_fn_list = ['lee-t1.jpg', 'lee-t2.jpg', 'pan-t1.jpg', 'pan-t2.jpg']

# load image data by large model of face landmarks
for data in known_face_list:
    img = cv2.imread(data['filename'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data['encode'] = face_recognition.face_encodings(img, model='small')[0]  # use small model of face landmarks
    
known_face_encodes = [data['encode'] for data in known_face_list]
    
# face recognition
for fn in test_fn_list:
    img = cv2.imread(fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    cur_face_locs = face_recognition.face_locations(img)
    cur_face_encodes = face_recognition.face_encodings(img, cur_face_locs, model='small')  # use small model of face landmarks
    
    for cur_face_encode in cur_face_encodes:
        face_distance_list = face_recognition.face_distance(known_face_encodes, cur_face_encode)
        
        min_distance_index = np.argmin(face_distance_list)
        if face_distance_list[min_distance_index] < tolerance:
            result = known_face_list[min_distance_index]['name']
        else:
            result = 'unknown'
            
        distance_with_name_list = [(face_data['name'], round(distance, 4)) for face_data, distance in zip(known_face_list, face_distance_list)]
        print(f'辨識檔案: {fn}, 辨識結果: {result}, 特徵距離: {distance_with_name_list}, 相差: {round(abs(distance_with_name_list[0][1] - distance_with_name_list[1][1]), 4)}')    