from chroma import query, storeToDb
import cv2
import numpy as np
from flask import Flask, request,jsonify
from main import encode_current_face, load_encodings, preprocessing_encode, recognize
import time
import os

app = Flask(__name__)
    
# 先載入將人臉特徵encode
def load():
    start_time = time.time()  # 記錄開始時間
    print('execute load funcion ')
    json_file= 'singer-map-with-encodings.json'
    global known_face_list, known_face_encodes
    if not os.path.exists(json_file):
       print(f'{json_file} not found. Running preprocessing_encode...')
       known_face_list, known_face_encodes = preprocessing_encode()
    else:
        print(f'{json_file} found. Loading encodings...')
        known_face_list, known_face_encodes = load_encodings(json_file)
    
    storeToDb(known_face_list)
    end_time = time.time()  # 記錄結束時間
    execution_time = end_time - start_time  # 計算執行時間
    
    print(f'Load function executed in {execution_time:.2f} seconds')

print("Loading model... Please wait.")
load()

@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/predict',methods=['POST'])
def predict():
    filestr = request.files['image'].read()
    npimg = np.frombuffer(filestr, np.uint8)
    image  = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # result= recognize(image,known_face_list,known_face_encodes,tolerance=0.6)
    encode_face_list = encode_current_face(image)
    db_result= query(encode_face_list)
    distance=db_result['distances'][0][0]
    # 距離差異
    tolerance=0.3
    if(distance>tolerance):
        return jsonify({"result":"unknown","distance":distance})
    else:
        name_value = db_result['metadatas'][0][0]['name']
        return jsonify({"result": name_value,"distance":distance})

@app.route('/help', methods=['GET'])
def helpfunc():
    s = "I can recognize "
    # 使用字典来去重
    unique_data = {}
    for item in known_face_list:
        if item['name'] not in unique_data:
            unique_data[item['name']] = item

    # 转换回列表
    filtered_list = list(unique_data.values())
    for i in range(len(filtered_list)):
        if i==len(filtered_list)-1:
                s += " and "+filtered_list[i]['name']
        else:
                s += " "+filtered_list[i]['name']+","
    return s

print('__name__ is ',__name__)
if __name__ == "__main__":
    print('will start to run app')
    app.run(debug=True, host='0.0.0.0', port=8080)