import cv2
import numpy as np
from flask import Flask, request
from main import preprocessing_encode, recognize
import time

app = Flask(__name__)

    
# 先載入將人臉特徵encode
def load():
    start_time = time.time()  # 記錄開始時間
    print('execute load funcion ')
    global known_face_list, known_face_encodes
    known_face_list,known_face_encodes=preprocessing_encode()
    end_time = time.time()  # 記錄結束時間
    execution_time = end_time - start_time  # 計算執行時間
    print(f'Load function executed in {execution_time:.2f} seconds')


@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/predict',methods=['POST'])
def predict():
    filestr = request.files['image'].read()
    npimg = np.frombuffer(filestr, np.uint8)
    image  = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    result= recognize(image,known_face_list,known_face_encodes,tolerance=0.6)
    return result

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
if __name__ == "app":
    print("Loading model... Please wait.")
    load()
    app.run(debug=True,host="0.0.0.0", port=8080,  use_reloader=True)