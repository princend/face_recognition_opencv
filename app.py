import cv2
import numpy as np
from flask import Flask, request
from main import preprocessing_encode, recognize

app = Flask(__name__)

# 先載入將人臉特徵encode
def load():
    global known_face_list, known_face_encodes
    known_face_list,known_face_encodes=preprocessing_encode()
    

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

if __name__ == "__main__":
    print("Loading model... Please wait.")
    load()
    app.run(debug=True, use_reloader=False, threaded=False)