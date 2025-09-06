from flask import Flask, jsonify, request, render_template
from keras.models import load_model  # TensorFlow is required for Keras to work
import numpy as np
import cv2
import json
import datetime
import time

app = Flask(__name__)
camera = cv2.VideoCapture(1)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

daftar_harga = None
with open("daftar harga.json", "r") as f:
    daftar_harga = json.load(f)

print(daftar_harga)

produk = {
    "nama": "Produk tidak diketahui",
    "harga": 0,
}

@app.route('/')
def home():
    return render_template('indexNew.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/retrieve_data')
def retrieve_data():
    data = None
    with open("checkout.csv", "r") as f:
        data = f.readlines()
    data = [line.strip().split(",") for line in data]
    return data

@app.route('/table')
def table():
    return render_template('tablesNew.html')

def generate_frames():
    global produk
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()

        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index].replace('\n','')
        confidence_score = prediction[0][index]

        produk['nama'] = class_name[2:]
        produk['harga'] = daftar_harga[produk['nama']]['harga'] if produk['nama'] in list(daftar_harga.keys()) else 0

        cv2.putText(frame, f"Class: {class_name[2:]} {np.round(confidence_score * 100)}%", (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1)
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Yield the frame as a byte stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return app.response_class(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_produk', methods=['GET'])
def get_produk():
    global produk
    return jsonify(produk)

@app.route('/save_checkout', methods=['POST'])
def save_data():
    data = request.get_json()
    print(data)
    tanggal_waktu = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open('checkout.csv', 'a') as f:
        for item in data['checkout_data']:
            f.write(f"{tanggal_waktu},{item['nama_produk']},{item['jumlah']},{item['harga_satuan']},{item['total_harga']}\n")
    return jsonify({"status": "success", "data": data})

if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)