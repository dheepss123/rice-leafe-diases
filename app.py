from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model1.h5')

# Mapping kelas
class_mapping = {'Bacterial leaf blight': 0, 'Brown spot': 1, 'Leaf smut': 2}

# Fungsi untuk memprediksi gambar
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = list(class_mapping.keys())[list(class_mapping.values()).index(predicted_class_index)]
    confidence = predictions[0][predicted_class_index]

    return predicted_class, confidence

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Route untuk prediksi gambar yang diunggah
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Simpan file dan prediksi
    file_path = f"./uploads/{file.filename}"
    file.save(file_path)
    predicted_class, confidence = predict_image(file_path)

    return jsonify({"predicted_class": predicted_class, "confidence": float(confidence)})

if __name__ == '__main__':
    app.run(debug=True)
