from flask import Flask, render_template, request
import numpy as np
import os
import cv2
import tensorflow

model_path = os.path.join(os.path.dirname(__file__), "model.h5")
model = tensorflow.keras.models.load_model(model_path, compile=False, safe_mode=False)

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PREDEFINED_IMAGES_FOLDER = 'static/predefined'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', "webp"}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDEFINED_IMAGES_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def form():
    predefined_images = [f for f in os.listdir(PREDEFINED_IMAGES_FOLDER) if allowed_file(f)]
    return render_template('index.html', predefined_images=predefined_images)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        file = request.files['file']
        
        if file.filename == '' or not allowed_file(file.filename):
            return "Invalid file type", 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
    elif 'predefined_image' in request.form:
        file_path = os.path.join(PREDEFINED_IMAGES_FOLDER, request.form['predefined_image'])
    else:
        return "No file selected", 400

    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = image.astype('float32') / 255.0  
    image = np.expand_dims(image, axis=0) 

    result = float(model.predict(image))
    prediction = "Organic" if result < 0.5 else "Recyclable"

    return render_template('predict.html', result=prediction, image_path=file_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860)) 
    app.run(host="0.0.0.0", port=port)
