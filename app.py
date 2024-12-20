# Guardar como app.py
from flask import Flask, request, jsonify
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model('/saved_model/img-class-cnn-fine.keras')  # Usar el nuevo formato .keras

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = Image.open(file)
        img = img.resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        pred = model.predict(img_array)
        class_idx = np.argmax(pred[0])
        confidence = float(pred[0][class_idx])
        
        return jsonify({
            'success': True,
            'class': int(class_idx),
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(port=5001)  # Usar un puerto diferente si 5000 está ocupado