# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

CLASS_NAMES = ['Parasitized', 'Uninfected']
model       = None
MODEL_ERROR = None

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
try:
    import tensorflow as tf
    from PIL import Image
    print("✅ TensorFlow imported")
except Exception as e:
    MODEL_ERROR = f"Import failed: {e}"
    print(f"❌ {MODEL_ERROR}")

if MODEL_ERROR is None:
    try:
        current_dir  = os.getcwd()
        keras_files  = [f for f in os.listdir(current_dir) if f.endswith('.keras')]

        if keras_files:
            MODEL_PATH = os.path.join(current_dir, keras_files[0])
            model      = tf.keras.models.load_model(MODEL_PATH, compile=False)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            print(f"✅ Loaded: {keras_files[0]}")
            print(f"   Input: {model.input_shape} | Output: {model.output_shape}")
        else:
            MODEL_ERROR = "No .keras file found in folder"
            print(f"❌ {MODEL_ERROR}")

    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"❌ {MODEL_ERROR}")


# ─────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────
def prepare_image(image_file):
    if model is None:
        return None
    try:
        img = Image.open(image_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize to match model input (128x128)
        img_size  = (model.input_shape[1], model.input_shape[2])
        img       = img.resize(img_size)
        img_array = np.array(img, dtype='float32') / 255.0          # Normalize
        return np.expand_dims(img_array, axis=0)                     # Add batch dim

    except Exception as e:
        print(f"Image error: {e}")
        return None


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html', model_loaded=model is not None, error=MODEL_ERROR)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        # Preprocess & predict
        img_array = prepare_image(file)
        if img_array is None:
            return jsonify({'success': False, 'error': 'Image processing failed'}), 500

        prediction = model.predict(img_array, verbose=0)
        prob       = float(prediction[0][0])

        # prob close to 0 → Parasitized, close to 1 → Uninfected
        if prob > 0.5:
            pred_class = 1   # Uninfected
            confidence = prob * 100
        else:
            pred_class = 0   # Parasitized
            confidence = (1 - prob) * 100

        result = CLASS_NAMES[pred_class]
        print(f"Result: {result} | Confidence: {confidence:.2f}%")

        return jsonify({
            'success'               : True,
            'prediction'            : result,
            'confidence'            : round(confidence, 2),
            'parasitized_probability': round((1 - prob) * 100, 2),
            'uninfected_probability' : round(prob * 100, 2),
            'timestamp'             : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'running', 'model_loaded': model is not None})


if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    print("\n" + "="*50)
    print(f"  {'✅' if model else '❌'} Model: {'LOADED' if model else 'NOT LOADED'}")
    if model:
        print(f"  Input: {model.input_shape}")
    print(f"  🌐 http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)