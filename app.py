from flask import Flask, request, render_template
import pickle
import numpy as np
import io
import imageio.v3 as iio  # This reads images without using PIL or OpenCV

# Load the trained model
model_path = 'model1.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # **Step 1: Check if File is Uploaded**
        if 'file' not in request.files:
            return render_template('index.html', prediction_text="No file uploaded! Please try again.")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', prediction_text="No file selected!")

        # **Step 2: Read Image Properly**
        file_bytes = file.read()  # Read file bytes
        image = iio.imread(io.BytesIO(file_bytes))  # Decode into image array

        # **Step 3: Ensure Image is Resized to (256, 256, 3)**
        image = np.resize(image, (256, 256, 3))  # Resize to expected input shape
        image = image / 255.0  # Normalize pixel values

        # **Step 4: Expand Dimensions for Model Input**
        image = np.expand_dims(image, axis=0)  # Shape becomes (1, 256, 256, 3)

        # **Step 5: Predict Using Model**
        prediction = model.predict(image)

        # **Step 6: Format Output**
        output = 'Class 1' if prediction[0] > 0.5 else 'Class 0'

        return render_template('index.html', prediction_text=f'Prediction: {output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
