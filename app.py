from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Define audio parameters (must match the ones used during training)
SAMPLE_RATE = 16000
DURATION = 4
N_MELS = 128
MAX_TIME_STEPS = 120

# Load the model only when needed
def load_model():
    return tf.keras.models.load_model("resnet50_audio_classifier.keras")

def preprocess_audio(file_path):
    """Load an audio file and convert it to a Mel spectrogram."""
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    # Compute Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Pad or truncate spectrogram
    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

    # Normalize spectrogram
    mel_spectrogram = (mel_spectrogram - np.min(mel_spectrogram)) / (np.max(mel_spectrogram) - np.min(mel_spectrogram))

    # Expand dimensions to match ResNet50 input
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
    mel_spectrogram = np.repeat(mel_spectrogram, 3, axis=-1)  # Convert to 3-channel

    return np.expand_dims(mel_spectrogram, axis=0).astype(np.float32)  # Add batch dimension

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file temporarily
    temp_file_path = "temp_audio." + file.filename.split(".")[-1]  # Preserve original extension
    file.save(temp_file_path)

    # Preprocess the audio
    try:
        input_data = preprocess_audio(temp_file_path)
    except Exception as e:
        return jsonify({"error": f"Error preprocessing audio: {str(e)}"}), 500

    # Load the model and make a prediction
    try:
        model = load_model()
        prediction = model.predict(input_data)
        predicted_label = np.argmax(prediction)
        confidence = float(np.max(prediction))
    except Exception as e:
        return jsonify({"error": f"Error making prediction: {str(e)}"}), 500

    # Convert label to text
    label_map = {0: "Fake", 1: "Real"}
    result = label_map[predicted_label]

    # Clean up the temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    return jsonify({"prediction": result, "confidence": confidence})

if __name__ == "__main__":
    # Bind to 0.0.0.0 and use the port specified by Render
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=False)