from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("resnet50_audio_classifier.keras")

# Define audio parameters (must match the ones used during training)
SAMPLE_RATE = 16000
DURATION = 4
N_MELS = 128
MAX_TIME_STEPS = 120

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
    file_path = "temp_audio.flac"
    file.save(file_path)

    # Preprocess the audio
    input_data = preprocess_audio(file_path)

    # Make a prediction
    prediction = model.predict(input_data)
    predicted_label = np.argmax(prediction)

    # Convert label to text
    label_map = {0: "Fake", 1: "Real"}
    result = label_map[predicted_label]

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True , port = 5001)