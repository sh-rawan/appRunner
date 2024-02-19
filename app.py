from flask import Flask, request, jsonify
import numpy as np
import torch
import json
import whisper

model = whisper.load_model("base")
app = Flask(__name__)

# Load your pre-trained model
# model = load_model()


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Receive audio data
    audio_data = request.files['audio'].read()

    # Convert audio data to numpy array
    audio = np.frombuffer(audio_data, dtype=np.int16).astype(
        np.float32) / 32768.0

    # Perform transcription
    result = model.transcribe(audio, fp16=torch.cuda.is_available(
    ), language=request.form['language'], word_timestamps=True)

    # Return transcription result
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
