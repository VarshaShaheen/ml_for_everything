import os
import tempfile
import whisper
from flask import Flask, request, render_template, jsonify
from io import BytesIO

from werkzeug.utils import secure_filename

from text_summarisation import summarize

app = Flask(__name__)

# Set up directory for file uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def index():
    """Render the main form page on GET request."""
    return render_template('index.html')


@app.route('/summarize_text', methods=['POST'])
def summ_text():
    """Handle text summarization requests."""
    input_text = request.form['text']
    summary = summarize(input_text)
    return render_template('index.html', corrected_text=summary)


@app.route('/summarize_audio_file', methods=['POST'])
def summ_audio():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    # Save the file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, secure_filename(file.filename))
    file.save(temp_path)

    try:
        text = audio_to_text_direct(temp_path)
        summary = summarize(text)
        os.remove(temp_path)  # Clean up the temp file
        os.rmdir(temp_dir)  # Clean up the temp directory
        return render_template('index.html', corrected_file_text=summary, corrected_text=text)
    except Exception as e:
        os.remove(temp_path)  # Ensure temp file is removed even if there's an error
        os.rmdir(temp_dir)  # Ensure temp directory is removed even if there's an error
        return jsonify(error=str(e)), 500


def audio_to_text_direct(file_path):
    """Convert audio file to text using the Whisper model."""
    model_whisper = whisper.load_model("medium.en")
    result = model_whisper.transcribe(file_path)
    return result['text']




if __name__ == '__main__':
    app.run(debug=True)
