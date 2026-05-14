import os
import tempfile
from flask import Flask, request, jsonify, render_template
from parser import parse_cv

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'cv' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['cv']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF or DOCX files are supported'}), 400

    suffix = '.' + file.filename.rsplit('.', 1)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = parse_cv(tmp_path, use_llama=True)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 422
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {e}'}), 500
    finally:
        os.unlink(tmp_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
