from flask import Flask, request, render_template, send_from_directory
import os
import sys
from model.gradcam import run_gradcam



# Add the backend directory to the path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)



app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            try:
                # Run Grad-CAM and get results
                gradcam_path, label, confidence = run_gradcam(filepath)
                
                # Clean up uploaded file
                os.remove(filepath)
                
                return render_template('result.html', 
                                     label=label, 
                                     confidence=round(confidence, 2),
                                     gradcam_filename=os.path.basename(gradcam_path))
            except Exception as e:
                return f"Error processing image: {str(e)}"
    
    return render_template('index.html')

@app.route('/outputs/<filename>')
def output_file(filename):
    output_dir = os.path.join(PROJECT_ROOT, 'storage', 'outputs')
    return send_from_directory(output_dir, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)