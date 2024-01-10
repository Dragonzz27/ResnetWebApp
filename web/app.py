from flask import Flask, render_template, request, redirect, url_for,send_from_directory
import os
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image

torch.cuda.is_available = lambda : False
device = torch.device('cpu')

model_path = os.path.join('..', 'model', 'demo_resnet18')
model = torch.load(model_path, map_location=device)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg','jpeg','png','gif'}
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def MyPredict(image_path):
    label_list = ['Biological','Fibres','Films_Coated_Surface','MEMS_devices_and_electrodes','Nanowires','Particles','Patterned_surface','Porous_Sponge','Powder','Tips']
    # Your data transformation
    normal_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model.eval()
    # Load and transform the single image
    
    image = Image.open(image_path)
    input_image = normal_transforms(image)
    input_image = input_image.unsqueeze(0)  # Add a batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(input_image.cpu())

    # Get the predicted label
    _, predicted_label = torch.max(output, 1)

    print(f"Predicted label:  {label_list[predicted_label.item()]}")
    
    return label_list[predicted_label.item()]
 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        result = MyPredict(file_path)
        return render_template('index.html', filename=filename, result=result)

@app.route('/uploads/<filename>') # uploads/L6_00f153d186dbf5fa02e0558e47537fde.jpg
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)