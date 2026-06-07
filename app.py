import os
import torch
import requests
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms
import gc

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')


encoder = None
decoder = None
device = torch.device("cpu") 

# Google Drive Direct Download Helper
def download_file_from_google_drive(file_id, destination):
    if os.path.exists(destination):
        return
    print(f"Downloading {destination} from Google Drive...", flush=True)
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    print(f"Finished downloading {destination}.", flush=True)

def load_models_lazy():
    global encoder, decoder
    if encoder is not None and decoder is not None:
        return # Models are already loaded in memory

    print("Initializing models for the first time...", flush=True)
    
    gc.collect() 

    encoder_path = 'vgg_normalized.pth'
    decoder_path = 'decoder_2.pth'

    download_file_from_google_drive('1CKxQm0W8GmB2NIg8whmgxrTgaCP5-sVP', encoder_path)
    download_file_from_google_drive('1GAWG6_ytKp07wY8QMIac9_JK-7m_mkLU', decoder_path)

    encoder = VGGEncoder(encoder_path).to(device)
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    encoder.eval()
    decoder.eval()
    
    gc.collect()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_image, alpha):
    load_models_lazy() # Ensure models are downloaded and loaded right before execution

    content_transform = transforms.Compose([
        transforms.Resize(200), # Dropped slightly from 256 to 200 for absolute safety
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(200), # Dropped slightly to 200 for absolute safety
        transforms.ToTensor()
    ])
    
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)
        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)
        
        # Free up variables and aggressively flush Python's Garbage Collection
        del content_feats, style_feats, stylized_feats
        gc.collect() 

    return stylized_image

def save_image(image, path):
    image = image.cpu().clone().squeeze(0).clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data

        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            
            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')

                alpha = float(form.alpha.data)
                stylized_image = style_transfer(content_image, style_image, alpha)

                result_filename = 'stylized_' + content_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)
                
                result_image = result_filename
            except Exception as e:
                error = str(e)
    else:
        if not content_filename:
            error = 'Please upload content image'
        if not style_filename:
            error = 'Please upload style image'

    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
                           style_image=style_filename, error=error)

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)