from flask import Flask, render_template, request, url_for
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/')
def landing():
    return render_template('landingpage.html')

@app.route('/predict_vit', methods=['GET', 'POST'])
def ViT_Model():
   
    if request.method == 'POST' and 'image' in request.files:
        image_file = request.files['image']
        image = Image.open(image_file)
        
        feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]
        prediction = f"Predicted class: {predicted_class}"
        
        return render_template('vitresults.html', accuracy=prediction)
    
    return "No image uploaded", 400

@app.route('/predict_yolo', methods=['GET', 'POST'])
def Yolo_Model():
    if request.method == 'POST':
        # Load a pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')

        # Assuming you have access to the uploaded image file
        uploaded_image = request.files['image']

        # Save the uploaded image to a file in the static directory
        uploaded_image_path = 'static/images/uploaded_image.png'
        uploaded_image.save(uploaded_image_path)

        # Run inference on the uploaded image
        results = model(uploaded_image_path, verbose=False)

        # Save the results as an image in the static directory
        for i, r in enumerate(results):
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            result_image_path = f'static/images/results_{i}.png'
            im.save(result_image_path)  # save image in the static directory

        display_image_path = '/static/images/results_0.png' 

        return render_template('yoloresults.html', image_path=display_image_path)

    return render_template('yoloresults.html')  # Redirect to upload page if not POST request
if __name__ == '__main__':
    app.run(debug=True)