import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.conv11 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv12 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv13 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(128 * 14 * 14, 400)  
        self.fc2 = nn.Linear(400, num_classes) 

    def forward(self, x):
        x = self.conv11(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv12(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv13(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(x.shape[0], -1)  
        
        x = self.fc1(x)  
        x = F.relu(x)
        
        x = self.fc2(x)  
        return x

app = Flask(__name__)

try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(0.2),
    transforms.RandomRotation(10),
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.0)),
    transforms.ToTensor(),
])

class_labels = {
    0: "Accessories",
    1: "Apparel",
    2: "Footwear",
    3: "FreeItems_SportingGoods",
    4: "PersonalCare"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']

    try:
        img = Image.open(file)
        img_transformed = transform(img)
        img_transformed = img_transformed.unsqueeze(0)  

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        img_transformed = img_transformed.to(device)

        model.eval()
        with torch.no_grad():
            output = model(img_transformed)
            _, predicted_class = torch.max(output, 1)

        class_label = class_labels[predicted_class.item()]

        return jsonify({"predicted_class": class_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
