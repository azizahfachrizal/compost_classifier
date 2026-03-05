import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# --- 1. MODEL DEFINITIONS (From your Notebook) ---

class CompostCNN(nn.Module):
    def __init__(self, num_classes):
        super(CompostCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class CompostMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CompostMLP, self).__init__()
        flat_features = 3 * input_size * input_size
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.network(x)

def CompostMOBILENET(num_classes):
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

def CompostVIT(num_classes):
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model

# --- 2. CONFIGURATIONS ---
CLASS_NAMES = ['immature', 'mature']
IMAGE_SIZE = 224

# Normalization 1 (Used for CNN/MLP in your code)
NORM_1 = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
# Normalization 2 (Used for MobileNet/ViT in your code)
NORM_2 = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

# --- 3. MODEL LOADING LOGIC ---
@st.cache_resource
def load_selected_model(model_choice):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # if model_choice == "Vision Transformer (ViT)":
    #     model = CompostVIT(len(CLASS_NAMES))
    #     weight_file = "../models/v4_25epochs/compost_vit_model.pth"
    #     norm = NORM_2
    if model_choice == "MobileNetV3":
        model = CompostMOBILENET(len(CLASS_NAMES))
        weight_file = "./models/v4_25epochs/compost_mobilenet_model.pth"
        norm = NORM_2
    elif model_choice == "Custom CNN":
        model = CompostCNN(len(CLASS_NAMES))
        weight_file = "./models/v4_25epochs/compost_cnn_model.pth"
        norm = NORM_1
    # else: # MLP
    #     model = CompostMLP(IMAGE_SIZE, len(CLASS_NAMES))
    #     weight_file = "../models/v4_25epochs/compost_mlp_model.pth"
    #     norm = NORM_1

    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.to(device)
    model.eval()
    return model, device, norm

# --- 4. UI SETUP ---
st.set_page_config(page_title="Compost Model Selector", layout="wide")
st.sidebar.title("Settings")
model_option = st.sidebar.selectbox(
    "Select Model to Use:",
    ("MobileNetV3", "Custom CNN") #"Vision Transformer (ViT)", "MLP"
)

st.title("🌱 Compost Maturity Classifier")
st.write(f"Currently using: **{model_option}**")

try:
    model, device, norm_values = load_selected_model(model_option)
    
    uploaded_file = st.file_uploader("Upload Compost Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Input Image', use_container_width=True)
            
        with col2:
            if st.button('Predict'):
                # Preprocessing
                transform = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_values["mean"], std=norm_values["std"])
                ])
                
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = F.softmax(output, dim=1)
                    conf, pred = torch.max(prob, 1)
                    
                label = CLASS_NAMES[pred.item()]
                confidence = conf.item() * 100
                
                st.subheader("Prediction Details")
                st.metric("Result", label.upper())
                st.metric("Confidence", f"{confidence:.2f}%")
                st.progress(confidence / 100)

except Exception as e:
    st.error(f"Error loading model: Make sure you have the correct .pth files in your directory.")
    st.info("Required file: compost_VIT.pth, compost_MOBILENET.pth, etc.")