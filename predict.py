import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ==========================================
# 1. SETTINGS
# ==========================================

MODEL_PATH = "models/cropguard_resnet18.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. LOAD MODEL DATA
# ==========================================

print("Loading CropGuard model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

class_to_index = checkpoint["class_to_index"]

index_to_class = {
    index: class_name
    for class_name, index in class_to_index.items()
}


# ==========================================
# 3. CREATE RESNET18
# ==========================================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_to_index)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ==========================================
# 4. IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 5. ASK FOR IMAGE
# ==========================================

image_path = input(
    "\nEnter the path of a leaf image: "
)


# ==========================================
# 6. LOAD IMAGE
# ==========================================

image = Image.open(
    image_path
).convert("RGB")


image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(DEVICE)


# ==========================================
# 7. AI PREDICTION
# ==========================================

with torch.no_grad():

    outputs = model(image_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted_index = torch.max(
        probabilities,
        1
    )


# ==========================================
# 8. RESULT
# ==========================================

predicted_class = index_to_class[
    predicted_index.item()
]

confidence_percentage = (
    confidence.item() * 100
)


print("\n===================================")
print("        🌱 CropGuard Result")
print("===================================")

print(
    "Prediction:",
    predicted_class
)

print(
    "Confidence:",
    f"{confidence_percentage:.2f}%"
)

print("===================================")