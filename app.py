import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="CropGuard AI",
    page_icon="🌱",
    layout="centered"
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        "models/cropguard_resnet18.pth",
        map_location=device,
        weights_only=False
    )

    class_to_index = checkpoint["class_to_index"]

    index_to_class = {
        index: class_name
        for class_name, index in class_to_index.items()
    }

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_to_index)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    return model, index_to_class, device


model, index_to_class, device = load_model()


# ==========================================
# IMAGE TRANSFORMATION
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
# HEADER
# ==========================================

st.title("🌱 CropGuard AI")

st.subheader(
    "AI-Powered Crop Disease Detection"
)

st.write(
    "Upload a crop leaf image and CropGuard "
    "will analyze it using artificial intelligence."
)


# ==========================================
# SYSTEM STATUS
# ==========================================

if torch.cuda.is_available():

    st.success(
        "🟢 GPU Ready — NVIDIA GeForce GTX 1650"
    )

else:

    st.info(
        "🔵 Running on CPU"
    )


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📸 Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# ANALYZE IMAGE
# ==========================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )

    if st.button("🔍 Analyze Leaf"):

        image_tensor = transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device)


        # ==================================
        # AI PREDICTION
        # ==================================

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


        predicted_class = index_to_class[
            predicted_index.item()
        ]

        confidence_percentage = (
            confidence.item() * 100
        )


        # ==================================
        # RESULT
        # ==================================

        st.divider()

        st.header("🌱 CropGuard Result")

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.metric(
            "AI Confidence",
            f"{confidence_percentage:.2f}%"
        )


        # ==================================
        # RECOMMENDATION
        # ==================================

        if "healthy" in predicted_class.lower():

            st.info(
                "🌿 The leaf appears healthy. "
                "Continue regular monitoring, "
                "proper irrigation, and good crop care."
            )

        else:

            st.warning(
                "⚠️ Possible disease detected. "
                "Consider isolating affected plants, "
                "removing severely affected leaves, "
                "and consulting an agricultural expert "
                "for appropriate treatment."
            )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "CropGuard AI • Intelligent Crop Health Monitoring"
)