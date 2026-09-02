import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CropGuard AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 20px;
    color: #888;
    margin-bottom: 25px;
}

.section-title {
    font-size: 30px;
    font-weight: 700;
}

.info-card {
    padding: 20px;
    border-radius: 12px;
    background-color: rgba(100, 100, 100, 0.12);
    margin-bottom: 15px;
}

.footer {
    text-align: center;
    color: #888;
    padding-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "models/cropguard_resnet18.pth"


@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    class_to_index = checkpoint["class_to_index"]

    index_to_class = {
        value: key
        for key, value in class_to_index.items()
    }

    num_classes = len(class_to_index)

    model = models.resnet18(weights=None)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        num_classes
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    model.eval()

    return model, index_to_class, device


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# 🌱 CropGuard AI")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔬 AI Disease Detection",
            "📊 Crop Health Risk",
            "🌦️ Weather & Environment",
            "💡 Recommendations",
            "📜 Analysis History",
            "ℹ️ About CropGuard"
        ]
    )

    st.markdown("---")

    st.caption("Intelligent Crop Health Monitoring")


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">🌱 CropGuard AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Intelligent Crop Health Monitoring Platform'
        '</div>',
        unsafe_allow_html=True
    )

    st.success(
        "🌾 Helping farmers detect, understand, and respond to crop health problems."
    )

    st.markdown("## 🚜 Welcome to CropGuard")

    st.write(
        """
        CropGuard AI is designed to help farmers identify crop diseases
        at an early stage using artificial intelligence.

        Our vision goes beyond disease detection. We aim to combine
        AI-based diagnosis with weather, environmental conditions,
        crop risk analysis, and farmer-friendly recommendations.
        """
    )

    st.markdown("## 🔍 What CropGuard Can Do")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### 🔬 AI Detection

            Analyze crop leaf images using deep learning
            and identify possible disease conditions.
            """
        )

    with col2:
        st.markdown(
            """
            ### 📊 Risk Analysis

            Combine disease information with environmental
            conditions to estimate crop health risk.
            """
        )

    with col3:
        st.markdown(
            """
            ### 💡 Recommendations

            Provide practical information to help farmers
            take timely preventive action.
            """
        )

    st.markdown("---")

    st.markdown("## 🌾 Current AI Coverage")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Crops", "4")

    with col2:
        st.metric("AI Classes", "21")

    with col3:
        st.metric("Model", "ResNet18")

    with col4:
        st.metric("Validation Accuracy", "98.69%")

    st.markdown("---")

    st.info(
        "🚀 More crops, diseases, weather intelligence, pest detection "
        "and farmer-focused features will be added in future versions."
    )


# ============================================================
# AI DISEASE DETECTION
# ============================================================

elif page == "🔬 AI Disease Detection":

    st.markdown("## 🔬 AI Crop Disease Detection")

    st.write(
        "Upload a clear crop leaf image and CropGuard will analyze it using AI."
    )

    # Device information
    if torch.cuda.is_available():
        st.success(
            f"🟢 GPU Ready — {torch.cuda.get_device_name(0)}"
        )
    else:
        st.info("🔵 Running on CPU")

    uploaded_file = st.file_uploader(
        "📤 Upload a crop leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Leaf",
            use_container_width=True
        )

        if st.button(
            "🔍 Analyze Leaf",
            type="primary"
        ):

            with st.spinner("AI is analyzing the leaf..."):

                model, index_to_class, device = load_model()

                image_tensor = transform(image)
                image_tensor = image_tensor.unsqueeze(0)
                image_tensor = image_tensor.to(device)

                with torch.no_grad():

                    outputs = model(image_tensor)

                    probabilities = torch.softmax(
                        outputs,
                        dim=1
                    )

                    confidence, predicted = torch.max(
                        probabilities,
                        1
                    )

                predicted_class = index_to_class[
                    predicted.item()
                ]

                confidence_value = confidence.item() * 100

            st.markdown("---")

            st.markdown("## 🌱 CropGuard Result")

            st.success(
                f"Prediction: {predicted_class}"
            )

            st.metric(
                "AI Confidence",
                f"{confidence_value:.2f}%"
            )

            if "healthy" in predicted_class.lower():

                st.success(
                    """
                    🌿 The leaf appears healthy.

                    Continue regular crop monitoring and maintain
                    appropriate agricultural practices.
                    """
                )

            else:

                st.warning(
                    """
                    ⚠️ Possible disease detected.

                    Consider monitoring affected plants,
                    removing severely affected leaves where appropriate,
                    and consulting an agricultural expert for treatment.
                    """
                )

            # Save result during current session
            if "history" not in st.session_state:
                st.session_state.history = []

            st.session_state.history.append({
                "Time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "Prediction": predicted_class,
                "Confidence": f"{confidence_value:.2f}%"
            })


# ============================================================
# CROP HEALTH RISK
# ============================================================

elif page == "📊 Crop Health Risk":

    st.markdown("## 📊 Crop Health Risk Analysis")

    st.write(
        """
        Disease detection is only one part of crop health.
        CropGuard is designed to combine disease information
        with environmental conditions to estimate crop health risk.
        """
    )

    st.info(
        "🚧 Risk analysis engine is currently under development."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Disease Status", "Pending")

    with col2:
        st.metric("Environmental Risk", "Pending")

    with col3:
        st.metric("Overall Risk", "Pending")

    st.markdown("### 🔮 Planned Risk Factors")

    st.write("""
    - 🌡️ Temperature
    - 💧 Humidity
    - 🌧️ Rainfall
    - 📍 Location
    - 🌱 Crop condition
    - 🦠 Detected disease
    """)

    st.warning(
        "This module will be connected to real-time weather and risk analysis later."
    )


# ============================================================
# WEATHER
# ============================================================

elif page == "🌦️ Weather & Environment":

    st.markdown("## 🌦️ Weather & Environment")

    st.write(
        """
        Weather conditions can influence crop health and disease development.
        CropGuard will use environmental information as part of its future
        crop-risk analysis.
        """
    )

    st.info(
        "🚧 Real-time weather integration is coming in the next development stage."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Temperature", "-- °C")

    with col2:
        st.metric("Humidity", "-- %")

    with col3:
        st.metric("Rainfall", "-- mm")

    with col4:
        st.metric("Disease Risk", "--")


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "💡 Recommendations":

    st.markdown("## 💡 Farmer Recommendations")

    st.write(
        """
        CropGuard aims to convert AI results into simple,
        practical information that farmers can understand and act upon.
        """
    )

    st.markdown("### 🌱 Example Recommendation")

    st.warning(
        """
        Possible disease detected.

        Recommended actions:

        • Monitor nearby plants regularly  
        • Remove severely affected leaves where appropriate  
        • Avoid conditions that promote excessive leaf wetness  
        • Follow recommended agricultural practices  
        • Consult an agricultural expert when necessary
        """
    )

    st.info(
        "Future versions will generate recommendations based on the detected crop, disease, weather, and risk level."
    )


# ============================================================
# HISTORY
# ============================================================

elif page == "📜 Analysis History":

    st.markdown("## 📜 Analysis History")

    if "history" not in st.session_state:
        st.session_state.history = []

    if len(st.session_state.history) == 0:

        st.info(
            "No analysis has been performed in this session yet."
        )

    else:

        st.dataframe(
            st.session_state.history,
            use_container_width=True
        )

        if st.button("🗑️ Clear History"):

            st.session_state.history = []

            st.rerun()


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About CropGuard":

    st.markdown("## ℹ️ About CropGuard AI")

    st.write(
        """
        CropGuard AI is an AI-powered crop health monitoring project
        developed to support farmers with early disease detection
        and intelligent crop-health insights.
        """
    )

    st.markdown("### 🧠 Current Technology")

    st.write("""
    - Python
    - PyTorch
    - TorchVision
    - ResNet18
    - Streamlit
    - PlantVillage Dataset
    """)

    st.markdown("### 🌾 Current Prototype")

    st.write("""
    CropGuard currently supports:

    **Apple • Grape • Potato • Tomato**

    across **21 disease and healthy classes**.
    """)

    st.markdown("### 🚀 Future Vision")

    st.write("""
    Our long-term goal is to build a complete intelligent crop-health
    platform with:

    - More crops and diseases
    - Pest detection
    - Weather-based risk prediction
    - Location intelligence
    - Multilingual farmer support
    - Unknown-crop detection
    - Expert-assisted recommendations
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="footer">'
    '🌱 CropGuard AI • Intelligent Crop Health Monitoring'
    '</div>',
    unsafe_allow_html=True
)