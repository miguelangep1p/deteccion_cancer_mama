import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import joblib
import numpy as np
import xgboost as xgb
import pandas as pd

# =======================
# CONFIGURACIÓN MODELOS
# =======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Modelo PyTorch (CNN) ---
model_cnn = torch.load("model_best_resnet50_breast_binary.pth", 
                       map_location=device, weights_only=False)
model_cnn.eval()

imagenet_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    imagenet_normalization
])

class_names = ["Benign", "Malignant"]

# --- Modelo XGBoost (Tabular) ---
scaler = joblib.load("scaler_xgboost.pkl")
booster = xgb.Booster()
booster.load_model("best_xgboost_model.json")

# =======================
# MULTILENGUAJE
# =======================
languages = {
    "Español": {
        "title": "🩺 Plataforma de Predicción Médica",
        "subtitle": "Bienvenido al sistema de **predicción de mamografías y análisis tabular**. Seleccione el modelo en el menú lateral para comenzar.",
        "sidebar_title": "⚙️ Configuración",
        "sidebar_option": "Selecciona el modelo:",
        "cnn_title": "🔍 Clasificador de Mamografías (Benigno vs Maligno)",
        "cnn_desc": "Sube una **imagen de mamografía** para obtener la predicción del modelo CNN.",
        "upload_img": "📂 Sube una imagen (JPG/PNG)",
        "prediction": "Predicción",
        "benign": "✅ Confianza Benigno",
        "malignant": "⚠️ Confianza Maligno",
        "xgb_title": "🧠 Clasificador Tabular (XGBoost)",
        "xgb_manual": "### 🔹 Opción 1: Ingresar manualmente los **30 parámetros numéricos**",
        "xgb_csv": "### 🔹 Opción 2: Subir un archivo CSV con varias filas de features",
        "upload_csv": "📂 Sube tu archivo CSV",
        "preview": "📋 Vista previa de los datos",
        "results": "📊 Resultados de Predicción",
        "error_pred": "❌ Error durante la predicción",
        "error_csv": "❌ Error procesando el archivo"
    },
    "English": {
        "title": "🩺 Medical Prediction Platform",
        "subtitle": "Welcome to the **mammography prediction and tabular analysis system**. Select the model in the sidebar to start.",
        "sidebar_title": "⚙️ Settings",
        "sidebar_option": "Choose model:",
        "cnn_title": "🔍 Mammography Classifier (Benign vs Malignant)",
        "cnn_desc": "Upload a **mammogram image** to get the prediction from the CNN model.",
        "upload_img": "📂 Upload an image (JPG/PNG)",
        "prediction": "Prediction",
        "benign": "✅ Benign Confidence",
        "malignant": "⚠️ Malignant Confidence",
        "xgb_title": "🧠 Tabular Classifier (XGBoost)",
        "xgb_manual": "### 🔹 Option 1: Manually enter the **30 numerical parameters**",
        "xgb_csv": "### 🔹 Option 2: Upload a CSV file with multiple rows of features",
        "upload_csv": "📂 Upload your CSV file",
        "preview": "📋 Data Preview",
        "results": "📊 Prediction Results",
        "error_pred": "❌ Error during prediction",
        "error_csv": "❌ Error processing file"
    }
    
}
feature_names = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness",
    "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
    # ... hasta llegar a 30
]

# =======================
# INTERFAZ STREAMLIT
# =======================
st.set_page_config(page_title="Predicción Mamografías", 
                   page_icon="🩺", 
                   layout="wide")

# --- Sidebar ---
st.sidebar.markdown("---")
lang_choice = st.sidebar.selectbox("🌐 Idioma / Language", ["Español", "English"])
txt = languages[lang_choice]

st.sidebar.title(txt["sidebar_title"])
option = st.sidebar.radio(txt["sidebar_option"], 
                          ["CNN - Clasificación de Imágenes", "XGBoost - Clasificación Tabular"])

# --- Encabezado principal ---
st.title(txt["title"])
st.markdown(txt["subtitle"])

# =======================
# SECCIÓN: CNN (IMÁGENES)
# =======================
if option == "CNN - Clasificación de Imágenes":
    st.header(txt["cnn_title"])
    st.markdown(txt["cnn_desc"])

    uploaded_file = st.file_uploader(txt["upload_img"], 
                                     type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="🖼️", use_container_width=True)

        with col2:
            img_tensor = val_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model_cnn(img_tensor)
                _, pred = torch.max(outputs, 1)
                probas = torch.softmax(outputs, dim=1)[0]

            pred_class = class_names[pred.item()]
            prob_benign = probas[0].item() * 100
            prob_malignant = probas[1].item() * 100

            st.metric(label=txt["prediction"], value=pred_class)
            st.write(f"{txt['benign']}: **{prob_benign:.2f}%**")
            st.write(f"{txt['malignant']}: **{prob_malignant:.2f}%**")

# =======================
# SECCIÓN: XGBOOST (TABULAR)
# =======================
if option == "XGBoost - Clasificación Tabular":
    st.header(txt["xgb_title"])

    # ----------------------------------------------------------
    st.markdown(txt["xgb_manual"])
    cols = st.columns(3)
    inputs = []

    for i, fname in enumerate(feature_names):
        with cols[i % 3]:
            val = st.number_input(fname, value=0.0, format="%.4f")
            inputs.append(val)

    if st.button("🔮 Predecir / Predict"):
        try:
            features = np.array(inputs).reshape(1, -1)
            features_scaled = scaler.transform(features)
            dmatrix = xgb.DMatrix(features_scaled)
            preds = booster.predict(dmatrix)
            prob = preds[0]
            pred_class = 1 if prob >= 0.5 else 0

            st.metric(label=txt["prediction"], value=f"Clase {pred_class}")
            st.write(f"📊 Probabilidad / Probability: **{prob*100:.2f}%**")

        except Exception as e:
            st.error(f"{txt['error_pred']}: {e}")

    # ----------------------------------------------------------
    st.markdown(txt["xgb_csv"])
    uploaded_csv = st.file_uploader(txt["upload_csv"], type=["csv"])

    if uploaded_csv is not None:
        try:
            data = pd.read_csv(uploaded_csv)
            st.subheader(txt["preview"])
            st.dataframe(data.head())

            features_scaled = scaler.transform(data)
            dmatrix = xgb.DMatrix(features_scaled)
            preds = booster.predict(dmatrix)

            results = pd.DataFrame({
                txt["prediction"]: (preds >= 0.5).astype(int),
                "Probabilidad / Probability (%)": preds * 100
            })

            st.subheader(txt["results"])
            st.dataframe(results)

        except Exception as e:
            st.error(f"{txt['error_csv']}: {e}")

# =======================
# FOOTER
# =======================
st.sidebar.markdown("---")
