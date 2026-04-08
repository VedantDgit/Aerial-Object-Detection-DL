import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Try to import ultralytics lazily — don't crash the whole app if it's missing
try:
    from ultralytics import YOLO
    ULTRALYTICS_OK = True
    yolo_load_error = None
except Exception as e:
    ULTRALYTICS_OK = False
    YOLO = None
    yolo_load_error = str(e)

# 🔹 Page config
st.set_page_config(page_title="Aerial Detection System", layout="centered")

# 🔹 Load models (paths resolved relative to this script)
base_dir = os.path.dirname(__file__)
model_path = os.path.normpath(os.path.join(base_dir, '..', 'models', 'best_model.h5'))
try:
    clf_model = load_model(model_path)
except TypeError as e:
    # Handle models saved with extra keys like `quantization_config`
    if 'quantization_config' in str(e):
        from tensorflow.keras.layers import Dense as KerasDense

        class DenseCompat(KerasDense):
            def __init__(self, *args, quantization_config=None, **kwargs):
                super().__init__(*args, **kwargs)

        clf_model = load_model(model_path, custom_objects={"Dense": DenseCompat})
    else:
        raise

# Prefer a valid YOLO weights path that exists in the workspace
yolo_path = os.path.normpath(os.path.join(base_dir, '..', 'runs', 'detect', 'yolo_bird_drone3', 'weights', 'best.pt'))
if not os.path.exists(yolo_path):
    # fallback to a generic path inside project models folder
    yolo_path = os.path.normpath(os.path.join(base_dir, '..', 'models', 'best.pt'))

yolo_model = None
if ULTRALYTICS_OK:
    try:
        if os.path.exists(yolo_path):
            yolo_model = YOLO(yolo_path)
        else:
            # keep yolo_model as None, show message later
            yolo_load_error = f"YOLO weights not found at {yolo_path}"
    except Exception as e:
        yolo_model = None
        yolo_load_error = str(e)

# 🔹 Title
st.markdown("<h1 style='text-align: center;'>🚀 Aerial Object Detection System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Detect and classify <b>Bird</b> or <b>Drone</b> using Deep Learning + YOLOv8</p>",
    unsafe_allow_html=True
)

st.divider()

# 🔹 Upload
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Ensure uploaded images are RGB (drop alpha channel if present)
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # Original Image
    with col1:
        st.subheader("📷 Original Image")
        st.image(img, use_column_width=True)

    # YOLO Detection
    with col2:
        st.subheader("🎯 Detection (YOLOv8)")
        if not ULTRALYTICS_OK:
            st.warning("YOLO detection disabled: 'ultralytics' package is not installed. Check deployment logs.")
            if yolo_load_error:
                st.caption(yolo_load_error)
        elif yolo_model is None:
            st.warning(f"YOLO detection disabled: failed to load weights. ({yolo_path})")
            if yolo_load_error:
                st.caption(yolo_load_error)
        else:
            # ultralytics YOLO accepts numpy arrays; convert from PIL to ensure compatibility
            try:
                results = yolo_model(np.array(img))
                detected_img = results[0].plot()
                st.image(detected_img, use_column_width=True)
            except Exception as e:
                st.error(f"YOLO detection failed: {e}")

    st.divider()

    # 🔹 Classification
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype("float32") / 255.0

    # Handle grayscale or alpha-channel images defensively
    if img_array.ndim == 2:
        # grayscale -> duplicate channels
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        # RGBA -> drop alpha
        img_array = img_array[:, :, :3]

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    prediction = clf_model.predict(img_array)

    # Robustly extract probability and predicted class from model output
    pred_arr = np.asarray(prediction)
    # default
    prob = None
    is_drone = False

    if pred_arr.ndim == 2 and pred_arr.shape[1] > 1:
        class_idx = int(np.argmax(pred_arr[0]))
        prob = float(pred_arr[0, class_idx])
        is_drone = (class_idx == 1)
    elif pred_arr.ndim == 2 and pred_arr.shape[1] == 1:
        prob = float(pred_arr[0, 0])
        is_drone = prob > 0.5
    elif pred_arr.ndim == 1:
        if pred_arr.size > 1:
            class_idx = int(np.argmax(pred_arr))
            prob = float(pred_arr[class_idx])
            is_drone = (class_idx == 1)
        else:
            prob = float(pred_arr[0])
            is_drone = prob > 0.5
    else:
        prob = float(np.ravel(pred_arr)[0])
        is_drone = prob > 0.5

    st.subheader("🧠 Classification Result")
    if is_drone:
        st.success("🚁 Drone Detected")
    else:
        st.success("🐦 Bird Detected")

    # Confidence (clamp between 0 and 1)
    prob = max(0.0, min(1.0, prob))
    st.write(f"Confidence: {prob:.2f}")

    st.divider()

# 🔹 Footer
st.markdown(
    "<p style='text-align: center; color: gray;'>Built using TensorFlow & YOLOv8</p>",
    unsafe_allow_html=True
)