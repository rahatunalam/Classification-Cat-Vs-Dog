import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")
st.title("🐾 Cat vs Dog Classifier")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cat_dog_cnn.h5")

try:
    model = load_model()
    input_h = model.input_shape[1]
    input_w = model.input_shape[2]
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model failed to load: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            img = image.resize((input_w, input_h))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array, verbose=0)[0][0]
            confidence = prediction if prediction > 0.5 else 1 - prediction

        if prediction > 0.5:
            st.success(f"🐶 It's a **Dog!** (Confidence: {confidence:.1%})")
        else:
            st.info(f"🐱 It's a **Cat!** (Confidence: {confidence:.1%})")

        col1, col2 = st.columns(2)
        col1.metric("Cat 🐱", f"{(1 - prediction):.1%}")
        col2.metric("Dog 🐶", f"{prediction:.1%}")
else:
    st.info("👆 Please upload a JPG or PNG image to get started.")