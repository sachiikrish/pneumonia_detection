import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------- Settings ----------------
MODEL_PATH = "best_pneumonia_model.keras"
IMG_SIZE = 224
# Decision threshold chosen on the validation set in the notebook (Tuned CNN = 0.346).
# Other models: Baseline 0.236 | Custom 0.177 | DenseNet121 0.153
THRESHOLD = 0.346

st.set_page_config(page_title="Pneumonia X-ray Classifier", layout="centered")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        return model

    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

model = load_model()


def preprocess(pil_img):
    gray = np.array(pil_img.convert("L"))
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return (gray.astype("float32") / 255.0).reshape(IMG_SIZE, IMG_SIZE, 1)


# # ---------------- Grad-CAM (computed on the pre-sigmoid logit) ----------------
# def gradcam(model, img):
#     last_conv = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1]
#     m = tf.keras.models.clone_model(model)
#     m.set_weights(model.get_weights())
#     m.layers[-1].activation = tf.keras.activations.linear   # logits instead of saturated sigmoid
#     grad_model = tf.keras.models.Model(m.inputs, [m.get_layer(last_conv).output, m.output])

#     x = tf.convert_to_tensor(img[np.newaxis], dtype=tf.float32)
#     with tf.GradientTape() as tape:
#         conv_out, logit = grad_model(x, training=False)
#         score = logit[:, 0]
#     grads = tape.gradient(score, conv_out)
#     weights = tf.reduce_mean(grads, axis=(1, 2))
#     cam = tf.nn.relu(tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0])
#     cam = (cam / (tf.reduce_max(cam) + 1e-8)).numpy()

#     base = np.uint8(255 * img.squeeze())
#     heat = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
#     colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
#     return cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_GRAY2RGB), 0.6, colored, 0.4, 0)


# ---------------- UI ----------------
st.title("Pneumonia Detection from Chest X-rays")
st.caption("CNN trained on the Kaggle pediatric chest X-ray dataset. "
           "Educational project. NOT a medical diagnostic tool.")

uploaded = st.file_uploader("Upload a chest X-ray (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_img = Image.open(uploaded)
    img = preprocess(pil_img)
    prob = float(model.predict(img[np.newaxis], verbose=0)[0, 0])
    label = "PNEUMONIA" if prob >= THRESHOLD else "NORMAL"

    col1, col2 = st.columns(2)
    col1.image(img.squeeze(), caption="Input (resized, grayscale)", clamp=True, use_container_width=True)

    with col2:
        if label == "PNEUMONIA":
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")
        st.metric("P(pneumonia)", f"{prob:.2%}")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(f"Decision threshold: {THRESHOLD}")

    if st.checkbox("Show Grad-CAM heatmap"):
        try:
            st.image(gradcam(model, img), caption="Grad-CAM (red = most influence)", use_container_width=True)
            st.warning("Heatmaps may highlight image markers or corners instead of lungs "
                       "(shortcut learning), so treat them as indicative only.")
        except Exception as e:
            st.info(f"Grad-CAM could not be computed: {e}")

st.markdown("---")
st.caption("Trained on children's X-rays from a single hospital. Results do not generalise to other populations.")
