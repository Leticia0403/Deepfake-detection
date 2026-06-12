import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from mtcnn import MTCNN
from tensorflow.keras.models import load_model


IMG_SIZE = 224
FRAMES_PER_VIDEO = 10
MODEL_PATH = "resnet50_attention_lstm_final.h5"
SUPPORTED_VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="att_u",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        v = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        vu = tf.tensordot(v, self.u, axes=1)
        alphas = tf.nn.softmax(vu)
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), axis=1)
        return output, alphas


@tf.keras.utils.register_keras_serializable()
def preprocess_input(x):
    # Needed because model contains Lambda(preprocess_input) internally.
    return tf.keras.applications.resnet50.preprocess_input(x)


@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: '{MODEL_PATH}'. Place it in the project root."
        )
    return load_model(
        MODEL_PATH,
        custom_objects={
            "AttentionLayer": AttentionLayer,
            "preprocess_input": preprocess_input,
        },
        compile=False,
        safe_mode=False,
    )


@st.cache_resource
def get_detector():
    return MTCNN()


@st.cache_resource
def get_attention_probe():
    """
    Try to expose attention weights from the custom attention layer.
    Returns None if model graph/layer output cannot be probed.
    """
    model = get_model()
    try:
        attention_layer = model.get_layer("Proposed_Attention")
        layer_output = attention_layer.output
        if isinstance(layer_output, (list, tuple)) and len(layer_output) > 1:
            attention_weights_tensor = layer_output[1]
            return tf.keras.Model(
                inputs=model.input,
                outputs=attention_weights_tensor,
                name="attention_probe",
            )
    except Exception:
        return None
    return None


def extract_frames_from_video(video_path, num_frames=FRAMES_PER_VIDEO, progress_cb=None):
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return frames

    if total_frames < num_frames:
        indices = range(total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    detector = get_detector()
    total_indices = len(indices)
    for step, idx in enumerate(indices, start=1):
        if progress_cb is not None:
            progress_cb(step, total_indices, f"Scanning frame {step}/{total_indices}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = detector.detect_faces(frame_rgb)
        except Exception:
            # If MTCNN fails on a frame (e.g. tiny/invalid conv input),
            # treat it as "no face found" instead of crashing.
            continue
        if not results:
            continue

        x, y, width, height = results[0]["box"]
        x, y = max(0, x), max(0, y)
        if width <= 0 or height <= 0:
            continue

        face_crop = frame_rgb[y : y + height, x : x + width]
        if face_crop.size == 0:
            continue

        face_crop = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
        frames.append(face_crop)

    cap.release()
    return frames


def prepare_sequence(face_frames, num_frames=FRAMES_PER_VIDEO):
    if not face_frames:
        raise ValueError("No face frames extracted.")

    padded_frames = [np.array(frame, copy=True) for frame in face_frames]
    while len(padded_frames) < num_frames:
        padded_frames.append(np.array(padded_frames[-1], copy=True))

    sequence = np.array(padded_frames[:num_frames], dtype=np.float32)
    sequence = np.expand_dims(sequence, axis=0)  # (1, 10, 224, 224, 3)
    return sequence


def predict_video(model, attention_probe, face_frames):
    x_input = prepare_sequence(face_frames, FRAMES_PER_VIDEO)
    fake_prob = float(model.predict(x_input, verbose=0)[0][0])
    real_prob = 1.0 - fake_prob
    label = "Fake" if fake_prob >= 0.5 else "Real"
    attention_weights = None
    if attention_probe is not None:
        try:
            weights = attention_probe.predict(x_input, verbose=0)
            attention_weights = np.squeeze(weights)
        except Exception:
            attention_weights = None
    return label, fake_prob, real_prob, attention_weights


def get_focus_level(weight_value):
    if weight_value >= 0.14:
        return "High focus"
    if weight_value >= 0.10:
        return "Medium focus"
    return "Low focus"


st.set_page_config(page_title="Deepfake Detector", page_icon="🎭", layout="centered")
st.title("🎭 Deepfake Video Detector")
st.markdown(
    """
    <style>
    .small-note { color: #9aa0a6; font-size: 0.92rem; }
    .prob-fake { color: #dc2626; font-weight: 700; }
    .prob-real { color: #16a34a; font-weight: 700; }
    .verdict-banner {
        margin: 0.4rem 0 0.8rem 0;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        font-size: 1.35rem;
        font-weight: 800;
        text-align: center;
        border: 1px solid transparent;
    }
    .verdict-fake {
        color: #ffffff;
        background: linear-gradient(90deg, #ef4444, #dc2626);
        box-shadow: 0 0 16px rgba(220, 38, 38, 0.45);
        border-color: rgba(127, 29, 29, 0.35);
    }
    .verdict-real {
        color: #ffffff;
        background: linear-gradient(90deg, #16a34a, #15803d);
        box-shadow: 0 0 16px rgba(22, 163, 74, 0.45);
        border-color: rgba(20, 83, 45, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.write("Upload a video and get a **Real/Fake** prediction.")

try:
    model = get_model()
    attention_probe = get_attention_probe()
    st.success("Model loaded.")
except Exception as exc:
    model = None
    attention_probe = None
    st.error(str(exc))

uploaded_file = st.file_uploader("Upload video", type=SUPPORTED_VIDEO_TYPES)

if uploaded_file is not None:
    video_bytes = uploaded_file.getvalue()
    st.video(video_bytes)
    if st.button("Run Prediction", type="primary"):
        if model is None:
            st.stop()

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(video_bytes)
                temp_path = tmp_file.name

            status_box = st.empty()
            progress = st.progress(0, text="Preparing...")

            def update_progress(step, total, message):
                ratio = int(65 * (step / max(total, 1)))
                progress.progress(ratio, text=message)
                status_box.info(message)

            frames = extract_frames_from_video(
                temp_path,
                FRAMES_PER_VIDEO,
                progress_cb=update_progress,
            )

            if len(frames) == 0:
                progress.progress(100, text="Done")
                st.warning("Face not found in sampled frames. Please use a clearer face video.")
            else:
                if len(frames) < FRAMES_PER_VIDEO:
                    st.info(
                        f"Video no error: only {len(frames)} face frame(s) found. "
                        f"Auto-padding to {FRAMES_PER_VIDEO} frames for prediction."
                    )

                progress.progress(80, text="Running prediction...")
                status_box.info("Running model inference...")
                label, fake_prob, real_prob, attention_weights = predict_video(
                    model,
                    attention_probe,
                    frames,
                )
                progress.progress(100, text="Prediction complete")
                status_box.success("Completed")

                st.subheader("Prediction")
                if label == "Fake":
                    st.error(f"Final Verdict: {label}")
                    st.markdown(
                        "<div class='verdict-banner verdict-fake'>FAKE VIDEO DETECTED</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.success(f"Final Verdict: {label}")
                    st.markdown(
                        "<div class='verdict-banner verdict-real'>REAL VIDEO DETECTED</div>",
                        unsafe_allow_html=True,
                    )

                col1, col2 = st.columns(2)
                col1.markdown(
                    f"<p class='prob-fake'>Fake probability: {fake_prob:.2%}</p>",
                    unsafe_allow_html=True,
                )
                col2.markdown(
                    f"<p class='prob-real'>Real probability: {real_prob:.2%}</p>",
                    unsafe_allow_html=True,
                )
                st.progress(fake_prob, text=f"Fake confidence: {fake_prob:.2%}")

                left_col, right_col = st.columns([1.25, 1], gap="large")
                with left_col:
                    st.subheader("Detected Face Frames")
                    st.caption(f"Detected face frames: {len(frames)}")
                    frame_cols = st.columns(2)
                    for i, frame in enumerate(frames[:10]):
                        frame_cols[i % 2].image(
                            frame,
                            caption=f"Frame {i + 1}",
                            use_container_width=True,
                        )

                with right_col:
                    st.subheader("Where The Model Focused")
                    if attention_weights is None:
                        st.info("Frame focus details are not available for this model file.")
                    else:
                        weights = np.array(attention_weights).astype(float).flatten()
                        if weights.size != FRAMES_PER_VIDEO:
                            st.info("Frame focus data format is not supported.")
                        else:
                            weight_sum = float(np.sum(weights))
                            if weight_sum > 0:
                                weights = weights / weight_sum

                            top_idx = int(np.argmax(weights))
                            st.write(
                                f"Most important moment for this prediction: **Frame {top_idx + 1}**."
                            )

                            for i, w in enumerate(weights):
                                st.progress(
                                    float(w),
                                    text=f"Frame {i + 1}: {get_focus_level(float(w))} ({float(w):.2%} focus)",
                                )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except PermissionError:
                    # On Windows, the temp file can still be locked briefly.
                    pass
