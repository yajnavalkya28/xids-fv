"""
Network Intrusion Detection using Explainable AI (XIDS) - Streamlit demo
Flow: title/team -> choose model -> upload CSV -> Safe/Attack prediction
      -> SHAP explanation -> (optional) Groq GenAI plain-language explanation
"""
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MAX_ROWS = 5000          # rows predicted per upload (keeps the demo fast)
SHAP_BG = 50             # background rows used for SHAP
TOP_K = 10               # features shown in plot / sent to GenAI
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_PREFERRED = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
]

MODEL_FILES = {
    "XGBoost": ("sk", "xgboost.joblib"),
    "Random Forest": ("sk", "random_forest.joblib"),
    "MLP": ("sk", "mlp.joblib"),
    "CNN": ("dl", "xids_cnn.keras"),
    "Transformer": ("dl", "xids_transformer.keras"),
    "CNN + Transformer": ("dl", "xids_cnn_transformer.keras"),
}

st.set_page_config(page_title="XIDS - Explainable NIDS", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading preprocessing files...")
def load_artifacts():
    p = lambda f: os.path.join(MODEL_DIR, f)
    scaler = joblib.load(p("scaler_all_features.joblib"))
    le = joblib.load(p("label_encoder.joblib"))
    all_cols = list(joblib.load(p("all_feature_columns.pkl")))
    sel_cols = list(joblib.load(p("selected_features.pkl")))
    bg = None
    if os.path.exists(p("shap_background.joblib")):
        bg = np.asarray(joblib.load(p("shap_background.joblib")), dtype="float32")
    return scaler, le, all_cols, sel_cols, bg


@st.cache_resource(show_spinner="Loading model...")
def load_model(name):
    kind, fname = MODEL_FILES[name]
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: models/{fname}")
    if kind == "sk":
        return joblib.load(path)

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    class PositionEmbedding(layers.Layer):
        def __init__(self, length, dim, **kw):
            super().__init__(**kw)
            self.length, self.dim = length, dim
            self.emb = layers.Embedding(length, dim)

        def call(self, x):
            return x + self.emb(tf.range(self.length))

        def get_config(self):
            c = super().get_config()
            c.update(length=self.length, dim=self.dim)
            return c

    return keras.models.load_model(
        path, custom_objects={"PositionEmbedding": PositionEmbedding}, compile=False
    )


def make_predict_fn(name, model):
    """Returns f(2D array) -> probabilities (N, classes)."""
    if MODEL_FILES[name][0] == "sk":
        return model.predict_proba
    return lambda a: np.asarray(
        model.predict(np.asarray(a, dtype="float32")[..., None], verbose=0)
    )


# --------------------------------------------------------------------------
# Preprocessing + prediction
# --------------------------------------------------------------------------
def preprocess(df, scaler, all_cols, sel_cols):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    true_label = None
    if "Label" in df.columns:
        true_label = df.pop("Label").astype(str).str.strip().values
    X = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing = [c for c in all_cols if c not in X.columns]
    X = X.reindex(columns=all_cols)
    X = X.fillna(X.median()).fillna(0)
    scaled = np.clip(scaler.transform(X.values), -10, 10).astype("float32")
    idx = [all_cols.index(c) for c in sel_cols]
    return scaled[:, idx], X[sel_cols].reset_index(drop=True), true_label, missing


@st.cache_data(show_spinner="Running model on your flows...")
def run_prediction(file_bytes, file_name, model_name):
    import io

    scaler, le, all_cols, sel_cols, _ = load_artifacts()
    df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False, nrows=MAX_ROWS)
    Xs, raw, true_label, missing = preprocess(df, scaler, all_cols, sel_cols)
    model = load_model(model_name)
    probs = make_predict_fn(model_name, model)(Xs)
    pred = probs.argmax(1)
    return Xs, raw, true_label, missing, probs, pred


def is_benign(cls):
    return cls.lower() in ("benign", "normal")


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------
def norm_sv(sv, N, F):
    if isinstance(sv, list):
        sv = np.stack([np.asarray(s).reshape(N, F) for s in sv], axis=-1)
    sv = np.asarray(sv)
    if sv.ndim == 4:
        sv = sv.reshape(N, F, -1)
    if sv.shape[0] != N:
        sv = np.moveaxis(sv, 0, -1)
    return sv


def explain_flow(model_name, model, x_row, background):
    """x_row: (1, F) scaled. Returns (F,) SHAP values for every class -> (F, C) and method name."""
    F = x_row.shape[1]
    kind = MODEL_FILES[model_name][0]
    predict_fn = make_predict_fn(model_name, model)

    if model_name in ("XGBoost", "Random Forest"):
        ex = shap.TreeExplainer(model)
        sv = (ex.shap_values(x_row) if model_name == "XGBoost"
              else ex.shap_values(x_row, check_additivity=False))
        return norm_sv(sv, 1, F)[0], "TreeExplainer"

    if kind == "dl":
        try:
            ex = shap.GradientExplainer(model, background[..., None])
            sv = ex.shap_values(x_row[..., None])
            return norm_sv(sv, 1, F)[0], "GradientExplainer"
        except Exception:
            pass  # fall back to model-agnostic explainer

    masker = shap.maskers.Independent(background, max_samples=SHAP_BG)
    ex = shap.Explainer(predict_fn, masker, algorithm="permutation")
    sv = ex(x_row, max_evals=2 * F + 1, silent=True).values
    return norm_sv(sv, 1, F)[0], "PermutationExplainer"


def shap_plot(names, values, title):
    order = np.argsort(np.abs(values))[::-1][:TOP_K][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([names[i] for i in order], values[order],
            color=["#d62728" if values[i] > 0 else "#1f77b4" for i in order])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("SHAP value (red = pushes toward predicted class, blue = pushes away)")
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Groq GenAI
# --------------------------------------------------------------------------
def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def groq_available_models(key):
    """Returns list of model ids if the key works, else None."""
    try:
        r = requests.get(f"{GROQ_BASE}/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=10)
        if r.status_code != 200:
            return None
        return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        return None


def groq_explain(key, models, prompt):
    """Try models in order; return (text, model_used) or (None, error)."""
    last_err = "unknown error"
    for m in models:
        body = {
            "model": m,
            "messages": [
                {"role": "system",
                 "content": "You explain intrusion-detection results to students in very simple words."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 600 if "gpt-oss" in m else 200,
        }
        if "gpt-oss" in m:
            body["reasoning_effort"] = "low"
        try:
            r = requests.post(f"{GROQ_BASE}/chat/completions",
                              headers={"Authorization": f"Bearer {key}"},
                              json=body, timeout=30)
            if r.status_code == 200:
                text = (r.json()["choices"][0]["message"].get("content") or "").strip()
                if text:
                    return text, m
                last_err = f"{m}: empty response"
            else:
                last_err = f"{m}: HTTP {r.status_code}"
        except Exception as e:
            last_err = f"{m}: {type(e).__name__}"
    return None, last_err


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🛡️ Network Intrusion Detection using Explainable AI")
st.caption("Detect network attacks with deep learning / machine learning and see *why* the model decided.")

with st.container(border=True):
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Team")
        st.markdown(
            "- Yajnavalkya M — 23BCE7243\n"
            "- Jayavardhan M — 23BCE8471\n"
            "- Tulasi Ram D — 23BCE9767\n"
            "- Abhishek P — 23BCE9091"
        )
    with c2:
        st.subheader("Guide")
        st.markdown("**Dr. Visalakshi Annepu**")

st.divider()

# Check that preprocessing files exist
try:
    scaler, le, all_cols, sel_cols, bg_saved = load_artifacts()
except Exception as e:
    st.error(f"Could not load preprocessing files from `models/`: {e}")
    st.stop()
class_names = list(le.classes_)

# Step 1 - model
st.header("1️⃣ Select a model")
model_name = st.selectbox("Model", list(MODEL_FILES.keys()), label_visibility="collapsed")

# Step 2 - upload
st.header("2️⃣ Upload a dataset")
st.caption(f"CSV of CICFlowMeter network flows (an optional `Label` column is fine). "
           f"First {MAX_ROWS:,} rows are used.")
up = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
if up is None:
    st.info("Upload a CSV to continue.")
    st.stop()

try:
    Xs, raw, true_label, missing, probs, pred = run_prediction(up.getvalue(), up.name, model_name)
except FileNotFoundError as e:
    st.error(str(e)); st.stop()
except Exception as e:
    st.error(f"Could not process this file/model: {type(e).__name__}: {e}")
    st.stop()

if len(missing) > 0.5 * len(all_cols):
    st.error("This file does not look like CICFlowMeter data (most expected columns are missing).")
    st.stop()
if missing:
    st.warning(f"{len(missing)} expected columns were missing and filled with 0.")

# Step 3 - verdict
st.header("3️⃣ Detection result")
pred_names = np.array([class_names[i] for i in pred])
conf = probs.max(1)
attack_mask = np.array([not is_benign(c) for c in pred_names])

m1, m2, m3 = st.columns(3)
m1.metric("Flows analysed", f"{len(pred):,}")
m2.metric("Safe flows", f"{int((~attack_mask).sum()):,}")
m3.metric("Attack flows", f"{int(attack_mask.sum()):,}")

if attack_mask.any():
    top_attack = pd.Series(pred_names[attack_mask]).value_counts().index[0]
    st.error(f"🚨 Attack traffic detected. Most common attack class: **{top_attack}**")
else:
    st.success("✅ All flows look safe (Benign).")

summary = pd.Series(pred_names).value_counts().rename_axis("Predicted class").reset_index(name="Flows")
summary["Status"] = summary["Predicted class"].apply(lambda c: "✅ Safe" if is_benign(c) else "🚨 Attack")
st.dataframe(summary, use_container_width=True, hide_index=True)

with st.expander("Per-flow predictions"):
    table = pd.DataFrame({
        "Flow #": np.arange(len(pred)),
        "Status": np.where(attack_mask, "🚨 Attack", "✅ Safe"),
        "Predicted class": pred_names,
        "Confidence": (conf * 100).round(1).astype(str) + " %",
    })
    if true_label is not None:
        table["True label"] = true_label
    st.dataframe(table.head(1000), use_container_width=True, hide_index=True)

# Step 4 - SHAP
st.header("4️⃣ SHAP explanation")
default_idx = int(np.argmax(attack_mask)) if attack_mask.any() else 0
flow_idx = st.number_input("Flow # to explain", 0, len(pred) - 1, default_idx, step=1)
flow_idx = int(flow_idx)
verdict = pred_names[flow_idx]
st.write(f"Flow **#{flow_idx}** → {'✅ Safe' if is_benign(verdict) else '🚨 Attack'} "
         f"(**{verdict}**, confidence {conf[flow_idx]*100:.1f}%)")

sig = (model_name, up.name, up.size, flow_idx)
if st.button("Explain with SHAP", type="primary"):
    with st.spinner("Computing SHAP values..."):
        try:
            background = bg_saved if bg_saved is not None else Xs[np.random.default_rng(42).choice(
                len(Xs), min(SHAP_BG, len(Xs)), replace=False)]
            background = background[:SHAP_BG]
            sv, method = explain_flow(model_name, load_model(model_name), Xs[flow_idx:flow_idx + 1], background)
            local = sv[:, int(pred[flow_idx])]
            st.session_state["shap"] = dict(sig=sig, local=local, method=method)
            st.session_state.pop("genai", None)
        except Exception as e:
            st.session_state.pop("shap", None)
            st.error(f"SHAP failed: {type(e).__name__}: {e}")

S = st.session_state.get("shap")
if S and S["sig"] == sig:
    local = S["local"]
    st.pyplot(shap_plot(sel_cols, local,
                        f"{model_name} | flow #{flow_idx} | predicted: {verdict} ({S['method']})"))
    order = np.argsort(np.abs(local))[::-1][:TOP_K]
    top_df = pd.DataFrame({
        "Feature": [sel_cols[i] for i in order],
        "Value in this flow": [float(raw.iloc[flow_idx][sel_cols[i]]) for i in order],
        "SHAP": local[order].round(4),
        "Effect": ["→ supports " + verdict if local[i] > 0 else "← goes against " + verdict for i in order],
    })
    st.dataframe(top_df, use_container_width=True, hide_index=True)

    # Step 5 - GenAI
    st.header("5️⃣ GenAI explanation (Groq)")
    key = get_groq_key()
    if not key:
        st.warning("Groq API key not found in Streamlit secrets (`GROQ_API_KEY`). "
                   "The explanation above is still valid; GenAI text is unavailable.")
    else:
        if st.button("Get GenAI explanation"):
            with st.spinner("Contacting Groq..."):
                avail = groq_available_models(key)
                if not avail:
                    st.session_state["genai"] = dict(sig=sig, error="Groq API is not reachable or the key is invalid.")
                else:
                    models = [m for m in GROQ_PREFERRED if m in avail]
                    if not models:
                        st.session_state["genai"] = dict(sig=sig, error="None of the expected Groq models are available.")
                    else:
                        feats = "\n".join(
                            f"- {sel_cols[i]} = {float(raw.iloc[flow_idx][sel_cols[i]]):.4g} "
                            f"({'supports' if local[i] > 0 else 'goes against'} this prediction)"
                            for i in order[:6])
                        status = "safe (benign) traffic" if is_benign(verdict) else f"a '{verdict}' network attack"
                        prompt = (
                            f"An intrusion detection model ({model_name}) classified a network flow as {status} "
                            f"with {conf[flow_idx]*100:.0f}% confidence. The most influential features according "
                            f"to SHAP were:\n{feats}\n\n"
                            "In 2 to 3 short sentences, in very simple words, explain why the model most likely "
                            "reached this decision. No bullet points, no jargon, no mention of SHAP values."
                        )
                        text, used = groq_explain(key, models, prompt)
                        st.session_state["genai"] = (
                            dict(sig=sig, text=text, model=used) if text
                            else dict(sig=sig, error=f"All Groq models failed ({used}).")
                        )
        G = st.session_state.get("genai")
        if G and G["sig"] == sig:
            if "text" in G:
                st.success(G["text"])
                st.caption(f"Generated by Groq · {G['model']}")
            else:
                st.warning(G["error"])
