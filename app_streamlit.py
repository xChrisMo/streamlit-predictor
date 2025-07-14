import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

st.set_page_config(page_title="Youth Outcome Predictor",
                   page_icon="🧭",
                   layout="wide")

LOGO_PATH = "cp_logo.png"          

# ---------- CSS ----------
st.markdown(
    """
    <style>
        #MainMenu {visibility:hidden;}
        footer   {visibility:hidden;}
        .stCard  {background:#1f1f1f;padding:2rem;border-radius:1rem;
                  box-shadow:0 0 10px rgba(0,0,0,.3);}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- model + scaler ----------
@st.cache_resource
def load_artifacts():
    model  = tf.keras.models.load_model("model_nn.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# exact order from training
ORDERED_COLS = [
    "Had_EET", "Care_Leaver", "Ever_Slept_Rough",
    "Gender_Male", "Gender_Non-Binary", "Gender_Gender Fluid", "Gender_Other",
    "Region_Manchester", "Region_Bradford", "Region_Barnsley"
]

GENDERS = ["Female", "Male", "Non-Binary", "Gender Fluid", "Other"]  # ASCII hyphen!
REGIONS = ["London", "Manchester", "Bradford", "Barnsley"]

# ---------- helpers ----------
GENDER_KEY = {
    "Male":          "Gender_Male",
    "Non-Binary":    "Gender_Non-Binary",
    "Gender Fluid":  "Gender_Gender Fluid",
    "Other":         "Gender_Other",
}

REGION_KEY = {
    "Manchester": "Region_Manchester",
    "Bradford":   "Region_Bradford",
    "Barnsley":   "Region_Barnsley",
}

def build_vector(had_eet, care_lv, slept_rgh, gender_sel, region_sel):
    vec_dict = dict.fromkeys(ORDERED_COLS, 0)
    vec_dict["Had_EET"]          = int(had_eet)
    vec_dict["Care_Leaver"]      = int(care_lv)
    vec_dict["Ever_Slept_Rough"] = int(slept_rgh)

    # one‑hot gender (Female is reference)
    if gender_sel != "Female":
        vec_dict[GENDER_KEY[gender_sel]] = 1

    # one‑hot region (London is reference)
    if region_sel != "London":
        vec_dict[REGION_KEY[region_sel]] = 1

    return np.array([list(vec_dict.values())])

# ---------- simple router ----------
page = st.query_params.get("page", "home")

# ---------- HOME ----------
if page == "home":
    st.image(LOGO_PATH, width=220)
    st.title("Youth Outcome Predictor")
    st.write("Estimate MTLI & EET success in seconds. "
             "Upload a CSV or fill the quick form below.")
    if st.button("Get started ➜"):
        st.query_params["page"] = "predict"
        st.experimental_rerun()

# ---------- PREDICT ----------
else:
    # --- sidebar form ---
    st.sidebar.image(LOGO_PATH, width=150)
    st.sidebar.title("Input a single young person")

    had_eet   = st.sidebar.selectbox("Had EET on arrival?", ["No", "Yes"]) == "Yes"
    care_lv   = st.sidebar.selectbox("Care leaver?", ["No", "Yes"]) == "Yes"
    slept_rgh = st.sidebar.selectbox("Ever slept rough?", ["No", "Yes"]) == "Yes"
    gender    = st.sidebar.selectbox("Gender",  GENDERS)
    region    = st.sidebar.selectbox("Region",  REGIONS)

    # --- centered card ---
    c1, c2, _ = st.columns([1, 2, 1])
    with c2:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("Prediction")

        if st.sidebar.button("Predict"):
            X = build_vector(had_eet, care_lv, slept_rgh, gender, region)
            X_scaled = scaler.transform(X)
            prob = float(model.predict(X_scaled)[0, 0])

            outcome = "YES, achieved MTLI and EET" if prob >= 0.5 else "NO"
            st.metric("Outcome", outcome, f"{prob*100:.1f}% confidence")
            st.progress(prob, text=f"{prob*100:.1f}%")

        else:
            st.info("Choose values in the sidebar and hit **Predict**")

        st.markdown("</div>", unsafe_allow_html=True)
