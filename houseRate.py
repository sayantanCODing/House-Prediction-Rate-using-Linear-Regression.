import streamlit as st
import numpy as np
import pickle

st.markdown("""
    <style>
    .main {
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-top: 1.0rem !important;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', 'Arial', sans-serif;
    }
    h1 {
        font-size: 2.4rem !important;
        margin-bottom: 0.25em;
    }
    h2 {
        font-size: 1.6rem !important;
        margin-bottom: 0.2em;
    }
    h3 {
        font-size: 1.2rem !important;
        margin-bottom: 0.15em;
    }
    /* Adjust normal text (paragraphs) */
    .block-container {
        font-size: 1.08rem !important;
        padding-top: 0.2rem !important;
    }
    /* Reduce image margin for tighter look */
    [data-testid="stImage"]{
        margin-bottom:0.5rem;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {background-color: #0066cc; color: white;}
    .stSuccess {background-color: #e0ffe0;}
    </style>
""", unsafe_allow_html=True)


st.title("🏡 Ames House Price Predictor")
st.markdown(
    """
    Welcome! Enter house details below to get an instant price prediction.<br>
    <small>Powered by Linear Regression on the famous Ames Housing dataset.</small>
    """, unsafe_allow_html=True
)
st.image(
    "image.jpg",
    caption="Your Home Value Estimator", use_container_width=True
)

st.header("Property Features")


col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider('Overall Quality (1–10)', 1, 10, 6)
    gr_liv_area = st.number_input('Above Ground Living Area (sqft)', 334, 5642, 1500)
    year_built = st.number_input('Year Built', 1872, 2010, 1970)
    year_remod_add = st.number_input('Year Remodeled (or Built)', 1950, 2010, 1970)
    first_flr_sf = st.number_input('1st Floor SF', 334, 4000, 1200)

with col2:
    garage_cars = st.selectbox('Garage Cars', list(range(0,6)), index=2)
    garage_area = st.number_input('Garage Area (sqft)', 0, 1500, 500)
    total_bsmt_sf = st.number_input('Total Basement (sqft)', 0, 3000, 800)
    full_bath = st.selectbox('Full Bathrooms', list(range(0,5)), index=2)


with open("linreg_model_advanced.pkl", "rb") as f:
    model = pickle.load(f)


with st.expander("Show Prediction"):
    if st.button("🏠 Predict Sale Price"):
        input_features = np.array([[
            overall_qual, gr_liv_area, garage_cars, garage_area, total_bsmt_sf,
            first_flr_sf, year_built, full_bath, year_remod_add
        ]])
        predicted_price = model.predict(input_features)[0]
        st.success(
            f"**Estimated House Price:**  \n:moneybag: **${predicted_price:,.2f}**"
        )
        st.caption("⬆️ Adjust features to see updated predictions in real time.")


with st.expander("ℹ️ About this app"):
    st.markdown("""
    - **Model:** Linear Regression trained on top-correlation features from the Ames dataset.
    - **How to use:** Tune the sliders/inputs to match house specs, and click Predict.
    - **Demo:** Designed for showcasing ML, UI design, and deployment skills.
    - **Advanced:** Try using more feature combinations or upload your test CSV data.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<center>Made by Sayantan</center>",
    unsafe_allow_html=True
)
