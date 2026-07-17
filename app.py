# app.py
# =====================================================
# 🎮 Game Recommender - SVM + Streamlit
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ---------- Page Config ----------
st.set_page_config(
    page_title="🎮 เกมไหนดี? - AI แนะนำเกม",
    page_icon="🎮",
    layout="wide"
)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    model = joblib.load('game_svm_model.pkl')
    encoders = joblib.load('encoders.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, encoders, scaler

@st.cache_data
def load_catalog():
    return pd.read_csv('game_catalog.csv')

model, encoders, scaler = load_model()
catalog = load_catalog()

# ---------- Sidebar ----------
st.sidebar.title("🎮 เกมไหนดี?")
st.sidebar.markdown("AI จะช่วยคุณเลือกเกมที่น่าเล่น!")
st.sidebar.markdown("---")

# ---------- Main Header ----------
st.title("🎮 เกมไหนดี? - AI แนะนำเกม")
st.markdown("### บอกความต้องการของคุณ แล้ว AI จะทำนายว่าเกมนั้นจะได้รับความนิยมแค่ไหน!")

# =====================================================
# Input Section
# =====================================================
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 ข้อมูลเกม")

    # Platform
    platform_options = sorted(catalog['Platform'].dropna().unique().tolist())
    platform = st.selectbox("📱 เลือก Platform", platform_options)

    # Genre
    genre_options = sorted(catalog['Genre'].dropna().unique().tolist())
    genre = st.selectbox("🎭 เลือก Genre ที่ชอบ", genre_options)

    # Rating
    rating_options = sorted(catalog['Rating'].dropna().unique().tolist())
    rating = st.selectbox("🔞 เลือกรายการ Rating", rating_options)

with col2:
    st.subheader("⭐ คะแนนที่ต้องการ")

    # Critic Score
    critic_score = st.slider(
        "📝 Critic Score ที่ต้องการ",
        min_value=0, max_value=100, value=75, step=5,
        help="คะแนนจากนักวิจารณ์ (0-100)"
    )

    # User Score
    user_score = st.slider(
        "👥 User Score ที่ต้องการ",
        min_value=0.0, max_value=10.0, value=7.0, step=0.5,
        help="คะแนนจากผู้เล่น (0-10)"
    )

    # Publisher (optional)
    publisher_options = sorted(catalog['Publisher'].dropna().unique().tolist())[:50]
    publisher = st.selectbox("🏢 Publisher (เลือกหรือไม่ก็ได้)",
                             ['Unknown'] + publisher_options)

    # Developer (optional)
    developer_options = sorted(catalog['Developer'].dropna().unique().tolist())[:50]
    developer = st.selectbox("🛠️ Developer (เลือกหรือไม่ก็ได้)",
                             ['Unknown'] + developer_options)

# =====================================================
# Prediction
# =====================================================
st.markdown("---")

if st.button("🔮 ทำนายเลย!", type="primary", use_container_width=True):
    try:
        # Prepare input
        input_data = pd.DataFrame({
            'Platform': [platform],
            'Genre': [genre],
            'Publisher': [publisher],
            'Developer': [developer],
            'Rating': [rating],
            'Critic_Score': [critic_score],
            'User_Score': [user_score]
        })

        # Encode categorical
        for col in ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']:
            le = encoders[col]
            # Handle unseen labels
            val = input_data[col].values[0]
            if val in le.classes_:
                input_data[col] = le.transform([val])
            else:
                input_data[col] = 0

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction_log = model.predict(input_scaled)[0]
        prediction = np.expm1(prediction_log)  # Convert back from log

        # ---------- Display Result ----------
        st.markdown("## 📊 ผลการทำนาย")

        if prediction > 5.0:
            st.success(f"### 🔥 เกมนี้มีแนวโน้มจะได้รับความนิยมสูงมาก!")
            st.balloons()
        elif prediction > 1.0:
            st.info(f"### ✅ เกมนี้มีแนวโน้มได้รับความนิยมดี")
        elif prediction > 0.1:
            st.warning(f"### ⚠️ เกมนี้ได้รับความนิยมปานกลาง")
        else:
            st.error(f"### ❌ เกมนี้ไม่ค่อยได้รับความนิยม")

        # Metric display
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("🌍 Predicted Global Sales", f"{prediction:.2f}M",
                      help="ล้านชุดทั่วโลก")
        col_m2.metric("📝 Critic Score", f"{critic_score}/100")
        col_m3.metric("👥 User Score", f"{user_score}/10")

        # Progress bar
        st.progress(min(prediction / 10.0, 1.0))
        st.caption(f"ความนิยม: {min(prediction/10*100, 100):.1f}%")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")

# =====================================================
# Recommendation Section
# =====================================================
st.markdown("---")
st.markdown("## 🏆 เกมแนะนำตามความชอบของคุณ")

# Filter games by user preferences
filtered = catalog[
    (catalog['Genre'] == genre) &
    (catalog['Rating'] == rating)
].copy()

if len(filtered) > 0:
    # Sort by Global_Sales
    filtered = filtered.sort_values('Global_Sales', ascending=False).head(10)

    st.dataframe(
        filtered[['Name', 'Platform', 'Publisher', 'Critic_Score',
                  'User_Score', 'Global_Sales']].reset_index(drop=True),
        use_container_width=True
    )
else:
    st.info("ไม่พบเกมที่ตรงกับเงื่อนไข ลองเปลี่ยน Genre หรือ Rating ดู nhé!")

# =====================================================
# Footer
# =====================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🤖 Powered by SVM (Support Vector Machine) + Streamlit</p>
    <p>📊 ข้อมูลจาก Video Games Sales Dataset</p>
</div>
""", unsafe_allow_html=True)