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


def get_catalog_options(column_name, limit=None):
    if column_name not in catalog.columns:
        return []

    options = sorted(catalog[column_name].dropna().unique().tolist())
    if limit is not None:
        options = options[:limit]
    return options


def get_similar_games(selected_platform, selected_genre, selected_rating, selected_publisher, selected_critic_score, selected_user_score):
    candidates = catalog.copy()

    if 'Genre' in candidates.columns:
        candidates = candidates[candidates['Genre'] == selected_genre]
    if 'Rating' in candidates.columns:
        candidates = candidates[candidates['Rating'] == selected_rating]
    if 'Platform' in candidates.columns and selected_platform:
        candidates = candidates[candidates['Platform'] == selected_platform]
    if 'Publisher' in candidates.columns and selected_publisher != 'Unknown':
        candidates = candidates[candidates['Publisher'] == selected_publisher]

    if len(candidates) == 0:
        candidates = catalog.copy()
        if 'Genre' in candidates.columns:
            candidates = candidates[candidates['Genre'] == selected_genre]
        if 'Rating' in candidates.columns:
            candidates = candidates[candidates['Rating'] == selected_rating]

    candidates = candidates.copy()
    candidates['Critic_Diff'] = (candidates['Critic_Score'].fillna(selected_critic_score) - selected_critic_score).abs()
    candidates['User_Diff'] = (candidates['User_Score'].fillna(selected_user_score) - selected_user_score).abs()
    candidates['Similarity_Score'] = candidates['Critic_Diff'] + candidates['User_Diff']

    return candidates.sort_values(
        ['Similarity_Score', 'Global_Sales'],
        ascending=[True, False]
    )

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
    platform_options = get_catalog_options('Platform')
    platform = st.selectbox("📱 เลือก Platform", platform_options)

    # Genre
    genre_options = get_catalog_options('Genre')
    genre = st.selectbox("🎭 เลือก Genre ที่ชอบ", genre_options)

    # Rating
    rating_options = get_catalog_options('Rating')
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
    publisher_options = get_catalog_options('Publisher', limit=50)
    publisher = st.selectbox("🏢 Publisher (เลือกหรือไม่ก็ได้)",
                             ['Unknown'] + publisher_options)

    # Developer (optional)
    developer_options = get_catalog_options('Developer', limit=50)
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

        similar_games = get_similar_games(
            platform,
            genre,
            rating,
            publisher,
            critic_score,
            user_score,
        )

        recommended_game = similar_games.iloc[0] if len(similar_games) > 0 else None

        # ---------- Display Result ----------
        st.markdown("## 🎯 เกมที่ควรซื้อในตอนนี้")

        if recommended_game is not None:
            result_col1, result_col2 = st.columns([1.2, 0.8])

            with result_col1:
                st.success(f"### {recommended_game['Name']}")
                st.write(f"**Platform:** {recommended_game['Platform']}")
                st.write(f"**Genre:** {recommended_game['Genre']}")
                st.write(f"**Publisher:** {recommended_game['Publisher']}")
                if 'Rating' in recommended_game and pd.notna(recommended_game['Rating']):
                    st.write(f"**Rating:** {recommended_game['Rating']}")

            with result_col2:
                st.metric("🌍 Global Sales", f"{recommended_game['Global_Sales']:.2f}M")
                if pd.notna(recommended_game.get('Critic_Score')):
                    st.metric("📝 Critic Score", f"{recommended_game['Critic_Score']}")
                if pd.notna(recommended_game.get('User_Score')):
                    st.metric("👥 User Score", f"{recommended_game['User_Score']}")

            st.caption(f"เหตุผล: ระบบเลือกเกมที่ใกล้เคียงกับความต้องการและมีแนวโน้มคุ้มซื้อมากที่สุดจากฐานข้อมูล")

        if prediction > 5.0:
            st.info(f"### 🔥 ค่าทำนายบอกว่าเกมแนวนี้มีแนวโน้มได้รับความนิยมสูงมาก")
            st.balloons()
        elif prediction > 1.0:
            st.info(f"### ✅ ค่าทำนายบอกว่าเกมแนวนี้มีแนวโน้มได้รับความนิยมดี")
        elif prediction > 0.1:
            st.warning(f"### ⚠️ ค่าทำนายบอกว่าเกมแนวนี้ได้รับความนิยมปานกลาง")
        else:
            st.error(f"### ❌ ค่าทำนายบอกว่าเกมแนวนี้ไม่ค่อยได้รับความนิยม")

        # Metric display
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("🔮 Predicted Sales", f"{prediction:.2f}M",
                      help="ยอดขายที่โมเดลทำนาย")
        col_m2.metric("📝 Critic Score", f"{critic_score}/100")
        col_m3.metric("👥 User Score", f"{user_score}/10")

        # Progress bar
        st.progress(min(prediction / 10.0, 1.0))
        st.caption(f"ความนิยม: {min(prediction/10*100, 100):.1f}%")

        st.markdown("## 🎮 เกมที่คล้ายกัน")
        if len(similar_games) > 1:
            similar_view = similar_games[['Name', 'Platform', 'Genre', 'Publisher', 'Critic_Score', 'User_Score', 'Global_Sales']].head(10)
            if recommended_game is not None:
                similar_view = similar_view[similar_view['Name'] != recommended_game['Name']]

            if len(similar_view) > 0:
                st.dataframe(similar_view.reset_index(drop=True), use_container_width=True)
            else:
                st.info("ไม่พบเกมที่คล้ายกันมากกว่านี้")
        else:
            st.info("ไม่พบเกมที่คล้ายกันจากเงื่อนไขที่เลือก")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")

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