# app.py
# =====================================================
# 🎮 Game Recommender - แนะนำเกมที่ควรซื้อ
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ---------- Page Config ----------
st.set_page_config(
    page_title="🎮 เกมไหนดี? - แนะนำเกมที่ควรซื้อ",
    page_icon="🎮",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .game-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .game-title {
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .game-info {
        font-size: 0.9em;
        opacity: 0.9;
    }
    .recommend-badge {
        background: #ff6b6b;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
        margin-bottom: 10px;
    }
            
    :root {
    --dark-bg: #1a1a2e;
    --card-bg: #2d2d44;
    --accent-color: #ffd700;
    --text-primary: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Load Model & Data ----------
@st.cache_resource
def load_model():
    model = joblib.load('game_svm_model.pkl')
    encoders = joblib.load('encoders.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, encoders, scaler

@st.cache_data
def load_catalog():
    df = pd.read_csv('game_catalog.csv')
    # แปลง Critic_Score และ User_Score เป็น numeric
    df['Critic_Score'] = pd.to_numeric(df['Critic_Score'], errors='coerce').fillna(0)
    df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce').fillna(0)
    return df

model, encoders, scaler = load_model()
catalog = load_catalog()

# ---------- Header ----------
st.title("🎮 เกมไหนดี? - แนะนำเกมที่ควรซื้อ")
st.markdown("### บอกความต้องการของคุณ แล้วเราจะแนะนำเกมที่น่าสนใจจากฐานข้อมูล!")

# =====================================================
# Input Section
# =====================================================
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 ข้อมูลเกมที่คุณสนใจ")

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
    st.subheader("⭐ เกณฑ์การเลือก")

    # Critic Score minimum
    min_critic_score = st.slider(
        "📝 Critic Score ขั้นต่ำ",
        min_value=0, max_value=100, value=70, step=5,
        help="คะแนนจากนักวิจารณ์ขั้นต่ำ (0-100)"
    )

    # User Score minimum
    min_user_score = st.slider(
        "👥 User Score ขั้นต่ำ",
        min_value=0.0, max_value=10.0, value=6.0, step=0.5,
        help="คะแนนจากผู้เล่นขั้นต่ำ (0-10)"
    )

    # จำนวนเกมที่แนะนำ
    num_recommendations = st.slider(
        "📋 จำนวนเกมที่แนะนำ",
        min_value=3, max_value=15, value=5, step=1,
        help="จำนวนเกมที่จะแสดงในรายการแนะนำ"
    )

# =====================================================
# Recommendation Logic
# =====================================================
st.markdown("---")

if st.button("🔮 แนะนำเกมที่น่าซื้อ!", type="primary", use_container_width=True):
    try:
        # 1. กรองเกมจาก dataset ที่ตรงกับเงื่อนไข
        filtered_games = catalog[
            (catalog['Platform'] == platform) &
            (catalog['Genre'] == genre) &
            (catalog['Rating'] == rating) &
            (catalog['Critic_Score'] >= min_critic_score) &
            (catalog['User_Score'] >= min_user_score)
        ].copy()

        # 2. เรียงตามยอดขาย (Global_Sales)
        filtered_games = filtered_games.sort_values('Global_Sales', ascending=False)

        # 3. แสดงผล
        if len(filtered_games) > 0:
            st.success(f"### ✅ พบ {len(filtered_games)} เกมที่ตรงกับเงื่อนไข!")

            # แสดง Top Games
            top_games = filtered_games.head(num_recommendations)

            st.markdown("## 🏆 เกมที่แนะนำ - ควรซื้อ!")
            st.markdown(f"*เรียงตามยอดขายและคะแนน*")

            for idx, row in top_games.iterrows():
                st.markdown(f"""
                <div class="game-card">
                    <div class="recommend-badge">🎮 แนะนำ</div>
                    <div class="game-title">{row['Name']}</div>
                    <div class="game-info">
                        📱 {row['Platform']} | 🎭 {row['Genre']} | 🔞 {row['Rating']}<br>
                        📝 Critic: {row['Critic_Score']}/100 | 👥 User: {row['User_Score']}/10<br>
                        💰 ยอดขาย: {row['Global_Sales']:.2f}M | 🏢 {row['Publisher']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # =====================================================
            # แสดงเกมที่คล้ายกัน (Similar Games)
            # =====================================================
            st.markdown("---")
            st.markdown("## 🔍 เกมที่คล้ายกัน (ต่างเงื่อนไขเล็กน้อย)")

            # หาเกมที่คล้ายกัน - ต่าง Genre หรือ Rating หรือ Platform
            similar_games = catalog[
                (catalog['Platform'] == platform) &
                (catalog['Critic_Score'] >= min_critic_score - 10) &
                (catalog['User_Score'] >= min_user_score - 1)
            ].copy()

            # เอาเกมที่แนะนำไปแล้วออก
            if len(top_games) > 0:
                similar_games = similar_games[~similar_games['Name'].isin(top_games['Name'])]

            # เรียงตามยอดขาย
            similar_games = similar_games.sort_values('Global_Sales', ascending=False).head(num_recommendations)

            if len(similar_games) > 0:
                for idx, row in similar_games.iterrows():
                    st.markdown(f"""
                    <div style="background: #f0f0f0; padding: 15px; border-radius: 10px; margin: 8px 0; border-left: 4px solid #667eea;">
                        <strong>{row['Name']}</strong><br>
                        <small>
                            📱 {row['Platform']} | 🎭 {row['Genre']} | 🔞 {row['Rating']} |
                            📝 {row['Critic_Score']}/100 | 👥 {row['User_Score']}/10 |
                            💰 {row['Global_Sales']:.2f}M
                        </small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("ไม่พบเกมที่คล้ายกันในเงื่อนไขนี้")

        else:
            st.warning("### ⚠️ ไม่พบเกมที่ตรงกับเงื่อนไข")
            st.markdown("**ลองปรับเงื่อนไขดังนี้:**")
            st.markdown("- ลด Critic Score หรือ User Score ขั้นต่ำ")
            st.markdown("- เปลี่ยน Platform, Genre หรือ Rating")

            # แสดงเกมที่ใกล้เคียงที่สุด
            st.markdown("### 🔍 เกมที่ใกล้เคียงที่สุด (ต่างเงื่อนไข)")
            close_games = catalog[
                (catalog['Platform'] == platform) |
                (catalog['Genre'] == genre)
            ].sort_values('Global_Sales', ascending=False).head(num_recommendations)

            for idx, row in close_games.iterrows():
                st.markdown(f"""
                <div style="background: rgba(255, 243, 205, 0.1); padding: 15px; border-radius: 10px; margin: 8px 0;">
                    <strong>{row['Name']}</strong><br>
                    <small>
                        📱 {row['Platform']} | 🎭 {row['Genre']} | 🔞 {row['Rating']} |
                        📝 {row['Critic_Score']}/100 | 👥 {row['User_Score']}/10 |
                        💰 {row['Global_Sales']:.2f}M
                    </small>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
        import traceback
        st.code(traceback.format_exc())

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