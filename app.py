"""
================================================================
Streamlit Web App: เกมแนวไหนที่คุณควรซื้อตอนนี้?
================================================================
รันด้วยคำสั่ง:  streamlit run app.py
ต้องมีไฟล์อยู่ในโฟลเดอร์เดียวกัน:
  - genre_svm_model.pkl
  - sample_games_by_genre.csv
================================================================
"""

import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------
# โหลดโมเดลและข้อมูลตัวอย่าง (cache ไว้ไม่ต้องโหลดซ้ำทุกครั้ง)
# ----------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("genre_svm_model.pkl")

@st.cache_data
def load_sample_games():
    return pd.read_csv("sample_games_by_genre.csv")

model = load_model()
sample_games = load_sample_games()

# ----------------------------------------------------------------
# ตั้งค่าหน้าเว็บ
# ----------------------------------------------------------------
st.set_page_config(page_title="เกมแนวไหนที่คุณควรซื้อ?", page_icon="🎮", layout="centered")

st.title("🎮 เกมแนวไหนที่คุณควรซื้อตอนนี้?")
st.write("ตอบคำถามสั้น ๆ เกี่ยวกับสไตล์การเล่นเกมที่คุณชอบ แล้วให้ SVM ทำนายแนวเกมที่เหมาะกับคุณ")

st.divider()
st.subheader("แบบสอบถาม")

# ----------------------------------------------------------------
# แบบสอบถาม (Quiz) -> map เป็น Features ที่โมเดลต้องการ
# ----------------------------------------------------------------
platform = st.selectbox(
    "1) อยากเล่นเกมบนแพลตฟอร์มไหน?",
    options=["PS4", "PS3", "PS2", "X360", "XB", "PC", "Wii", "DS", "3DS", "PSP", "GC", "GBA"],
)

rating = st.radio(
    "2) รับได้กับความรุนแรง/เนื้อหาในเกมระดับไหน?",
    options=["E", "E10+", "T", "M"],
    format_func=lambda x: {
        "E": "E (เหมาะกับทุกวัย)",
        "E10+": "E10+ (10 ปีขึ้นไป)",
        "T": "T (วัยรุ่น 13+)",
        "M": "M (18+ เนื้อหารุนแรง)",
    }[x],
    horizontal=True,
)

critic_score = st.slider(
    "3) ให้ความสำคัญกับคะแนนนักวิจารณ์ (Critic Score) แค่ไหน? (0-100)",
    min_value=0, max_value=100, value=75,
)

user_score = st.slider(
    "4) อยากเล่นเกมที่ผู้เล่นคนอื่นให้คะแนนสูงแค่ไหน? (0-10)",
    min_value=0.0, max_value=10.0, value=7.5, step=0.1,
)

global_sales = st.slider(
    "5) ชอบเกมกระแสหลัก/ขายดีทั่วโลก หรือเกมกลุ่มเฉพาะ? (ยอดขายล้านชุด)",
    min_value=0.0, max_value=20.0, value=1.0, step=0.1,
    help="ค่ามาก = เกมฮิตระดับโลก, ค่าน้อย = เกมกลุ่มเฉพาะ/เฉพาะทาง",
)

st.divider()

# ----------------------------------------------------------------
# ทำนายผล
# ----------------------------------------------------------------
if st.button("🔮 ทำนายแนวเกมที่ควรซื้อ", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "Platform": platform,
        "Rating": rating,
        "Critic_Score": critic_score,
        "User_Score": user_score,
        "Global_Sales": global_sales,
    }])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    prob_df = pd.DataFrame({"แนวเกม": classes, "ความน่าจะเป็น": probabilities})
    prob_df = prob_df.sort_values("ความน่าจะเป็น", ascending=False).reset_index(drop=True)

    st.success(f"### 🎯 เกมที่คุณควรซื้อตอนนี้คือแนว: **{prediction}**")

    st.write("ความน่าจะเป็นของแต่ละแนวเกม:")
    st.bar_chart(prob_df.set_index("แนวเกม"))

    # แสดงตัวอย่างเกมยอดนิยมในแนวที่ทำนายได้
    st.subheader(f"ตัวอย่างเกมแนว {prediction} ที่ขายดี")
    matched = sample_games[sample_games["Genre"] == prediction].head(5)
    if not matched.empty:
        st.table(matched[["Name", "Platform", "Global_Sales"]].rename(
            columns={"Name": "ชื่อเกม", "Platform": "แพลตฟอร์ม", "Global_Sales": "ยอดขาย (ล้านชุด)"}
        ))
    else:
        st.write("ไม่พบตัวอย่างเกมในหมวดนี้")

st.divider()
st.caption("โมเดลนี้เทรนด้วย SVM (Support Vector Machine) จากดาต้าเซ็ต Video Game Sales with Ratings")
