import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="MBTI Type by Country", layout="wide")

st.title("🌍 MBTI 유형별 상위 국가 분석")
st.markdown("국가별 MBTI 데이터에서 특정 유형이 높은 상위 10개국을 시각적으로 확인합니다.")

# ---------------------------
# 파일 업로드 또는 로컬 파일 로드
# ---------------------------
uploaded_file = st.sidebar.file_uploader("📂 CSV 파일을 업로드하세요", type=["csv"])

@st.cache_data
def load_data(upload):
    if upload is not None:
        return pd.read_csv(upload)
    else:
        return pd.read_csv("countriesMBTI_16types.csv")

try:
    df = load_data(uploaded_file)
except FileNotFoundError:
    st.error("⚠️ CSV 파일을 찾을 수 없습니다. 업로드하거나 같은 폴더에 'countriesMBTI_16types.csv'를 추가하세요.")
    st.stop()

# ---------------------------
# 데이터 미리보기
# ---------------------------
st.subheader("📋 데이터 미리보기")
st.dataframe(df.head(), use_container_width=True)

# ---------------------------
# 사용자 선택
# ---------------------------
st.sidebar.header("⚙️ 분석 설정")
mbti_types = [c for c in df.columns if c != "Country"]
selected_type = st.sidebar.selectbox("분석할 MBTI 유형을 선택하세요:", mbti_types)

# ---------------------------
# 상위 10개국 계산
# ---------------------------
top10 = df.nlargest(10, selected_type)[["Country", selected_type]].sort_values(by=selected_type, ascending=True)

# ---------------------------
# Altair 시각화
# ---------------------------
st.subheader(f"🏆 {selected_type} 유형 비율이 높은 국가 TOP 10")

chart = (
    alt.Chart(top10)
    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
    .encode(
        x=alt.X(selected_type, title=f"{selected_type} 비율(%)"),
        y=alt.Y("Country", sort="-x", title="국가"),
        color=alt.Color(selected_type, scale=alt.Scale(scheme="blues")),
        tooltip=["Country", selected_type],
    )
    .properties(width=700, height=400)
    .interactive()
)

st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.markdown("""
💡 **Tip:**  
- CSV 파일을 업로드하면 즉시 그래프가 갱신됩니다.  
- 사이드바에서 다른 MBTI 유형을 선택할 수 있습니다.  
""")
