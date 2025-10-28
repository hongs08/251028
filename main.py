import streamlit as st
import pandas as pd
import altair as alt

# ---------------------------
# 기본 설정
# ---------------------------
st.set_page_config(
    page_title="MBTI Type by Country",
    layout="wide",
)

st.title("🌍 MBTI 유형별 상위 국가 분석")
st.markdown("""
이 앱은 국가별 MBTI 분포 데이터를 기반으로 특정 유형이 높은 상위 10개국을 시각적으로 보여줍니다.
""")

# ---------------------------
# 데이터 불러오기
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("countriesMBTI_16types.csv")
    return df

df = load_data()

# ---------------------------
# 데이터 확인
# ---------------------------
st.subheader("📋 데이터 미리보기")
st.dataframe(df.head(), use_container_width=True)

# ---------------------------
# 사용자 입력
# ---------------------------
st.sidebar.header("⚙️ 분석 설정")

# 선택 가능한 MBTI 유형 목록
mbti_types = [col for col in df.columns if col != "Country"]
selected_type = st.sidebar.selectbox("분석할 MBTI 유형을 선택하세요:", mbti_types)

# ---------------------------
# 상위 10개 국가 추출
# ---------------------------
top10 = df.nlargest(10, selected_type)[["Country", selected_type]].copy()
top10 = top10.sort_values(by=selected_type, ascending=True)  # 시각적 정렬 (낮은→높은)

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

# ---------------------------
# 추가 정보
# ---------------------------
st.markdown("---")
st.markdown("""
💡 **Tip:**  
- 왼쪽 사이드바에서 다른 MBTI 유형을 선택하면 즉시 그래프가 업데이트됩니다.  
- 막대 위에 마우스를 올리면 국가별 정확한 값을 확인할 수 있습니다.
""")
