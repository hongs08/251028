import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="MBTI 국가 분석", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("countriesMBTI_16types.csv")
    return df

# 데이터 불러오기
df = load_data()

st.title("🌍 MBTI 유형별 국가 분석 대시보드")

st.markdown("MBTI 유형별로 어떤 국가에서 높은 비율을 차지하는지를 시각적으로 분석합니다.")

# 데이터 확인
if st.checkbox("데이터 미리보기"):
    st.dataframe(df)

# MBTI 유형 선택
types = [col for col in df.columns if col not in ["Country", "Total"]]
selected_type = st.selectbox("분석할 MBTI 유형을 선택하세요:", types)

# Top N 선택
top_n = st.slider("표시할 상위 국가 수", 5, 20, 10)

# 선택한 유형 기준 정렬
top_df = df.sort_values(by=selected_type, ascending=False).head(int(top_n))

# 시각화
st.subheader(f"🌟 {selected_type} 유형 비율이 높은 국가 TOP {top_n}")

chart = (
    alt.Chart(top_df)
    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
    .encode(
        x=alt.X(selected_type, title=f"{selected_type} 비율(%)"),
        y=alt.Y("Country", sort="-x", title="국가"),
        color=alt.Color(selected_type, scale=alt.Scale(scheme="tealblues")),
        tooltip=["Country", selected_type]
    )
    .properties(height=500)
)

st.altair_chart(chart, use_container_width=True)

# 상세 데이터 표시
st.markdown("#### 📊 상세 데이터")
st.dataframe(top_df.reset_index(drop=True))
