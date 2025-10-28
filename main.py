import streamlit as st
import pandas as pd
import altair as alt

# ------------------------------
# 앱 기본 설정
# ------------------------------
st.set_page_config(
    page_title="시간대별 교통량 분석 대시보드",
    layout="wide",
    page_icon="🚗",
)

st.title("🚗 시간대별 교통량 분석 대시보드")
st.markdown("교통량 데이터를 시간대별로 분석하고 시각화하는 대시보드입니다.")

# ------------------------------
# 데이터 업로드 섹션
# ------------------------------
st.sidebar.header("📂 데이터 업로드")

uploaded_file = st.sidebar.file_uploader(
    "traffic_processed.csv 파일을 업로드하세요", type=["csv"]
)

# ------------------------------
# 데이터 불러오기 함수
# ------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding="utf-8")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.day_name(locale="ko_KR")  # 요일명
    return df

# ------------------------------
# 파일 업로드 여부 확인
# ------------------------------
if uploaded_file is not None:
    df = load_data(uploaded_file)

    # ------------------------------
    # 사이드바 필터 설정
    # ------------------------------
    st.sidebar.header("🔍 필터 설정")

    unique_sites = df["지점명"].unique()
    selected_site = st.sidebar.selectbox("지점 선택", ["전체"] + list(unique_sites))

    direction = st.sidebar.selectbox("유입/유출 선택", ["전체", "유입", "유출"])

    weekday_options = ["전체"] + df["weekday"].dropna().unique().tolist()
    selected_weekday = st.sidebar.selectbox("요일 선택", weekday_options)

    # ------------------------------
    # 필터 적용
    # ------------------------------
    filtered_df = df.copy()

    if selected_site != "전체":
        filtered_df = filtered_df[filtered_df["지점명"] == selected_site]

    if direction != "전체":
        filtered_df = filtered_df[filtered_df["유입/유출"] == direction]

    if selected_weekday != "전체":
        filtered_df = filtered_df[filtered_df["weekday"] == selected_weekday]

    # ------------------------------
    # 데이터 집계 (시간대별 평균 교통량)
    # ------------------------------
    hourly_df = (
        filtered_df.groupby("hour")["교통량"]
        .mean()
        .reset_index()
        .sort_values("hour")
    )

    # ------------------------------
    # Altair 시각화
    # ------------------------------
    st.subheader("⏰ 시간대별 평균 교통량")

    if len(hourly_df) > 0:
        chart = (
            alt.Chart(hourly_df)
            .mark_line(point=True, interpolate="monotone", color="#0072B2")
            .encode(
                x=alt.X("hour:O", title="시간대 (시)", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("교통량:Q", title="평균 교통량"),
                tooltip=["hour", "교통량"]
            )
            .properties(
                width=900,
                height=400,
                title=f"{selected_site if selected_site != '전체' else '전체 지점'} - 시간대별 평균 교통량"
            )
        )

        bar = (
            alt.Chart(hourly_df)
            .mark_bar(color="#56B4E9", opacity=0.6)
            .encode(
                x=alt.X("hour:O", title="시간대 (시)"),
                y=alt.Y("교통량:Q", title="평균 교통량"),
                tooltip=["hour", "교통량"]
            )
            .properties(width=900, height=400)
        )

        st.altair_chart(chart + bar, use_container_width=True)
    else:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")

    # ------------------------------
    # 상세 데이터 테이블
    # ------------------------------
    with st.expander("📊 시간대별 데이터 상세 보기"):
        st.dataframe(hourly_df)

    # ------------------------------
    # 주석 및 안내
    # ------------------------------
    st.markdown(
        """
        ---
        **설명**
        - 상단 그래프는 선택한 지점 및 조건에 따른 시간대별 평균 교통량을 나타냅니다.
        - 데이터는 사용자가 업로드한 `traffic_processed.csv` 파일을 기반으로 합니다.
        """
    )

else:
    st.info("📤 왼쪽 사이드바에서 `traffic_processed.csv` 파일을 업로드해주세요.")
