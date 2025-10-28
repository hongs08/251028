import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="MBTI Type by Country (Robust)", layout="wide")
st.title("🌍 MBTI 유형별 상위 국가 분석 — (디버그/견고 버전)")

# 파일 업로드 위젯 (또는 로컬 파일 사용)
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드 (옵션)", type=["csv"])
use_local = st.sidebar.checkbox("앱 디렉터리의 'countriesMBTI_16types.csv' 사용", value=True)

@st.cache_data
def load_data(upload, use_local_file=True):
    if upload is not None:
        df = pd.read_csv(upload)
    else:
        if use_local_file:
            df = pd.read_csv("countriesMBTI_16types.csv")
        else:
            raise FileNotFoundError("업로드된 파일이 없고 로컬 파일 사용이 비활성화되어 있습니다.")
    # 기본 정리: 컬럼명 공백 제거
    df.columns = df.columns.str.strip()
    return df

# 데이터 로드 (오류 시 메시지와 중단)
try:
    df = load_data(uploaded_file, use_local)
except FileNotFoundError as e:
    st.error("CSV 파일을 불러올 수 없습니다. 파일을 업로드하거나 앱 폴더에 'countriesMBTI_16types.csv'를 넣으세요.")
    st.stop()
except Exception as e:
    st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
    st.stop()

# 간단한 진단 정보 표시 — 문제 해결에 도움됨
st.subheader("데이터 진단 정보")
st.write("행 × 열:", df.shape)
st.write("컬럼 목록:", df.columns.tolist())
st.write("상위 5개 행:")
st.dataframe(df.head())

st.write("컬럼별 자료형:")
dtypes = df.dtypes.apply(lambda x: str(x))
st.write(dtypes)

# Country 컬럼 유연 탐지
country_col = None
if "Country" in df.columns:
    country_col = "Country"
else:
    for c in df.columns:
        if "country" in c.lower() or "nation" in c.lower():
            country_col = c
            break
    # 마지막 수단: 문자열형 컬럼 중 고유값이 많은 컬럼을 국가로 추정
    if country_col is None:
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for c in obj_cols:
            if df[c].nunique() >= max(10, int(0.5 * min(df.shape[0], 100))):
                country_col = c
                break

if country_col is None:
    st.error("국가를 나타내는 열을 자동으로 찾지 못했습니다. 'Country' 컬럼명이 존재하는지 확인하세요.")
    st.stop()

st.write(f"판정된 국가 컬럼: **{country_col}** (고유값 수: {df[country_col].nunique()})")

# MBTI 후보 열 추출 — Country 컬럼 제외한 나머지 중 숫자열로 변환 가능한 것들
candidate_cols = [c for c in df.columns if c != country_col]
# 강제로 숫자형 변환 (문자/빈값 때문에 그래프가 안 뜨는 문제 방지)
for c in candidate_cols:
    # 숫자로 변환 불가하면 NaN이 됨
    df[c] = pd.to_numeric(df[c], errors="coerce")

# NaN이 많은 열 필터링 (사용자가 원하면 포함 가능하도록 알림)
nan_ratios = df[candidate_cols].isna().mean().sort_values(ascending=False)
high_nan = nan_ratios[nan_ratios > 0.5]
if not high_nan.empty:
    st.warning("몇몇 열에 결측치가 많습니다(>50%). 결과가 왜곡될 수 있습니다:\n" + high_nan.to_frame(name="nan_ratio").to_string())

# 기본적으로 NaN을 0으로 채움(원하면 다른 전략으로 바꿀 수 있음)
df[candidate_cols] = df[candidate_cols].fillna(0)

# 사이드바: MBTI 유형 선택 (숫자형으로 변환된 열만)
numeric_mbti_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
if not numeric_mbti_cols:
    st.error("MBTI 유형으로 사용할 숫자형 열을 찾지 못했습니다.")
    st.stop()

selected_type = st.sidebar.selectbox("분석할 MBTI 유형을 선택하세요:", numeric_mbti_cols)

# 상위 10개 국가 추출
top_n = st.sidebar.number_input("상위 N (국가 수)", min_value=1, max_value=50, value=10)
top_df = df[[country_col, selected_type]].copy()
# 정렬 및 상위 N
top_df = top_df.sort_values(by=selected_type, ascending=False).head(int(top_n_
