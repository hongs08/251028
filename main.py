import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # Import matplotlib for plt.close()
import matplotlib as mpl
import platform
import matplotlib.font_manager as fm
import requests
import tempfile

# --- 한글 폰트 설정: 시스템 폰트 우선, 없으면 NanumGothic을 런타임에 다운로드하여 등록 ---
def set_korean_font():
    preferred = ["Malgun Gothic", "Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "NanumSquare"]
    available = {f.name for f in fm.fontManager.ttflist}
    # 시스템에 이미 있는 폰트 사용
    for name in preferred:
        if name in available:
            mpl.rcParams['font.family'] = name
            mpl.rcParams['axes.unicode_minus'] = False
            return

    # 없으면 NanumGothic-Regular.ttf 를 다운로드하여 추가 시도
    try:
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            tmp = tempfile.gettempdir()
            ttf_path = os.path.join(tmp, "NanumGothic-Regular.ttf")
            with open(ttf_path, "wb") as f:
                f.write(r.content)
            fm.fontManager.addfont(ttf_path)
            # 폰트 매니저 재구성 (필요시)
            fm._rebuild()
            mpl.rcParams['font.family'] = "NanumGothic"
            mpl.rcParams['axes.unicode_minus'] = False
            return
    except Exception:
        pass

    # 모든 시도 실패 시: 기본 폰트에 unicode_minus만 설정
    mpl.rcParams['axes.unicode_minus'] = False

# 실행
set_korean_font()
# -----------------------------------------------

import tensorflow as tf # Import tensorflow for model summary

# 사용자 정의 모듈 import
from traffic_data_generator import TrafficDataGenerator
from model_builder import TrafficPredictionModel
from visualizer import TrafficVisualizer

st.set_page_config(layout="wide")

# --- Custom CSS for improved aesthetics ---
st.markdown("""
<style>
/* 전체 페이지 폰트 및 배경색 설정 */
body {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
    color: #333;
    background-color: #f0f2f6;
}

/* 메인 타이틀 */
h1 {
    color: #2E86AB;
    text-align: center;
    font-size: 2.8em;
    margin-bottom: 0.5em;
}

/* 서브 타이틀 및 섹션 헤더 */
h3 {
    color: #4F8BA0;
    font-size: 1.6em;
    border-bottom: 2px solid #D6E8EE;
    padding-bottom: 0.3em;
    margin-top: 1.5em;
    margin-bottom: 1em;
}

h2 {
    color: #2E86AB;
    font-size: 2em;
    margin-top: 1.5em;
    margin-bottom: 1em;
}

/* 사이드바 헤더 */
.stSidebar .st-emotion-cache-vk33as h2 {
    color: #06A77D;
    font-size: 1.8em;
    border-bottom: 2px solid #A2D9D2;
    padding-bottom: 0.3em;
}

/* 정보 메시지 (st.info) */
.stAlert.info {
    background-color: #e0f2f7;
    color: #2E86AB;
    border-left: 5px solid #2E86AB;
}

/* 성공 메시지 (st.success) */
.stAlert.success {
    background-color: #e6ffed;
    color: #06A77D;
    border-left: 5px solid #06A77D;
}

/* 경고 메시지 (st.warning) */
.stAlert.warning {
    background-color: #fff8e1;
    color: #F18F01;
    border-left: 5px solid #F18F01;
}

/* 버튼 스타일 */
.stButton>button {
    background-color: #2E86AB;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.6em 1.2em;
    font-size: 1.1em;
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    background-color: #1F6B8A;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

/* expander 헤더 스타일 */
.streamlit-expanderHeader {
    font-weight: bold;
    color: #4F8BA0;
    font-size: 1.1em;
}

/* metric 스타일 */
.stMetric {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    text-align: center;
}
.stMetric label {
    color: #666;
    font-size: 0.9em;
}
.stMetric .stMetricValue {
    color: #2E86AB;
    font-size: 1.8em;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

st.title("🚗 AI 기반 교통혼잡 예측 시스템")
st.markdown("""
    ### 🚦 CNN-LSTM 딥러닝 모델을 활용한 실시간 교통혼잡 예측 데모
""")

st.sidebar.header("⚙️ 설정")

def create_sequences(X, y, time_steps=10):
    """시계열 데이터를 LSTM 입력 형식으로 변환"""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# 1. 데이터 생성 섹션
st.sidebar.subheader("1. 데이터 생성 설정")
days = st.sidebar.slider("데이터 생성 기간 (일)", min_value=7, max_value=365, value=30, step=1)
samples_per_hour_options = {"5분 간격 (12)": 12, "10분 간격 (6)": 6, "15분 간격 (4)": 4, "30분 간격 (2)": 2, "60분 간격 (1)": 1}
samples_per_hour_selected = st.sidebar.selectbox(
    "시간당 샘플 수 (간격)",
    options=list(samples_per_hour_options.keys()),
    format_func=lambda x: x,
    index=0 # default to 5분 간격 (12)
)
samples_per_hour = samples_per_hour_options[samples_per_hour_selected]

# 캐싱된 데이터 생성 함수
@st.cache_data
def generate_data(days, samples_per_hour):
    generator = TrafficDataGenerator(days=days, samples_per_hour=samples_per_hour)
    df = generator.generate_complete_dataset()
    df.to_csv('traffic_data.csv', index=False)
    return df


if 'df' not in st.session_state:
    st.session_state.df = None

# st.columns를 사용하여 데이터 생성 버튼과 미리보기를 나란히 배치
col1, col2 = st.columns([0.3, 0.7])

with col1:
    st.subheader("📊 1단계: 교통 데이터 생성")
    if st.button("데이터 생성", key="generate_data_button") or st.session_state.df is None:
        with st.spinner(f"지정된 기간({days}일)의 교통 데이터를 생성 중입니다..."):
            try:
                st.session_state.df = generate_data(days, samples_per_hour)
                st.success("✅ 데이터 생성 및 'traffic_data.csv' 저장 완료!")
            except Exception as e:
                st.error(f"데이터 생성 중 오류 발생: {e}")

with col2:
    if st.session_state.df is not None:
        with st.expander("생성된 데이터 미리보기 (상위 5행)", expanded=False):
            st.dataframe(st.session_state.df.head())
        st.success(f"총 {len(st.session_state.df)}개의 교통 데이터 샘플이 준비되었습니다.")
    else:
        st.info("좌측 사이드바에서 설정을 조절하고, '데이터 생성' 버튼을 눌러 작업을 시작하세요.")

# 2. 데이터 탐색 및 시각화 섹션
st.subheader("📈 2단계: 데이터 탐색적 분석 (EDA) 및 시각화")

if st.session_state.df is not None:
    visualizer = TrafficVisualizer()

    st.markdown("**데이터 분포**")
    with st.spinner("데이터 분포 그래프를 생성 중입니다..."):
        fig_dist = visualizer.plot_data_distribution(st.session_state.df)
        st.pyplot(fig_dist)
        plt.close(fig_dist) # Close the figure

    st.markdown("**시계열 패턴**")
    st.info("데이터가 너무 많을 경우 시각화에 시간이 오래 걸릴 수 있습니다. 초기 500 샘플만 표시합니다.")
    with st.spinner("시계열 패턴 그래프를 생성 중입니다..."):
        fig_ts = visualizer.plot_time_series(st.session_state.df, samples=500)
        st.pyplot(fig_ts)
        plt.close(fig_ts) # Close the figure
else:
    st.warning("데이터가 생성되지 않았습니다. 먼저 '데이터 생성' 버튼을 눌러주세요.")

# 3. 데이터 전처리 섹션
st.sidebar.subheader("2. 모델 학습 설정")
time_steps = st.sidebar.slider(
    "시퀀스 길이 (Time Steps)",
    min_value=5,
    max_value=30,
    value=10,
    step=1,
    help="과거 몇 개의 시간 단계를 입력으로 사용할지 설정합니다."
)

features = ['speed', 'volume', 'occupancy', 'weather']
target = 'congestion_level'
st.sidebar.write(f"선택된 입력 특성: {', '.join(features)}")
st.sidebar.write(f"예측 타겟: {target}")

# 세션 상태 초기화
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
    st.session_state.y_train = None
    st.session_state.X_val = None
    st.session_state.y_val = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.scaler_X = None
    st.session_state.scaler_y = None
    st.session_state.model = None
    st.session_state.history = None
    st.session_state.y_pred_original = None
    st.session_state.y_test_original = None

if st.sidebar.button("데이터 전처리"): # Changed from initial load to button click for clearer flow
    if st.session_state.df is not None:
        st.subheader("🔧 3단계: 데이터 전처리")
        with st.spinner("데이터 정규화, 시퀀스 생성 및 학습/검증/테스트 분할 중..."):
            try:
                X = st.session_state.df[features].values
                y = st.session_state.df[target].values

                # 정규화
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()

                X_scaled = scaler_X.fit_transform(X)
                y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

                # 시퀀스 생성
                X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

                # 학습/검증/테스트 분할
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X_seq, y_seq, test_size=0.2, random_state=42
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=0.125, random_state=42  # 0.125 * 0.8 = 0.1
                )

                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.X_val = X_val
                st.session_state.y_val = y_val
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.scaler_X = scaler_X
                st.session_state.scaler_y = scaler_y

                st.success("✅ 데이터 전처리 완료!")
                with st.expander("전처리된 데이터 요약", expanded=False):
                    st.write(f"- 입력 특성: {features}")
                    st.write(f"- 출력 타겟: {target}")
                    st.write(f"- 시퀀스 길이 (Time Steps): {time_steps}")
                    st.write(f"- 학습 데이터 shape: X={X_train.shape}, y={y_train.shape}")
                    st.write(f"- 검증 데이터 shape: X={X_val.shape}, y={y_val.shape}")
                    st.write(f"- 테스트 데이터 shape: X={X_test.shape}, y={y_test.shape}")

            except Exception as e:
                st.error(f"데이터 전처리 중 오류 발생: {e}")
    else:
        st.warning("데이터가 생성되지 않았습니다. 먼저 '데이터 생성' 버튼을 눌러주세요.")

# 4. 모델 구축 및 학습 섹션
st.sidebar.subheader("3. 모델 학습 및 평가")
epochs = st.sidebar.number_input("에포크 수", min_value=10, max_value=200, value=100, step=10)
batch_size = st.sidebar.number_input("배치 크기", min_value=16, max_value=128, value=32, step=16)

if st.sidebar.button("모델 학습 시작"): # key="train_model_button"
    if st.session_state.X_train is not None:
        st.subheader("🤖 4단계: CNN-LSTM 모델 학습")
        with st.spinner("모델을 구축하고 학습 중입니다. 다소 시간이 소요될 수 있습니다..."):
            try:
                model_builder = TrafficPredictionModel(time_steps=time_steps, n_features=len(features))
                model = model_builder.build_model()
                st.session_state.model = model

                with st.expander("모델 구조 요약", expanded=False):
                    # Capturing model summary to display in Streamlit
                    from io import StringIO
                    buffer = StringIO()
                    model.summary(print_fn=lambda x: buffer.write(x + '\n'))
                    st.text(buffer.getvalue())

                history = model_builder.train(
                    st.session_state.X_train, st.session_state.y_train,
                    st.session_state.X_val, st.session_state.y_val,
                    epochs=epochs,
                    batch_size=batch_size
                )
                st.session_state.history = history

                st.success("✅ 모델 학습 완료!")
                visualizer = TrafficVisualizer()
                st.markdown("**학습 과정 시각화**")
                fig_history = visualizer.plot_training_history(st.session_state.history)
                st.pyplot(fig_history)
                plt.close(fig_history) # Close the figure

            except Exception as e:
                st.error(f"모델 학습 중 오류 발생: {e}")
    else:
        st.warning("데이터가 전처리되지 않았습니다. 먼저 '데이터 전처리' 버튼을 눌러주세요.")

# 5. 모델 평가 및 예측 시각화 섹션
if st.sidebar.button("모델 평가 및 예측"): # key="evaluate_predict_button"
    if st.session_state.model is not None and st.session_state.X_test is not None:
        st.subheader("📊 5단계: 모델 성능 평가 및 예측")
        visualizer = TrafficVisualizer()

        with st.spinner("모델을 평가하고 예측 결과를 생성 중입니다..."):
            try:
                # 평가
                results = st.session_state.model.evaluate(st.session_state.X_test, st.session_state.y_test, verbose=0)
                # Removed st.write for test results as these are also part of overall metrics below.

                # 예측
                y_pred_scaled = st.session_state.model.predict(st.session_state.X_test, verbose=0)

                # 역정규화
                y_test_original = st.session_state.scaler_y.inverse_transform(st.session_state.y_test.reshape(-1, 1)).flatten()
                y_pred_original = st.session_state.scaler_y.inverse_transform(y_pred_scaled).flatten()

                st.session_state.y_pred_original = y_pred_original
                st.session_state.y_test_original = y_test_original

                st.success("✅ 모델 평가 및 예측 완료!")
                st.markdown("**예측 결과 시각화**")
                mae, mse, rmse, r2, fig_pred = visualizer.plot_prediction_results(
                    y_test_original,
                    y_pred_original
                )
                st.pyplot(fig_pred) # Display the matplotlib figure
                plt.close(fig_pred) # Close the figure

                st.markdown("**인터랙티브 예측 비교 (Plotly)**")
                st.info("브라우저에서 05_interactive_comparison.html 파일을 열어 인터랙티브 그래프를 확인할 수 있습니다.")
                plotly_fig = visualizer.plot_interactive_comparison(
                    y_test_original,
                    y_pred_original,
                    sample_size=min(len(y_test_original), 300)
                )
                st.plotly_chart(plotly_fig, use_container_width=True)

                st.subheader("🎉 최종 결과 요약")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(label="MAE (평균 절대 오차)", value=f"{mae:.4f}")
                with col2:
                    st.metric(label="MSE (평균 제곱 오차)", value=f"{mse:.4f}")
                with col3:
                    st.metric(label="RMSE (평균 제곱근 오차)", value=f"{rmse:.4f}")
                with col4:
                    st.metric(label="R\u00b2 (결정 계수)", value=f"{r2:.4f}")
                st.success(f"**🎯 예측 정확도:** {r2*100:.2f}%")

                # 경제적 효과 추정
                st.markdown("**\ud83d\udcb0 경제적 효과 추정:**")
                base_cost = 65.2  # 2021년 기준 교통혼잡비용 (조원)
                reduction_rate = (r2 * 0.15)  # 예측 정확도 기반 혼잡 감소율 (보수적 추정 15%)
                economic_effect = base_cost * reduction_rate

                col_econ1, col_econ2, col_econ3 = st.columns(3)
                with col_econ1:
                    st.metric(label="한국 연간 교통혼잡비용", value=f"{base_cost:.1f}조원")
                with col_econ2:
                    st.metric(label="예상 혼잡 감소율", value=f"{reduction_rate*100:.1f}%")
                with col_econ3:
                    st.metric(label="연간 경제적 효과", value=f"{economic_effect:.2f}조원")

            except Exception as e:
                st.error(f"모델 평가 및 예측 중 오류 발생: {e}")
    else:
        st.warning("모델이 학습되지 않았거나 테스트 데이터가 준비되지 않았습니다. 먼저 '데이터 전처리' 및 '모델 학습 시작' 버튼을 눌러주세요.")
