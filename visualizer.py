
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 한글 폰트 설정 (Windows 11)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 스타일 설정
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

class TrafficVisualizer:
    """교통 데이터 및 예측 결과 시각화"""

    def __init__(self):
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#06A77D',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'info': '#6A4C93'
        }

    def plot_data_distribution(self, df):
        """데이터 분포 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('📊 교통 데이터 분포 분석', fontsize=20, fontweight='bold', y=1.02)

        features = ['speed', 'volume', 'occupancy', 'weather', 'congestion_level']

        for idx, feature in enumerate(features):
            row = idx // 3
            col = idx % 3

            # 히스토그램
            axes[row, col].hist(df[feature], bins=50, color=self.colors['primary'],
                               alpha=0.7, edgecolor='black')
            axes[row, col].set_title(f'{feature.upper()} 분포', fontsize=14, fontweight='bold')
            axes[row, col].set_xlabel('값', fontsize=12)
            axes[row, col].set_ylabel('빈도', fontsize=12)
            axes[row, col].grid(True, alpha=0.3)

            # 통계 정보 추가
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            axes[row, col].axvline(mean_val, color=self.colors['danger'],
                                  linestyle='--', linewidth=2, label=f'평균: {mean_val:.2f}')
            axes[row, col].legend()

        # 마지막 subplot은 전체 통계
        axes[1, 2].axis('off')
        stats_text = f"""
        📈 전체 통계

        총 샘플 수: {len(df):,}
        평균 속도: {df['speed'].mean():.1f} km/h
        평균 교통량: {df['volume'].mean():.1f} 대
        평균 점유율: {df['occupancy'].mean():.1f}%
        평균 혼잡도: {df['congestion_level'].mean():.3f}
        """
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=14,
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('01_data_distribution.png', dpi=300, bbox_inches='tight')
        print("✅ 데이터 분포 그래프 저장: 01_data_distribution.png")
        plt.close(fig) # close the figure to prevent it from showing up in other contexts
        return fig # Return the figure object

    def plot_time_series(self, df, samples=500):
        """시계열 데이터 시각화"""
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        fig.suptitle('📈 시간에 따른 교통 패턴 변화', fontsize=20, fontweight='bold')

        # 샘플링 (너무 많으면 느림)
        df_sample = df.head(samples)
        x = df_sample['timestamp']

        # 속도
        axes[0].plot(x, df_sample['speed'], color=self.colors['primary'], linewidth=2)
        axes[0].set_title('🚗 속도 (Speed)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('km/h', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].fill_between(x, df_sample['speed'], alpha=0.3, color=self.colors['primary'])

        # 교통량
        axes[1].plot(x, df_sample['volume'], color=self.colors['success'], linewidth=2)
        axes[1].set_title('🚙 교통량 (Volume)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('대/시간', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].fill_between(x, df_sample['volume'], alpha=0.3, color=self.colors['success'])

        # 점유율
        axes[2].plot(x, df_sample['occupancy'], color=self.colors['warning'], linewidth=2)
        axes[2].set_title('📊 점유율 (Occupancy)', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('%', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].fill_between(x, df_sample['occupancy'], alpha=0.3, color=self.colors['warning'])

        # 혼잡도
        axes[3].plot(x, df_sample['congestion_level'], color=self.colors['danger'], linewidth=2)
        axes[3].set_title('🚦 혼잡도 (Congestion Level)', fontsize=14, fontweight='bold')
        axes[3].set_ylabel('0~1', fontsize=12)
        axes[3].set_xlabel('시간 (5분 단위)', fontsize=12)
        axes[3].grid(True, alpha=0.3)
        axes[3].fill_between(x, df_sample['congestion_level'], alpha=0.3, color=self.colors['danger'])

        plt.tight_layout()
        plt.savefig('02_time_series.png', dpi=300, bbox_inches='tight')
        print("✅ 시계열 그래프 저장: 02_time_series.png")
        plt.close(fig) # close the figure to prevent it from showing up in other contexts
        return fig # Return the figure object

    def plot_training_history(self, history):
        """학습 과정 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('🎓 모델 학습 과정', fontsize=20, fontweight='bold')

        # Loss
        axes[0].plot(history.history['loss'], label='학습 손실',
                    color=self.colors['primary'], linewidth=2, marker='o', markersize=4)
        axes[0].plot(history.history['val_loss'], label='검증 손실',
                    color=self.colors['danger'], linewidth=2, marker='s', markersize=4)
        axes[0].set_title('손실 (Loss) 변화', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('MSE', fontsize=12)
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # MAE
        axes[1].plot(history.history['mae'], label='학습 MAE',
                    color=self.colors['success'], linewidth=2, marker='o', markersize=4)
        axes[1].plot(history.history['val_mae'], label='검증 MAE',
                    color=self.colors['warning'], linewidth=2, marker='s', markersize=4)
        axes[1].set_title('평균 절대 오차 (MAE) 변화', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].legend(fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('03_training_history.png', dpi=300, bbox_inches='tight')
        print("✅ 학습 과정 그래프 저장: 03_training_history.png")
        plt.close(fig) # close the figure to prevent it from showing up in other contexts
        return fig # Return the figure object

    def plot_prediction_results(self, y_true, y_pred, sample_size=200):
        """예측 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('🎯 교통혼잡 예측 결과', fontsize=20, fontweight='bold', y=1.02)

        # 샘플링
        y_true_sample = y_true[:sample_size]
        y_pred_sample = y_pred[:sample_size]

        # 1. 실제 vs 예측 시계열
        x = np.arange(len(y_true_sample))
        axes[0, 0].plot(x, y_true_sample, label='실제값',
                       color=self.colors['primary'], linewidth=2, marker='o', markersize=3)
        axes[0, 0].plot(x, y_pred_sample, label='예측값',
                       color=self.colors['danger'], linewidth=2, marker='s', markersize=3, alpha=0.7)
        axes[0, 0].set_title('실제 혼잡도 vs 예측 혼잡도', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('샘플 번호', fontsize=12)
        axes[0, 0].set_ylabel('혼잡도 (0~1)', fontsize=12)
        axes[0, 0].legend(fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 산점도 (Scatter Plot)
        axes[0, 1].scatter(y_true, y_pred, alpha=0.5, color=self.colors['info'], s=30)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='완벽한 예측선')
        axes[0, 1].set_title('실제값 vs 예측값 산점도', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('실제 혼잡도', fontsize=12)
        axes[0, 1].set_ylabel('예측 혼잡도', fontsize=12)
        axes[0, 1].legend(fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 오차 분포
        errors = y_true - y_pred.flatten()
        axes[1, 0].hist(errors, bins=50, color=self.colors['success'],
                       alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='오차 0')
        axes[1, 0].set_title('예측 오차 분포', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('오차', fontsize=12)
        axes[1, 0].set_ylabel('빈도', fontsize=12)
        axes[1, 0].legend(fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 성능 지표
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        axes[1, 1].axis('off')
        metrics_text = f"""
        📊 모델 성능 지표

        ✓ MAE (평균 절대 오차)
          {mae:.4f}

        ✓ MSE (평균 제곱 오차)
          {mse:.4f}

        ✓ RMSE (평균 제곱근 오차)
          {rmse:.4f}

        ✓ R² (결정 계수)
          {r2:.4f}

        🎯 예측 정확도: {r2*100:.2f}%
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=16,
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig('04_prediction_results.png', dpi=300, bbox_inches='tight')
        print("✅ 예측 결과 그래프 저장: 04_prediction_results.png")
        plt.close(fig) # close the figure to prevent it from showing up in other contexts
        return mae, mse, rmse, r2, fig # Return the figure object and metrics

    def plot_interactive_comparison(self, y_true, y_pred, sample_size=500):
        """인터랙티브 비교 그래프 (Plotly)"""
        y_true_sample = y_true[:sample_size]
        y_pred_sample = y_pred[:sample_size].flatten()

        fig = go.Figure()

        # 실제값
        fig.add_trace(go.Scatter(
            x=list(range(len(y_true_sample))),
            y=y_true_sample,
            mode='lines+markers',
            name='실제 혼잡도',
            line=dict(color='royalblue', width=2),
            marker=dict(size=4)
        ))

        # 예측값
        fig.add_trace(go.Scatter(
            x=list(range(len(y_pred_sample))),
            y=y_pred_sample,
            mode='lines+markers',
            name='예측 혼잡도',
            line=dict(color='crimson', width=2, dash='dash'),
            marker=dict(size=4)
        ))

        fig.update_layout(
            title='🚦 실시간 교통혼잡 예측 비교 (Interactive)',
            xaxis_title='시간 단계 (5분 간격)',
            yaxis_title='혼잡도 (0~1)',
            font=dict(size=14),
            hovermode='x unified',
            template='plotly_white',
            width=1200,
            height=600
        )

        fig.write_html('05_interactive_comparison.html')
        print("✅ 인터랙티브 그래프 저장: 05_interactive_comparison.html")
        # fig.show() # Plotly figures are not displayed directly by st.pyplot
        return fig # Return Plotly figure object

# 테스트
if __name__ == "__main__":
    visualizer = TrafficVisualizer()
    print("시각화 모듈 준비 완료!")
