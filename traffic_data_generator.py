
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TrafficDataGenerator:
    """실제와 유사한 교통 데이터 생성 클래스"""

    def __init__(self, days=30, samples_per_hour=12):
        """
        Parameters:
        -----------
        days : int
            생성할 데이터의 일수 (기본: 30일)
        samples_per_hour : int
            시간당 샘플 수 (기본: 12 = 5분 간격)
        """
        self.days = days
        self.samples_per_hour = samples_per_hour
        self.total_samples = days * 24 * samples_per_hour

    def generate_base_pattern(self):
        """기본 교통 패턴 생성 (시간대별 특성)"""
        samples = []
        start_time = datetime.now() - timedelta(days=self.days)

        for i in range(self.total_samples):
            current_time = start_time + timedelta(minutes=5*i)
            hour = current_time.hour
            day_of_week = current_time.weekday()  # 0=월요일, 6=일요일

            # 시간대별 기본 교통량 패턴
            if 7 <= hour <= 9:  # 출근 시간대
                base_congestion = 0.7 + np.random.normal(0, 0.1)
            elif 17 <= hour <= 19:  # 퇴근 시간대
                base_congestion = 0.8 + np.random.normal(0, 0.1)
            elif 12 <= hour <= 13:  # 점심 시간대
                base_congestion = 0.5 + np.random.normal(0, 0.1)
            elif 22 <= hour or hour <= 5:  # 심야 시간대
                base_congestion = 0.1 + np.random.normal(0, 0.05)
            else:  # 평시
                base_congestion = 0.3 + np.random.normal(0, 0.1)

            # 요일 효과 (주말은 패턴이 다름)
            if day_of_week >= 5:  # 주말
                base_congestion *= 0.7

            samples.append(base_congestion)

        return np.array(samples)

    def add_weather_effect(self, base_pattern):
        """날씨 효과 추가"""
        weather = np.random.choice([0, 1, 2], size=self.total_samples,
                                   p=[0.7, 0.2, 0.1])  # 0=맑음, 1=비, 2=눈

        weather_effect = np.ones(self.total_samples)
        weather_effect[weather == 1] = 1.3  # 비: 30% 증가
        weather_effect[weather == 2] = 1.5  # 눈: 50% 증가

        return base_pattern * weather_effect, weather

    def calculate_traffic_features(self, congestion_level):
        """혼잡도로부터 교통 특성 계산"""
        # 속도 (혼잡할수록 느림)
        speed = 80 * (1 - congestion_level) + 20  # 20~100 km/h
        speed = np.clip(speed, 10, 100)

        # 교통량 (혼잡도에 비례)
        volume = congestion_level * 200 + np.random.normal(0, 20)
        volume = np.clip(volume, 0, 250)

        # 점유율 (혼잡도에 비례)
        occupancy = congestion_level * 90 + np.random.normal(0, 5)
        occupancy = np.clip(occupancy, 0, 100)

        return speed, volume, occupancy

    def generate_complete_dataset(self):
        """완전한 교통 데이터셋 생성"""
        print("🚗 교통 데이터 생성 중...")

        # 기본 혼잡도 패턴
        congestion = self.generate_base_pattern()

        # 날씨 효과 추가
        congestion, weather = self.add_weather_effect(congestion)

        # 혼잡도를 0~1 범위로 정규화
        congestion = np.clip(congestion, 0, 1)

        # 교통 특성 계산
        data = []
        for i in range(self.total_samples):
            speed, volume, occupancy = self.calculate_traffic_features(congestion[i])

            data.append({
                'timestamp': i,
                'speed': speed,
                'volume': volume,
                'occupancy': occupancy,
                'weather': weather[i],
                'congestion_level': congestion[i]
            })

        df = pd.DataFrame(data)

        print(f"✅ {len(df)}개 샘플 생성 완료!")
        print(f"   - 기간: {self.days}일")
        print(f"   - 샘플링: 5분 간격")
        print(f"   - 혼잡도 범위: {df['congestion_level'].min():.2f} ~ {df['congestion_level'].max():.2f}")

        return df

# 간단한 테스트
if __name__ == "__main__":
    generator = TrafficDataGenerator(days=30)
    df = generator.generate_complete_dataset()
    df.to_csv('traffic_data.csv', index=False)
    print("\n💾 traffic_data.csv 파일로 저장 완료!")
