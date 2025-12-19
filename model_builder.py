
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class TrafficPredictionModel:
    """CNN-LSTM 기반 교통 혼잡 예측 모델"""

    def __init__(self, time_steps=10, n_features=4):
        """
        Parameters:
        -----------
        time_steps : int
            과거 몇 개의 시간 단계를 사용할지 (기본: 10)
        n_features : int
            입력 특성 개수 (speed, volume, occupancy, weather)
        """
        self.time_steps = time_steps
        self.n_features = n_features
        self.model = None
        self.history = None

    def build_model(self):
        """CNN-LSTM 하이브리드 모델 구축"""
        print("\n🏗️  CNN-LSTM 모델 구축 중...")

        # 입력층
        inputs = Input(shape=(self.time_steps, self.n_features),
                      name='traffic_input')

        # CNN 블록 1: 지역적 패턴 추출
        x = Conv1D(filters=64, kernel_size=2, activation='relu',
                   padding='same', name='conv1d_1')(inputs)
        x = BatchNormalization(name='batch_norm_1')(x)
        x = Dropout(0.2, name='dropout_1')(x)

        # CNN 블록 2: 더 복잡한 패턴 추출
        x = Conv1D(filters=32, kernel_size=2, activation='relu',
                   padding='same', name='conv1d_2')(x)
        x = BatchNormalization(name='batch_norm_2')(x)
        x = Dropout(0.2, name='dropout_2')(x)

        # LSTM 블록: 시간적 의존성 학습
        x = LSTM(50, return_sequences=False, name='lstm_layer')(x)
        x = Dropout(0.3, name='dropout_3')(x)

        # Dense 블록: 최종 예측
        x = Dense(25, activation='relu', name='dense_1')(x)
        x = Dropout(0.2, name='dropout_4')(x)

        # 출력층: 혼잡도 예측 (0~1)
        outputs = Dense(1, activation='sigmoid', name='congestion_output')(x)

        # 모델 생성
        self.model = Model(inputs=inputs, outputs=outputs, name='Traffic_CNN_LSTM')

        # 모델 컴파일
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )

        print("✅ 모델 구축 완료!")
        print(f"\n📊 모델 구조:")
        self.model.summary()

        return self.model

    def get_callbacks(self):
        """학습 콜백 함수 정의"""
        callbacks = [
            # 조기 종료: 10 epoch 동안 개선 없으면 중단
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),

            # 학습률 감소: 5 epoch 동안 개선 없으면 학습률 감소
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),

            # 최적 모델 저장
            ModelCheckpoint(
                'best_traffic_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        return callbacks

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """모델 학습"""
        print("\n🎯 모델 학습 시작!")
        print(f"   - 학습 데이터: {X_train.shape[0]} 샘플")
        print(f"   - 검증 데이터: {X_val.shape[0]} 샘플")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch Size: {batch_size}\n")

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(),
            verbose=1
        )

        print("\n✅ 학습 완료!")

        return self.history

    def evaluate(self, X_test, y_test):
        """모델 평가"""
        print("\n📈 모델 평가 중...")

        results = self.model.evaluate(X_test, y_test, verbose=0)

        print(f"   ✓ Test Loss (MSE): {results[0]:.4f}")
        print(f"   ✓ Test MAE: {results[1]:.4f}")

        return results

    def predict(self, X):
        """예측 수행"""
        return self.model.predict(X, verbose=0)

# 테스트
if __name__ == "__main__":
    model_builder = TrafficPredictionModel(time_steps=10, n_features=4)
    model = model_builder.build_model()
