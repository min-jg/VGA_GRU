import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from gru import assign_metadata, add_rolling_features, generate_all_sequences
from DB.db_config import df_all

def train_dat():
    # 1. 데이터 전처리
    df = assign_metadata(df_all.copy())
    df, _ = add_rolling_features(df)
    data = generate_all_sequences(df, seq_len=20)

    X = data['X_price']
    y = data['y']

    # 2. GRU 모델 구성
    model = Sequential()
    model.add(GRU(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    # 3. 학습
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stop])

    # 4. 저장
    model.save("Model/gru_model.keras")
    print("✅ 모델 저장 완료: gru_model.keras")

if __name__ == "__main__":
    train_dat()