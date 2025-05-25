import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from gru import assign_metadata, add_rolling_features, generate_all_sequences
from DB.db_config import df_all

def train_dat():
    # 1. 데이터 전처리
    df = assign_metadata(df_all.copy())
    df, _ = add_rolling_features(df)
    data = generate_all_sequences(df, seq_len=20)

    X = data['X_price']
    y = data['y']

    print("X contains NaN?", np.isnan(X).any()) # 결측값 검증
    print("y contains NaN?", np.isnan(y).any())
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 2. GRU 모델 구성
    model = Sequential()
    model.add(GRU(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='relu')) # relu / 음수 출력 방지

    model.compile(optimizer='adam', loss='mse')

    # 3. 학습
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)                                  # A. 기존 방식
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])
    # early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)              # B. lr 추가 방식
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    # model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop, reduce_lr])

    # 4. 저장
    model.save("Model/gru_model.h5")
    print("✅ 모델 저장 완료: gru_model.h5")

if __name__ == "__main__":
    train_dat()