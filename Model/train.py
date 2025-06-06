import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from gru import assign_metadata, add_rolling_features, generate_all_sequences
from DB.db_config import df_all
import joblib


def train_dat():
    # 1. 데이터 전처리
    df = assign_metadata(df_all.copy())
    df, _ = add_rolling_features(df)                    # A. series 단일 scale
    # df, scaler, name_scaler = add_rolling_features(df)  # B. 역정규화 name scale 추가
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
    # early_stop = EarlyStopping(patience=10, restore_best_weights=True)                                  # A. 기존 방식
    # history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, shuffle=False, callbacks=[early_stop])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)              # B. lr 추가 방식
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, shuffle=False, callbacks=[early_stop, reduce_lr])

    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Model/loss_plot.png")
    plt.show()

    # 4. 저장
    model.save("Model/gru_model.h5")
    print("✅ 모델 저장 완료: gru_model.h5")

    # # 5. 이름 기반 scaler도 저장
    # joblib.dump(name_scaler, "Model/name_scaler.pkl")

if __name__ == "__main__":
    train_dat()