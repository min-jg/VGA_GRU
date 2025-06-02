# 데이터 처리 및 수치 연산 관련 라이브러리
import pandas as pd
import numpy as np
import re

# 데이터 정규화를 위한 MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from DB.db_config import df_all

# 시리즈별로 정규화하는 함수
def extract_ids(name):
    match = re.search(r'RTX(\d{4})', name)
    level_match = re.search(r'(\d{2})(?=\D|$)', name)
    capacity_match = re.search(r'(\d+)\s?GB', name.upper())
    if match and level_match:
        series = int(match.group(1)) // 1000
        level = int(level_match.group(1))
        capacity = int(capacity_match.group(1))
    else:
        series = -1
        level = -1
        capacity = -1
    return series, level, capacity

# 이름, 레벨, 용량 추출
def assign_metadata(df):
    df[['series_id', 'level_id', 'capacity_id']] = df['name'].apply(lambda x: pd.Series(extract_ids(x)))
    return df

# 정규화
def add_rolling_features(df):
    df.sort_values(['name', 'date'], inplace=True)

    # # A. 가공없이 피쳐만 생성
    # df['rolling_mean_7'] = df.groupby('name')['avg_price'].transform(lambda x: x.rolling(7).mean().bfill())
    # df['rolling_std_7'] = df.groupby('name')['avg_price'].transform(lambda x: x.rolling(7).std().bfill())
    # df['rolling_mean_30'] = df.groupby('name')['avg_price'].transform(lambda x: x.rolling(30).mean().bfill())
    # df['rolling_std_30'] = df.groupby('name')['avg_price'].transform(lambda x: x.rolling(30).std().bfill()) # ~A

    # # B. rolling feature 생성 후 NaN(결측값) 평균 값으로 처리
    # for window in [7, 30]:
    #     mean_col = f'rolling_mean_{window}'
    #     std_col = f'rolling_std_{window}'
    #
    #     df[mean_col] = (
    #         df.groupby('name')['avg_price']
    #         .transform(lambda x: x.rolling(window).mean())
    #         .bfill().ffill()
    #         .fillna(df['avg_price'].mean())
    #     )
    #     df[std_col] = (
    #         df.groupby('name')['avg_price']
    #         .transform(lambda x: x.rolling(window).std())
    #         .bfill().ffill()
    #         .fillna(0.0)
    #     )   # ~B

    # C. A&B 혼합 방식
    for window in [7, 30]:
        mean_col = f'rolling_mean_{window}'
        std_col = f'rolling_std_{window}'

        df[mean_col] = (
            df.groupby('name')['avg_price']
            .transform(lambda x: x.rolling(window).mean())
            .bfill().ffill()
        )

        df[std_col] = (
            df.groupby('name')['avg_price']
            .transform(lambda x: x.rolling(window).std())
            .bfill().ffill()
        )   # ~C

    scalers = {}
    # name_scaler = {}    # 역정규화를 위한 name scale

    for sid, group in df.groupby('series_id'):  # A. series scale
    # for name, group in df.groupby('name'):  # B. name scale
        # 정규화 확인 코드
        print(f"[series_id={sid}] 가격 범위: {group['avg_price'].min():,.0f} ~ {group['avg_price'].max():,.0f}")
        # print(f"[name={name}] 가격 범위: {group['avg_price'].min():,.0f} ~ {group['avg_price'].max():,.0f}")

        # A. 데이터 전체
        scaler_price = MinMaxScaler()
        scaler_mean_7 = MinMaxScaler() # 변경: 각 mean, std 피처에 개별 스케일러
        scaler_std_7 = MinMaxScaler()
        scaler_mean_30 = MinMaxScaler()
        scaler_std_30 = MinMaxScaler()
        df.loc[group.index, 'price_scaled'] = scaler_price.fit_transform(group[['avg_price']])
        df.loc[group.index, 'mean_scaled_7'] = scaler_mean_7.fit_transform(group[['rolling_mean_7']])
        df.loc[group.index, 'std_scaled_7'] = scaler_std_7.fit_transform(group[['rolling_std_7']])
        df.loc[group.index, 'mean_scaled_30'] = scaler_mean_30.fit_transform(group[['rolling_mean_30']])
        df.loc[group.index, 'std_scaled_30'] = scaler_std_30.fit_transform(group[['rolling_std_30']])


        # # B. 이상치 제외한 범위로 정규화 스케일러 fit
        # q_low = group['avg_price'].quantile(0.01)
        # q_high = group['avg_price'].quantile(0.99)
        # trimmed = group[(group['avg_price'] >= q_low) & (group['avg_price'] <= q_high)]
        #
        # # ✅ 각각 fit
        # scaler_price = MinMaxScaler()
        # scaler_price.fit(trimmed[['avg_price']])
        # scaler_mean = MinMaxScaler()
        # scaler_mean.fit(trimmed[['rolling_mean_7']])
        # scaler_std = MinMaxScaler()
        # scaler_std.fit(trimmed[['rolling_std_7']])
        # scaler_mean_30 = MinMaxScaler()
        # scaler_mean_30.fit(trimmed[['rolling_mean_30']])
        # scaler_std_30 = MinMaxScaler()
        # scaler_std_30.fit(trimmed[['rolling_std_30']])
        #
        # # ✅ 전체 데이터 transform
        # df.loc[group.index, 'price_scaled'] = scaler_price.transform(group[['avg_price']])
        # df.loc[group.index, 'mean_scaled_7'] = scaler_mean.transform(group[['rolling_mean_7']])
        # df.loc[group.index, 'std_scaled_7'] = scaler_std.transform(group[['rolling_std_7']])
        # df.loc[group.index, 'mean_scaled_30'] = scaler_mean_30.transform(group[['rolling_mean_30']])
        # df.loc[group.index, 'std_scaled_30'] = scaler_std_30.transform(group[['rolling_std_30']])

        scalers[sid] = { # A. series scale
        # scaler[name] = { # B. name scale
            'price': scaler_price,
            'mean_7': scaler_mean_7,
            'std_7': scaler_std_7,
            'mean_30': scaler_mean_30,
            'std_30': scaler_std_30
        }

    # # ✅ 예측 후 복원을 위한 name 단위 scaler 추가 저장
    # for name, group in df.groupby('name'):
    #     scaler_price = MinMaxScaler()
    #     scaler_price.fit(group[['avg_price']])
    #     name_scaler[name] = {'price': scaler_price}

    return df, scalers#, name_scaler  # name_scaler 추가

# 시퀸스 생성
def create_sequences(df_group, seq_len):
    X_price, X_series_id, X_level_id, X_capacity_id = [], [], [], []
    X_mean_7, X_std_7, X_mean_30, X_std_30, y = [], [], [], [], []

    series_id = df_group['series_id'].iloc[0]
    level_id = df_group['level_id'].iloc[0]
    capacity_id = df_group['capacity_id'].iloc[0]

    price = df_group['price_scaled'].values
    mean_7 = df_group['mean_scaled_7'].values
    std_7 = df_group['std_scaled_7'].values
    mean_30 = df_group['mean_scaled_30'].values
    std_30 = df_group['std_scaled_30'].values

    for i in range(len(df_group) - seq_len):
        X_p = np.stack([price[i:i+seq_len], mean_7[i:i+seq_len], std_7[i:i+seq_len],
                        mean_30[i:i+seq_len], std_30[i:i+seq_len]], axis=1)
        X_price.append(X_p)
        X_series_id.append([series_id] * seq_len)
        X_level_id.append([level_id] * seq_len)
        X_capacity_id.append([capacity_id] * seq_len)
        X_mean_7.append(mean_7[i:i+seq_len])
        X_std_7.append(std_7[i:i+seq_len])
        X_mean_30.append(mean_30[i:i+seq_len])
        X_std_30.append(std_30[i:i+seq_len])
        y.append(price[i + seq_len])                        # A. 가격 예측 로직 수정

        # delta = price[i + seq_len] - price[i + seq_len - 1] # B. 가격 변화량 기반 예측
        # y.append(delta)

    return X_price, X_series_id, X_level_id, X_capacity_id, X_mean_7, X_std_7, X_mean_30, X_std_30, y

# 전체 시퀸스
def generate_all_sequences(df, seq_len=20):
    X_price, X_series_id, X_level_id, X_capacity_id = [], [], [], []
    X_mean_7, X_std_7, X_mean_30, X_std_30, y = [], [], [], [], []

    for name, group in df.groupby('name'):
        xp, xs, xl, xc, xm7, xs7, xm30, xs30, yy = create_sequences(group, seq_len)
        X_price += xp
        X_series_id += xs
        X_level_id += xl
        X_capacity_id += xc
        X_mean_7 += xm7
        X_std_7 += xs7
        X_mean_30 += xm30
        X_std_30 += xs30
        y += yy

    return {
        'X_price': np.array(X_price),
        'X_series_id': np.array(X_series_id),
        'X_level_id': np.array(X_level_id),
        'X_capacity_id': np.array(X_capacity_id),
        'X_mean_7': np.array(X_mean_7),
        'X_std_7': np.array(X_std_7),
        'X_mean_30': np.array(X_mean_30),
        'X_std_30': np.array(X_std_30),
        'y': np.array(y)
    }
