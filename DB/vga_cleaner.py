import pandas as pd
import numpy as np
from db_config import get_engine

def run():
    # DB 연결
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM vga_price", engine)
    df = df.sort_values(by=['name', 'date'])

    # 0을 np.nan으로 대체 (핵심 수정)
    df['price'] = df['price'].replace(0, np.nan)

    def clean_group(group):
        is_na = group['price'].isna()
        na_runs = (is_na != is_na.shift()).cumsum()
        run_lengths = is_na.groupby(na_runs).transform('sum')

        # 연속된 NaN 구간 삭제
        group = group[~((is_na) & (run_lengths > 1))]

        # float으로 변환 후 보간
        group['price'] = group['price'].astype(float)
        group['price'] = group['price'].interpolate(method='linear', limit_direction='both')
        return group

    df_cleaned = (
        df.groupby('name', group_keys=False)
        .apply(clean_group)
        .reset_index(drop=True)
    )

    # num 컬럼 제거 후 다시 추가
    if 'num' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=['num'])

    df_cleaned.insert(0, 'num', range(1, len(df_cleaned) + 1))
    df_cleaned.to_sql("vga_ref", con=engine, if_exists="replace", index=False)

    print("✅ 정제 완료: vga_ref 테이블에 저장됨.")

if __name__ == "__main__":
    run()
