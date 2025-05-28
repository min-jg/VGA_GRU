import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors
import joblib

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from DB.db_config import df_all
from gru import assign_metadata, add_rolling_features, generate_all_sequences
from train import train_dat

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

class PricePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GPU 가격 예측기 (GRU 기반)")
        self.root.geometry("1000x800")

        self.df = df_all.copy()
        self.df = assign_metadata(self.df)
        self.df, self.scalers = add_rolling_features(self.df)                   # A. series > series 복원
        # self.df, self.scalers, self.name_scaler = add_rolling_features(self.df) # B. series > name 복원

        self.seq_len = 20
        self.gpu_list = sorted(self.df['name'].unique())

        # self.name_scaler = joblib.load("Model/name_scaler.pkl")  # ✅ 추가
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.root, text="그래픽카드 선택:").pack(pady=10)
        self.gpu_combo = ttk.Combobox(self.root, values=self.gpu_list, width=50)
        self.gpu_combo.pack()

        self.predict_btn = ttk.Button(self.root, text="2개월 예측하기", command=self.predict)
        self.predict_btn.pack(pady=20)

        self.result_label = ttk.Label(self.root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

    def predict_all_and_save(self):
        model = load_model("Model/gru_model.h5")

        os.makedirs("IMG", exist_ok=True)

        for gpu_name in self.gpu_list:
            df_gpu = self.df[self.df['name'] == gpu_name].copy()
            if len(df_gpu) <= self.seq_len:
                print(f"{gpu_name}: 데이터 부족으로 예측 생략")
                continue

            sequence = df_gpu.iloc[:self.seq_len].copy()
            preds = []
            dates = []

            total_steps = (len(df_gpu) - self.seq_len) + 60
            current_date = df_gpu['date'].min() + pd.Timedelta(days=self.seq_len)

            series_id = df_gpu['series_id'].iloc[0]
            scaler_price = self.scalers[series_id]['price']

            for i in range(total_steps):
                X_input = sequence[['price_scaled', 'mean_scaled_7', 'std_scaled_7',
                                    'mean_scaled_30', 'std_scaled_30']].values
                X_input = np.expand_dims(X_input, axis=0)

                pred_scaled = model.predict(X_input, verbose=0)[0][0]
                pred_real = scaler_price.inverse_transform([[pred_scaled]])[0][0]
                preds.append(round(pred_real))
                dates.append(current_date)
                current_date += pd.Timedelta(days=1)

                if i + self.seq_len < len(df_gpu):
                    next_row = df_gpu.iloc[i + self.seq_len].copy()
                else:
                    next_row = sequence.iloc[-1].copy()

                next_row['price_scaled'] = pred_scaled
                next_row_df = pd.DataFrame([next_row])

                sequence = pd.concat([sequence, next_row_df], ignore_index=True)
                sequence = sequence[-self.seq_len:]

            # 저장 경로 지정
            save_path = os.path.join("IMG", f"{gpu_name}.png")
            self.draw_plot(df_gpu, dates, preds, save_path=save_path)
            print(f"{gpu_name} 예측 이미지 저장 완료")

    def predict(self):
        gpu_name = self.gpu_combo.get()
        if not gpu_name:
            self.result_label.config(text="그래픽카드를 선택하세요.")
            return

        df_gpu = self.df[self.df['name'] == gpu_name].copy()
        if len(df_gpu) <= self.seq_len:
            self.result_label.config(text="데이터가 부족하여 예측할 수 없습니다.")
            return
        # 정규화 확인 코드
        print(f"[{gpu_name}] 실제 가격 범위: {df_gpu['avg_price'].min():,.0f} ~ {df_gpu['avg_price'].max():,.0f}")

        # 모델 없으면 학습
        if not os.path.exists("Model/gru_model.h5"):
            from train import train_dat
            train_dat()

        model = load_model("Model/gru_model.h5")

        # 시퀀스 시작: 가장 처음부터 예측
        sequence = df_gpu.iloc[:self.seq_len].copy()
        preds = []
        dates = []

        # 예측 구간 = 과거 전체 + 미래 60일
        total_steps = (len(df_gpu) - self.seq_len) + 60
        current_date = df_gpu['date'].min() + pd.Timedelta(days=self.seq_len)

        # 역정규화용 스케일러
        series_id = df_gpu['series_id'].iloc[0]             # A. series scale
        scaler_price = self.scalers[series_id]['price']

        # scaler_price = self.scalers[gpu_name]['price']      # B. name scale

        # scaler_price = self.name_scaler[gpu_name]['price']  # C. 이름 기준 역정규화

        for i in range(total_steps):
            X_input = sequence[['price_scaled', 'mean_scaled_7', 'std_scaled_7',
                                'mean_scaled_30', 'std_scaled_30']].values
            X_input = np.expand_dims(X_input, axis=0)

            pred_scaled = model.predict(X_input, verbose=0)[0][0]             # A. 가격 자체를 예측
            pred_real = scaler_price.inverse_transform([[pred_scaled]])[0][0]

            # ✅ warm-up 현상 진단: 첫 입력 대비 예측값 확인
            if i == 0:
                last_scaled = sequence['price_scaled'].iloc[-1]
                print(f"🔥 Warm-up 진단용")
                print(f" - 입력 마지막 price_scaled: {last_scaled:.4f}")
                print(f" - 첫 예측 pred_scaled: {pred_scaled:.4f}")
                print(f" - 첫 예측 역정규화 결과: {pred_real:,.0f}원")

            preds.append(round(pred_real))

            # last_scaled_price = sequence['price_scaled'].iloc[-1]  # B. 변화량 예측 / 마지막 시점의 정규화 된 가격
            # pred_delta = model.predict(X_input, verbose=0)[0][0]
            # pred_scaled = last_scaled_price + pred_delta  # 누적 예측값
            #
            # # 예측값 역정규화 및 반올림    ~ B
            # pred_real = scaler_price.inverse_transform([[pred_scaled]])[0][0]
            # pred_real = round(max(pred_real, 0))
            # preds.append(pred_real)

            # 날짜 기록
            dates.append(current_date)
            current_date += pd.Timedelta(days=1)

            # 다음 입력 시퀀스 준비
            if i + self.seq_len < len(df_gpu):
                next_row = df_gpu.iloc[i + self.seq_len].copy()
            else:
                next_row = sequence.iloc[-1].copy()

            next_row['price_scaled'] = pred_scaled  # ✅ 업데이트된 가격 사용
            next_row_df = pd.DataFrame([next_row])

            sequence = pd.concat([sequence, next_row_df], ignore_index=True)
            sequence = sequence[-self.seq_len:]

        self.draw_plot(df_gpu, dates, preds)
        self.result_label.config(text=f"{gpu_name} 예측 완료")

    def draw_plot(self, df_gpu, dates, preds, save_path=None):
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        plt.subplots_adjust(top=0.95, bottom=0.15)  # 하단 여백 확보

        ax.plot(df_gpu['date'], df_gpu['avg_price'], label='실제 평균가', color='blue')
        line_predict = ax.plot(dates, preds, label='예측가 (전체)', color='red', linestyle='--')

        ax.axvline(df_gpu['date'].max(), color='gray', linestyle='--', alpha=0.5)

        ax.set_title('가격 추세 및 2개월 예측')
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격 (정규화된 값)')
        ax.legend()

        fig.autofmt_xdate(rotation=30)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))

        # ✅ 오차율 및 절댓값 차이 계산
        common_len = min(len(df_gpu), len(dates))
        actual_prices = df_gpu['avg_price'].values[-common_len:]
        pred_prices = preds[:common_len]

        abs_diffs = np.abs(np.array(pred_prices) - np.array(actual_prices))
        mean_abs_diff = np.mean(abs_diffs)
        mean_pct_diff = np.mean((abs_diffs / actual_prices) * 100)

        # ✅ 오차 정보 텍스트 구성 및 표시
        error_text = f"평균 오차율: {mean_pct_diff:.2f}% | 평균 가격 차이: {mean_abs_diff:,.0f}원"
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.5, -0.21, error_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='center', bbox=props)

        if save_path:  # ✅ 저장용일 경우
            fig.savefig(save_path)
            plt.close(fig)  # 메모리 절약을 위해 닫기
            return

        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=10)

        def format_tooltip(date, price):
            return f"날짜: {date.strftime('%Y-%m-%d')}\n예측가: {int(round(price)):,}원"

        # cursor = mplcursors.cursor(line_predict, hover=True)
        # @cursor.connect("add")
        # def on_add(sel):
        #     index = int(round(sel.index))
        #     date = dates
        #     price = preds
        #
        #     r_price = None
        #     if 'avg_price' in locals() and date in df_gpu['date']:
        #         r_date = df_gpu['date'].index(date)
        #         r_price = df_gpu['avg_price']
        #
        #     tooltip = f"날짜: {date.strftime('%Y-%m-%d')}\n예측가: {int(price):,}원"
        #     if r_price is not None:
        #         tooltip += f"\n실제가: {int(r_price):,}원"
        #
        #     sel.annotation.set(text=tooltip)

        cursor1 = mplcursors.cursor(ax.lines[0], hover=True)

        @cursor1.connect("add")
        def on_add(sel):
            index = int(round(sel.index))
            date = df_gpu['date'].iloc[index]
            price = df_gpu['avg_price'].iloc[index]
            sel.annotation.set(text=format_tooltip(date, price))

        cursor2 = mplcursors.cursor(ax.lines[1], hover=True)

        @cursor2.connect("add")
        def on_add(sel):
            index = int(round(sel.index))
            if index < len(dates):
                date = dates[index]
                price = preds[index]
                sel.annotation.set(text=format_tooltip(date, price))

if __name__ == "__main__":
    root = tk.Tk()
    if not os.path.exists("Model/gru_model.h5"):
        print("모델이 없어 학습을 시작합니다...")
        train_dat()
    print(f"학습된 모델 {'gru_model'} 이 존재합니다.")
    app = PricePredictorApp(root)
    root.mainloop()

    # Tkinter GUI 종료 후 전체 이미지 저장
    app.predict_all_and_save()