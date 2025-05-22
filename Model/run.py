import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors

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
        self.df, self.scalers = add_rolling_features(self.df)

        self.seq_len = 20
        self.gpu_list = sorted(self.df['name'].unique())

        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.root, text="그래픽카드 선택:").pack(pady=10)
        self.gpu_combo = ttk.Combobox(self.root, values=self.gpu_list, width=50)
        self.gpu_combo.pack()

        self.predict_btn = ttk.Button(self.root, text="2개월 예측하기", command=self.predict)
        self.predict_btn.pack(pady=20)

        self.result_label = ttk.Label(self.root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

    def predict(self):
        gpu_name = self.gpu_combo.get()
        if not gpu_name:
            self.result_label.config(text="그래픽카드를 선택하세요.")
            return

        df_gpu = self.df[self.df['name'] == gpu_name].copy()
        if len(df_gpu) <= self.seq_len:
            self.result_label.config(text="데이터가 부족하여 예측할 수 없습니다.")
            return

        # 모델 없으면 학습
        if not os.path.exists("Model/gru_model.keras"):
            from train import train_dat
            train_dat()

        model = load_model("Model/gru_model.keras")

        # 시퀀스 시작: 가장 처음부터 예측
        sequence = df_gpu.iloc[:self.seq_len].copy()
        preds = []
        dates = []

        # 예측 구간 = 과거 전체 + 미래 60일
        total_steps = (len(df_gpu) - self.seq_len) + 60
        current_date = df_gpu['date'].min() + pd.Timedelta(days=self.seq_len)

        # 역정규화용 스케일러
        series_id = df_gpu['series_id'].iloc[0]
        scaler_price = self.scalers[series_id]['price']

        for i in range(total_steps):
            X_input = sequence[['price_scaled', 'mean_scaled_7', 'std_scaled_7',
                                'mean_scaled_30', 'std_scaled_30']].values
            X_input = np.expand_dims(X_input, axis=0)

            pred_scaled = model.predict(X_input, verbose=0)[0][0]
            pred_real = scaler_price.inverse_transform([[pred_scaled]])[0][0]
            preds.append(pred_real)

            # 날짜 기록
            dates.append(current_date)
            current_date += pd.Timedelta(days=1)

            # 다음 입력 시퀀스 준비
            if i + self.seq_len < len(df_gpu):
                next_row = df_gpu.iloc[i + self.seq_len].copy()
            else:
                next_row = sequence.iloc[-1:].copy()

            next_row['price_scaled'] = pred_scaled
            sequence = pd.concat([sequence, next_row.to_frame().T], ignore_index=True)
            sequence = sequence[-self.seq_len:]

        self.draw_plot(df_gpu, dates, preds)
        self.result_label.config(text=f"{gpu_name} 예측 완료")

    def draw_plot(self, df_gpu, dates, preds):
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

        ax.plot(df_gpu['date'], df_gpu['avg_price'], label='실제 평균가', color='blue')
        ax.plot(dates, preds, label='예측가 (전체)', color='red', linestyle='--')

        ax.plot(dates[:1], preds[:1], 'o', color='green', label='예측 시작점')
        ax.axvline(df_gpu['date'].max(), color='gray', linestyle='--', alpha=0.5)

        ax.set_title('가격 추세 및 2개월 예측')
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격 (정규화된 값)')
        ax.legend()

        fig.autofmt_xdate(rotation=30)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))

        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=10)

        def format_tooltip(date, price):
            return f"날짜: {date.strftime('%Y-%m-%d')}\n예측가: {price:.4f}"

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
    if not os.path.exists("gru_model.keras"):
        print("모델이 없어 학습을 시작합니다...")
        train_dat()
    app = PricePredictorApp(root)
    root.mainloop()
