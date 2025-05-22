import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from DB.db_config import df_all
from gru import assign_metadata, add_rolling_features, generate_all_sequences

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
        self.df, _ = add_rolling_features(self.df)

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

        model = load_model("gru_model.h5")  # 학습된 GRU 모델 로드
        sequence = df_gpu[-self.seq_len:].copy()

        future_dates = pd.bdate_range(start=df_gpu['date'].max() + pd.Timedelta(days=1), periods=60)
        preds = []

        for _ in range(60):
            price = sequence[['price_scaled', 'mean_scaled_7', 'std_scaled_7', 'mean_scaled_30', 'std_scaled_30']].values
            X_input = np.expand_dims(price, axis=0)

            pred = model.predict(X_input, verbose=0)[0][0]
            preds.append(pred)

            new_row = sequence.iloc[-1:].copy()
            new_row['price_scaled'] = pred
            sequence = pd.concat([sequence, new_row], ignore_index=True)
            sequence = sequence[-self.seq_len:]

        self.draw_plot(df_gpu, future_dates, preds)
        self.result_label.config(text=f"{gpu_name}의 2개월 예측 완료")

    def draw_plot(self, df_gpu, future_dates, pred_prices):
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

        ax.plot(df_gpu['date'], df_gpu['avg_price'], label='과거 평균가', color='blue')
        ax.plot(future_dates, pred_prices, label='예측가 (60일)', color='red', linestyle='--')
        ax.plot(future_dates[:1], pred_prices[:1], 'o', color='green', label='예측 시작점')
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
            date = future_dates[index]
            price = pred_prices[index]
            sel.annotation.set(text=format_tooltip(date, price))


if __name__ == "__main__":
    root = tk.Tk()
    app = PricePredictorApp(root)
    root.mainloop()
