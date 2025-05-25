import mysql.connector
import statistics
import re
import numpy as np
from sklearn.cluster import KMeans
from db_config import get_connection
import time
from collections import defaultdict
import csv

def extract_gpu_info(text):
    pattern = re.search(
        r'(지포스|라데온)\s+'  # 브랜드
        r'(GTX|RTX|RX)?\s*'  # 시리즈
        r'(\d{3,5})'  # 숫자 (모델 번호)
        r'(?:\s*(Ti|SUPER|XTX|XT(?=\s|$|\d)|GRE))?'  # 옵션1
        r'(?:\s*(Ti|SUPER|XTX|XT(?=\s|$|\d)|GRE))?'  # 옵션2
        r'.*?(\d{1,3}GB)',  # 메모리 크기 (1~3자리 숫자)
        text, re.IGNORECASE
    )
    if pattern:
        brand, prefix, number, suffix1, suffix2, memory = pattern.groups()
        suffixes = {s.upper() for s in (suffix1, suffix2) if s}
        model = f"{brand} {prefix or ''}{number}"
        if suffixes:
            model += ' ' + ' '.join(suffixes)
        model += f" {memory}"
        return model.strip(), brand.upper(), (prefix or '').upper(), number, frozenset(suffixes)
    return None, None, None, None, frozenset()

def remove_outliers_with_zscore(prices, threshold=2):
    if len(prices) < 2:
        return prices
    if np.var(prices) < 1e-6:
        return prices
    z_scores = (prices - np.mean(prices)) / np.std(prices)
    return [price for price, z in zip(prices, z_scores) if abs(z) <= threshold]

def run():
    # 성능 점수 파일 불러오기 (옵션까지 추출)
    performance_map = {}
    with open("gpu_scores.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            name, score = row
            match = re.search(r'(지포스|라데온)\s+(GTX|RTX|RX)?\s*(\d{3,5})(?:\s*(Ti|SUPER|XT|XTX|GRE))?(?:\s*(Ti|SUPER|XT|XTX|GRE))?', name, re.IGNORECASE)
            if match:
                brand, prefix, number, suffix1, suffix2 = match.groups()
                suffixes = {s.upper() for s in (suffix1, suffix2) if s}
                key = (brand.upper(), (prefix or '').upper(), number, frozenset(suffixes))
                performance_map[key] = int(score.strip())

    start_time = time.time()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name, date, price FROM vga_ref")
    rows = cursor.fetchall()

    grouped = defaultdict(list)
    for name, date, price in rows:
        refined_full, brand, prefix, number, suffixes = extract_gpu_info(name)
        if refined_full and price > 0:
            key = (f"{refined_full}", date, brand, prefix, number, suffixes)
            grouped.setdefault(key, []).append(price)

    bulk_insert_data = []

    for (refined_name, date, brand, prefix, number, suffixes), price_list in grouped.items():
        if len(price_list) < 2:
            continue

        cleaned_prices = remove_outliers_with_zscore(np.array(price_list))
        if len(cleaned_prices) < 1:
            continue

        prices_np = np.array(cleaned_prices).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(prices_np)

        cluster_data = {}
        for label in set(labels):
            cluster_prices = [price for price, l in zip(cleaned_prices, labels) if l == label]
            cluster_data[label] = {'prices': cluster_prices, 'avg': np.mean(cluster_prices)}

        sorted_labels = sorted(cluster_data.items(), key=lambda x: x[1]['avg'])
        tiers = ["보급형", "상급형"]
        label_to_tier = {label: tiers[i] for i, (label, _) in enumerate(sorted_labels)}

        for label, info in cluster_data.items():
            tier = label_to_tier[label]
            avg_price = round(np.mean(info['prices']))
            min_price = min(info['prices'])
            max_price = max(info['prices'])
            std_dev = round(statistics.stdev(info['prices']), 2) if len(info['prices']) > 1 else 0.0
            refined_with_tier = f"{refined_name} ({tier})"

            # 옵션까지 포함한 성능 점수 매칭
            perf_key = (brand, prefix, number, suffixes)
            std_performance = performance_map.get(perf_key, None)

            bulk_insert_data.append((
                refined_with_tier, date, avg_price, min_price, max_price, std_dev, std_performance
            ))

    if bulk_insert_data:
        cursor.executemany(
            "INSERT INTO ref_vga_stats "
            "(name, date, avg_price, min_price, max_price, std_dev, std_performance) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            bulk_insert_data
        )

    conn.commit()
    cursor.close()
    conn.close()
    end_time = time.time()
    print(f"분류 및 통계, 성능 점수 추가 완료. 소요시간: {round(end_time - start_time, 2)}초")

if __name__ == "__main__":
    run()