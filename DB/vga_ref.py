import mysql.connector
import statistics
import re
import numpy as np
from sklearn.cluster import KMeans
from db_config import get_connection

def extract_gpu_info(text):
    pattern = re.search(
        r'(지포스|라데온)\s+'  # 브랜드
        r'(GTX|GT|RTX|RX|XT|XTX)?\s*'  # 시리즈
        r'(\d{3,5})'  # 숫자 (모델 번호)
        r'(?:\s*(Ti|SUPER|XT|XTX))?'  # 옵션 (예: Ti, SUPER 등)
        r'(?:\s*(Ti|SUPER|XT|XTX))?'  # 옵션 (예: Ti, SUPER 등)
        r'.*?(\d{1,2}GB)',  # 메모리 크기
        text, re.IGNORECASE
    )
    if pattern:
        brand, prefix, number, suffix1, suffix2, memory = pattern.groups()
        suffixes = []
        for s in (suffix1, suffix2):
            if s and s.upper() not in [x.upper() for x in suffixes]:
                suffixes.append(s.upper())
        model = f"{brand} {prefix or ''}{number}"
        if suffixes:
            model += ' ' + ' '.join(suffixes)
        model += f" {memory}"
        return model.strip()
    return None

def run():
    conn = get_connection()
    cursor = conn.cursor()

    # 1. GPU만 정제해서 가져오기
    cursor.execute("SELECT name, date, price FROM vga_ref")
    rows = cursor.fetchall()

    grouped = {}
    for name, date, price in rows:
        refined = extract_gpu_info(name)
        if refined and price > 0:
            key = (refined, date)
            grouped.setdefault(key, []).append(price)

    # 2. 통계 계산 및 저장
    for (refined_name, date), price_list in grouped.items():
        if len(price_list) < 3 or len(set(price_list)) < 2:
            # 클러스터링 생략
            avg_price = round(sum(price_list) / len(price_list))
            min_price = min(price_list)
            max_price = max(price_list)
            std_dev = round(statistics.stdev(price_list), 2) if len(price_list) > 1 else 0.0
            cursor.execute(
                "INSERT INTO ref_vga_stats (name, date, avg_price, min_price, max_price, std_dev) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (refined_name, date, avg_price, min_price, max_price, std_dev)
            )
            continue

        # 클러스터링 (2단계: 보급형, 상급형)
        prices_np = np.array(price_list).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')  # 클러스터 수 2로 설정
        labels = kmeans.fit_predict(prices_np)

        # 그룹별 평균 계산
        cluster_data = {}
        for label in set(labels):
            cluster_prices = [price for price, l in zip(price_list, labels) if l == label]
            cluster_data[label] = {
                'prices': cluster_prices,
                'avg': np.mean(cluster_prices)
            }

        # 평균 기준으로 정렬
        sorted_labels = sorted(cluster_data.items(), key=lambda x: x[1]['avg'])

        # 보급/상급 등급 이름
        tiers = ["보급형", "상급형"]  # 2개의 등급으로 변경

        # 실제 클러스터 수에 맞게 매핑
        label_to_tier = {}
        for i, (label, _) in enumerate(sorted_labels):
            tier_name = tiers[i] if i < len(tiers) else f"{i + 1}단계"
            label_to_tier[label] = tier_name

        # 저장
        for label, info in cluster_data.items():
            prices = info['prices']
            tier = label_to_tier[label]
            avg_price = round(np.mean(prices))
            min_price = min(prices)
            max_price = max(prices)
            std_dev = round(statistics.stdev(prices), 2) if len(prices) > 1 else 0.0

            refined_with_tier = f"{refined_name} ({tier})"
            cursor.execute(
                "INSERT INTO ref_vga_stats (name, date, avg_price, min_price, max_price, std_dev) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (refined_with_tier, date, avg_price, min_price, max_price, std_dev)
            )

    conn.commit()
    cursor.close()
    conn.close()
    print(" 2단계 분류 통계 완료: ref_vga_stats 테이블에 저장됨.")

if __name__ == "__main__":
    run()
