import re
import csv

def matches_valid_pattern(name):
    # 패턴 끝에 용량 (예: 8GB, 12GB) 허용
    pattern = re.compile(
        r'^(지포스|라데온)\s+(GTX|RTX|RX)?\s*\d{3,5}(?:\s*(?:Ti|SUPER|XT|XTX|GRE)){0,2}(?:\s*\d{1,3}GB)?$',
        re.IGNORECASE
    )
    return bool(pattern.match(name))

def extract_name_and_score(lines):
    results = []

    for i in range(0, len(lines), 4):
        try:
            name = lines[i].strip()
            score_line = lines[i + 2].strip()
            score = int(re.sub(r"[^\d]", "", score_line))

            # 브랜드 이름 치환
            if "GeForce" in name:
                name = name.replace("GeForce", "지포스")
            elif "Radeon" in name:
                name = name.replace("Radeon", "라데온")
            else:
                continue  # 기타 브랜드 제외

            # 이름 정리: 공백 정리
            name = re.sub(r'\s+', ' ', name).strip()

            # 패턴 검사: 정확히 패턴에 맞는 것만 허용
            if not matches_valid_pattern(name):
                continue  # 패턴 불일치 → 제외

            results.append((name, score))
        except (IndexError, ValueError):
            continue

    return results

# ✅ 텍스트 파일에서 읽기
with open("input.txt", "r", encoding="utf-8") as file:
    lines = file.read().strip().splitlines()

# ✅ 데이터 처리
gpu_data = extract_name_and_score(lines)

# ✅ CSV 파일로 저장
with open("gpu_scores.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Score"])
    writer.writerows(gpu_data)

print("✅ 처리 완료: 'gpu_scores.csv' 파일이 생성되었습니다.")
