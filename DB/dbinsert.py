import csv
import os
import mysql.connector
import re  # 정규 표현식 사용
from db_config import get_connection

def extract_price(price_info):
    """ '정품_'이 포함된 데이터에서 마지막 숫자 가격만 추출하는 함수 """
    if "정품_" in price_info:
        matches = re.findall(r'(\d{1,3}(?:,\d{3})*)', price_info)
        if matches:
            return matches[-1].replace(",", "")
    elif price_info.replace(",", "").isdigit():
        return price_info.replace(",", "")
    return None

def insert_csv_to_db(file_path, table_name, cursor):
    """ 주어진 CSV 파일에서 가격을 추출하고 지정된 테이블에 삽입 """
    if not os.path.exists(file_path):
        print(f"파일 없음: {file_path}")
        return

    try:
        with open(file_path, encoding="utf-8") as file:
            reader = csv.reader(file)
            headers = next(reader)
            dates = headers[2:]

            for row in reader:
                try:
                    name = row[1]
                    for i, price_info in enumerate(row[2:], start=2):
                        try:
                            price = extract_price(price_info)
                            if price is not None:
                                date = dates[i - 2].split()[0]
                                cursor.execute(
                                    f"INSERT INTO {table_name} (name, date, price) VALUES (%s, %s, %s)",
                                    (name, date, int(price))
                                )
                        except (IndexError, ValueError) as e:
                            print(f"오류 발생: {price_info}, 이유: {e} -> 무시됨")
                        except mysql.connector.Error as err:
                            print(f"MySQL 삽입 오류: {err}")
                except Exception as e:
                    print(f"행 처리 중 예외 발생: {e}")

    except FileNotFoundError:
        print(f"파일을 찾을 수 없음: {file_path}")
    except Exception as e:
        print(f"예외 발생 ({file_path}): {e}")

def run():
    data_folder = "Data"
    table_mapping = {
        "VGA.csv": "vga_price",
    }

    conn = get_connection()
    cursor = conn.cursor()

    for year in range(2020, 2026):
        for month in range(1, 13):
            folder_name = f"{year}-{month:02d}"
            folder_path = os.path.join(data_folder, folder_name)
            if not os.path.exists(folder_path):
                continue

            for file_name, table_name in table_mapping.items():
                file_path = os.path.join(folder_path, file_name)
                insert_csv_to_db(file_path, table_name, cursor)

            print(f"{year}-{month} 삽입완료")

    conn.commit()
    conn.close()
    print("모든 데이터 삽입 완료 및 MySQL 연결 종료!")

if __name__ == "__main__":
    run()