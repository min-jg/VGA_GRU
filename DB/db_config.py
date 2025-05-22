import pymysql
from sqlalchemy import create_engine, text
import pandas as pd

# DB 설정을 한 곳에서 관리
DB_USER = 'root'
DB_PASSWORD = '0000'
DB_NAME = 'danawa_vga'
DB_HOST = 'localhost'
DB_CHARSET = 'utf8'
DB_PORT = 3306

# SQLAlchemy 엔진 반환
def get_engine():
    return create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset={DB_CHARSET}"
    )

# pymysql 커넥션 반환
def get_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        charset=DB_CHARSET
    )

query = text("""
    SELECT name, date, avg_price 
    FROM ref_vga_stats
    WHERE avg_price > 0 AND name LIKE :pattern
      AND date BETWEEN :start_date AND :end_date
    ORDER BY name, date
""")

engine = get_engine()

df_all = pd.read_sql(query, engine, params={
    "pattern": "%RTX%",
    "start_date": "2020-01-01",
    "end_date": "2025-03-31"
})

df_all['date'] = pd.to_datetime(df_all['date'])