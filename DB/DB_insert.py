import dbinsert
import vga_ref
import vga_cleaner

print("데이터베이스 삽입시작")
dbinsert.run()
print("0원 제거 시작")
vga_cleaner.run()
print("데이터 정제시작")
vga_ref.run()
