import logging

# 로거 객체 생성
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 핸들러 생성 및 설정
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# 포맷터 생성 및 핸들러에 추가
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# 로거에 핸들러 추가
logger.addHandler(handler)

# 새로운 함수를 추가합니다.
def print_formatted_table_structure(table_structure):
    for i, row in enumerate(table_structure):
        for j, cell in enumerate(row):
            if cell:
                content = cell.get('text', '')[:20]  # 첫 20자만 표시
                print(f"{i+1}행 {j+1}열: {content}")
            else:
                print(f"{i+1}행 {j+1}열: 빈 셀")
        print()  # 행 사이에 빈 줄 추가