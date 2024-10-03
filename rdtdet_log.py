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

