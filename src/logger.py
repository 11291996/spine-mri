import os
import logging
import datetime

class TrainLogger:
    def __init__(self, log_dir="", prefix="train"):
        # 로그 저장할 디렉토리 생성 (없으면 자동 생성)
        os.makedirs(os.path.join(log_dir, 'logs'), exist_ok=True)
        
        # 각 프로세스가 독립적인 로그 파일을 생성하도록 시간 기반 파일명 설정
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = os.path.join(log_dir, 'logs', f"{prefix}_{timestamp}.log")

        # Logger 설정
        self.logger = logging.getLogger(log_filename)  # 개별 Logger 인스턴스 생성
        self.logger.setLevel(logging.INFO)  # 로그 레벨 설정
        
        # 로그 포맷 설정
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

        # 파일 핸들러 설정
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)

        # 로거에 핸들러 추가
        self.logger.addHandler(file_handler)

    def log(self, message):
        self.logger.info(message)
        