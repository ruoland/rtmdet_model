import sys
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

class TimetableViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("시간표 뷰어")
        self.setGeometry(100, 100, 1000, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.table_widget = QTableWidget()
        layout.addWidget(self.table_widget)

        self.load_timetable()

    def load_timetable(self):
        try:
            with open('ocr_cell/timetable.json', 'r', encoding='utf-8') as file:
                data = json.load(file)

            # 요일 순서 정의
            day_order = ['월', '화', '수', '목', '금', '토', '일']
            periods = sorted(set(entry['period'] for entry in data))

            # 테이블 크기 설정
            self.table_widget.setRowCount(len(periods))
            self.table_widget.setColumnCount(len(day_order))

            # 열 헤더 설정 (요일)
            self.table_widget.setHorizontalHeaderLabels(day_order)

            # 행 헤더 설정 (교시)
            self.table_widget.setVerticalHeaderLabels(periods)

            # 데이터 채우기
            for entry in data:
                if entry['day_of_week'] in day_order:
                    row = periods.index(entry['period'])
                    col = day_order.index(entry['day_of_week'])
                    consecutive_classes = entry.get('consecutive_classes', 1)
                    
                    item = QTableWidgetItem(f"{entry['subject']}\n{entry['start_time']}-{entry['end_time']}")
                    item.setTextAlignment(Qt.AlignCenter)
                    
                    # 셀 타입에 따른 배경색 설정
                    if entry['cell_type'] == 'merged_cell':
                        item.setBackground(QColor(200, 200, 255))  # 연한 파란색
                    elif entry['cell_type'] == 'overflow_cell':
                        item.setBackground(QColor(255, 200, 200))  # 연한 빨간색
                    
                    self.table_widget.setItem(row, col, item)
                    
                    # 긴 셀 처리
                    if consecutive_classes > 1:
                        self.table_widget.setSpan(row, col, consecutive_classes, 1)

            # 셀 크기 조정
            self.table_widget.resizeColumnsToContents()
            self.table_widget.resizeRowsToContents()

        except Exception as e:
            print(f"Error loading timetable: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = TimetableViewer()
    viewer.show()
    sys.exit(app.exec_())
