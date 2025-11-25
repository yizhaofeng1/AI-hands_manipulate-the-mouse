import sys
import cv2
import numpy as np
import time
import pyautogui
import mediapipe as mp
import math
import threading
import os

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QSlider, QGroupBox,
                             QTextEdit, QCheckBox, QGridLayout, QPushButton,
                             QStackedWidget, QFrame, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

# --- 环境设置 ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import winsound

    WINDOWS_SYSTEM = True
except ImportError:
    WINDOWS_SYSTEM = False


# ==========================================
# 1. 全局配置
# ==========================================
class Config:
    smoothing = 5
    frame_margin = 100
    click_threshold = 40
    sound_enabled = True
    show_landmarks = True


# ==========================================
# 2. 摄像头线程 (保持原有逻辑不变)
# ==========================================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def play_sound(self, freq, duration):
        if not Config.sound_enabled: return
        if WINDOWS_SYSTEM:
            threading.Thread(target=winsound.Beep, args=(freq, duration), daemon=True).start()

    def run(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(model_complexity=0, max_num_hands=1,
                               min_detection_confidence=0.7, min_tracking_confidence=0.5)

        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        wCam, hCam = 640, 480
        wScr, hScr = pyautogui.size()

        plocX, plocY = 0, 0
        clocX, clocY = 0, 0
        is_dragging = False
        last_click_time = 0
        last_page_time = 0
        PAGE_COOLDOWN = 1.2

        self.log_signal.emit("系统就绪 - AI 引擎已启动")

        while self._run_flag:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            frameR = Config.frame_margin
            cv2.rectangle(frame, (frameR, frameR), (w - frameR, h - frameR), (255, 0, 255), 2)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (100, h), (0, 255, 0), -1)
            cv2.rectangle(overlay, (w - 100, 0), (w, h), (0, 100, 255), -1)
            frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if Config.show_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    lm = hand_landmarks.landmark
                    x1, y1 = int(lm[8].x * w), int(lm[8].y * h)
                    x2, y2 = int(lm[12].x * w), int(lm[12].y * h)
                    thumb_x, thumb_y = int(lm[4].x * w), int(lm[4].y * h)
                    fingers = []
                    if lm[8].y < lm[6].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                    if lm[12].y < lm[10].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                    if lm[16].y < lm[14].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    if fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0:
                        x3 = np.interp(x1, (frameR, w - frameR), (0, wScr))
                        y3 = np.interp(y1, (frameR, h - frameR), (0, hScr))
                        clocX = plocX + (x3 - plocX) / Config.smoothing
                        clocY = plocY + (y3 - plocY) / Config.smoothing
                        pyautogui.moveTo(clocX, clocY)
                        plocX, plocY = clocX, clocY
                        dist = math.hypot(x1 - thumb_x, y1 - thumb_y)
                        if dist < Config.click_threshold:
                            cv2.circle(frame, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                            if not is_dragging:
                                pyautogui.mouseDown()
                                is_dragging = True
                                self.play_sound(800, 50)
                                self.log_signal.emit("左键按下 / 拖拽")
                        else:
                            if is_dragging:
                                pyautogui.mouseUp()
                                is_dragging = False
                                self.log_signal.emit("左键松开")

                    elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0:
                        dist_right = math.hypot(x2 - thumb_x, y2 - thumb_y)
                        cv2.line(frame, (thumb_x, thumb_y), (x2, y2), (0, 255, 255), 2)
                        if dist_right < Config.click_threshold:
                            cv2.circle(frame, (x2, y2), 15, (0, 0, 255), cv2.FILLED)
                            if time.time() - last_click_time > 0.6:
                                pyautogui.rightClick()
                                self.play_sound(1500, 100)
                                self.log_signal.emit("触发：右键点击")
                                last_click_time = time.time()

                    elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1:
                        cx = int(lm[9].x * w)
                        if time.time() - last_page_time > PAGE_COOLDOWN:
                            if cx < 100:
                                pyautogui.press('left')
                                self.play_sound(600, 200)
                                self.log_signal.emit("触发：上一页")
                                last_page_time = time.time()
                            elif cx > (w - 100):
                                pyautogui.press('right')
                                self.play_sound(600, 200)
                                self.log_signal.emit("触发：下一页")
                                last_page_time = time.time()

            if not results.multi_hand_landmarks and is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
                self.log_signal.emit("手势丢失 - 自动释放")

            self.change_pixmap_signal.emit(frame)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


# ==========================================
# 3. 页面一：欢迎页 (WelcomePage)
# ==========================================
class WelcomePage(QWidget):
    def __init__(self, start_callback):
        super().__init__()
        # 设置对象名以便在 QSS 中定位
        self.setObjectName("WelcomePage")

        layout = QVBoxLayout()
        layout.addStretch()

        # 标题区域
        title_label = QLabel("AI GESTURE CONTROL")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setObjectName("TitleLabel")  # 用于设置字体样式

        subtitle_label = QLabel("未来的交互体验 · 触手可及")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setObjectName("SubtitleLabel")

        # 开始按钮
        self.btn_start = QPushButton("开启控制台")
        self.btn_start.setFixedSize(220, 60)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.setObjectName("StartButton")
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 212, 255, 150))
        shadow.setOffset(0, 0)
        self.btn_start.setGraphicsEffect(shadow)

        self.btn_start.clicked.connect(start_callback)

        # 布局组装
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addSpacing(50)
        layout.addWidget(self.btn_start, 0, Qt.AlignCenter)
        layout.addStretch()

        self.setLayout(layout)


# ==========================================
# 4. 页面二：控制台页 (ConsolePage)
# ==========================================
class ConsolePage(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.thread = None
        self.back_callback = back_callback

        # 顶部栏
        top_bar = QHBoxLayout()
        btn_back = QPushButton("← 返回")
        btn_back.setObjectName("BackButton")
        btn_back.setFixedSize(80, 30)
        btn_back.clicked.connect(self.stop_and_back)

        title = QLabel("控制中心")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #fff;")

        top_bar.addWidget(btn_back)
        top_bar.addWidget(title)
        top_bar.addStretch()

        # 主内容
        content_layout = QHBoxLayout()

        # 左侧：视频
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #00d4ff; background-color: #000; border-radius: 5px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("摄像头正在启动...")

        v_box = QVBoxLayout()
        v_box.addWidget(self.video_label)
        v_box.addStretch()

        # 右侧：参数
        right_panel = QVBoxLayout()

        # 参数组
        group_motion = self.create_group("运动参数", [
            ("平滑系数", 1, 20, Config.smoothing, self.update_smooth),
            ("映射边距", 0, 200, Config.frame_margin, self.update_margin)
        ])

        group_interact = self.create_group("交互设置", [
            ("点击灵敏度", 20, 80, Config.click_threshold, self.update_click)
        ])

        # 开关
        chk_sound = QCheckBox("启用音效")
        chk_sound.setChecked(True)
        chk_sound.toggled.connect(lambda v: setattr(Config, 'sound_enabled', v))

        chk_skeleton = QCheckBox("显示骨架")
        chk_skeleton.setChecked(True)
        chk_skeleton.toggled.connect(lambda v: setattr(Config, 'show_landmarks', v))

        # 日志
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setObjectName("LogBox")

        right_panel.addWidget(group_motion)
        right_panel.addWidget(group_interact)
        right_panel.addWidget(chk_sound)
        right_panel.addWidget(chk_skeleton)
        right_panel.addWidget(QLabel("系统日志:"))
        right_panel.addWidget(self.log_box)

        content_layout.addLayout(v_box, stretch=2)
        content_layout.addLayout(right_panel, stretch=1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_bar)
        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)

    def create_group(self, title, sliders):
        group = QGroupBox(title)
        layout = QGridLayout()
        for idx, (label, min_v, max_v, init_v, callback) in enumerate(sliders):
            lbl = QLabel(f"{label}: {init_v}")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(init_v)
            # 使用闭包保持对 label 的引用
            slider.valueChanged.connect(lambda v, l=lbl, t=label, c=callback: (l.setText(f"{t}: {v}"), c(v)))
            layout.addWidget(lbl, idx * 2, 0)
            layout.addWidget(slider, idx * 2 + 1, 0)
        group.setLayout(layout)
        return group

    def start_camera(self):
        if self.thread is None or not self.thread.isRunning():
            self.thread = VideoThread()
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.log_signal.connect(self.update_log)
            self.thread.start()

    def stop_and_back(self):
        if self.thread:
            self.thread.stop()
        self.back_callback()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img).scaled(640, 480, Qt.KeepAspectRatio))

    @pyqtSlot(str)
    def update_log(self, msg):
        self.log_box.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def update_smooth(self, val):
        Config.smoothing = val

    def update_margin(self, val):
        Config.frame_margin = val

    def update_click(self, val):
        Config.click_threshold = val


# ==========================================
# 5. 主窗口 (Main Window - 管理页面切换)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Gesture Master 2025")
        self.resize(1100, 700)

        # 堆叠窗口用于页面切换
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 初始化两个页面
        self.welcome_page = WelcomePage(self.go_to_console)
        self.console_page = ConsolePage(self.go_to_welcome)

        self.stack.addWidget(self.welcome_page)
        self.stack.addWidget(self.console_page)

        # 加载样式表
        self.load_styles()

    def load_styles(self):
        # 1. 获取 bg.jpg 的绝对路径
        # __file__ 是当前脚本文件的路径，dirname 获取它所在的文件夹
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(base_dir, 'bg.jpg')

        # [关键步骤]：Windows路径中的反斜杠 '\' 在 CSS 中无效，必须替换为正斜杠 '/'
        img_path = img_path.replace('\\', '/')

        # 2. 检查图片是否存在 (用于调试)
        if not os.path.exists(img_path):
            print(f"【错误】找不到图片，请确认路径: {img_path}")

        # 3. 设置样式
        # 使用 border-image 可以让图片自动拉伸填满窗口
        # 即使图片尺寸很奇怪，也能铺满
        bg_style = f"""
            #WelcomePage {{
                border-image: url("{img_path}") 0 0 0 0 stretch stretch;
            }}
        """

        common_style = """
            QMainWindow { background-color: #2b2b2b; }

            /* 标题样式 */
            #TitleLabel { font-family: 'Arial Black'; font-size: 48px; color: #fff; margin-bottom: 10px; }
            #SubtitleLabel { font-family: 'Arial'; font-size: 20px; color: #00d4ff; letter-spacing: 2px; }

            /* 按钮样式 */
            #StartButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d4ff, stop:1 #005bea);
                color: white; border-radius: 30px; font-size: 20px; font-weight: bold; border: none;
            }
            #StartButton:hover { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00c3eb, stop:1 #004ec4); margin-top: 2px; }
            #StartButton:pressed { background-color: #003399; margin-top: 5px; }

            #BackButton {
                background-color: transparent; color: #bbb; border: 1px solid #555; border-radius: 5px;
            }
            #BackButton:hover { color: #fff; border-color: #fff; }

            /* 通用控件 */
            QGroupBox { border: 1px solid #444; border-radius: 8px; margin-top: 10px; color: #00d4ff; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLabel { color: #ddd; }
            QSlider::groove:horizontal { height: 6px; background: #444; border-radius: 3px; }
            QSlider::handle:horizontal { background: #00d4ff; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }
            QTextEdit { background-color: #1a1a1a; color: #0f0; border: 1px solid #444; font-family: Consolas; }
            QCheckBox { color: #fff; padding: 5px; }
        """
        self.setStyleSheet(bg_style + common_style)

    def go_to_console(self):
        self.stack.setCurrentIndex(1)  # 切换到控制台页面
        self.console_page.start_camera()  # 启动摄像头

    def go_to_welcome(self):
        self.stack.setCurrentIndex(0)  # 切换回欢迎页

    def closeEvent(self, event):
        self.console_page.stop_and_back()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())