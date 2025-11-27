import sys
import signal
import os
from datetime import datetime
import cv2

from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap

class CaptureThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, cam_index=0, parent=None):
        super().__init__(parent)
        self._running = False
        self._cam_index = cam_index

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self._cam_index)
        if not cap.isOpened():
            return
        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                self.changePixmap.emit(qimg)

        finally:
            cap.release()

    def stop(self):
        self._running = False
        self.wait()  # shuts the camera down smoothly

class VideoContainer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 Video")
        self.resize(1200, 800)
        self.padding = 10

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setGeometry(self.padding, self.padding,
                               self.width() - 2*self.padding,
                               self.height() - 2*self.padding)

        self.th = CaptureThread(0, self)
        self.th.changePixmap.connect(self.setImage)
        self.th.start()


    @pyqtSlot(QImage)
    def setImage(self, image):
        imw = self.width() - 2*self.padding
        imh = self.height() - 2*self.padding
        scaled = image.scaled(
            imw, imh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.label.resize(scaled.width(), scaled.height())
        self.label.move(
            int((self.width()  - scaled.width())  / 2),
            int((self.height() - scaled.height()) / 2)
        )
        self.label.setPixmap(QPixmap.fromImage(scaled))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.label.setGeometry(self.padding, self.padding,
                               self.width() - 2*self.padding,
                               self.height() - 2*self.padding)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    ex = VideoContainer()
    ex.show()
    app.aboutToQuit.connect(ex.th.stop)
    sys.exit(app.exec())