import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QComboBox, QProgressBar, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from food_classifier import predict_food, train_model, load_model, extract_features
import librosa
import librosa.display
import subprocess

class AudioProcessThread(QThread):
    """音频处理线程"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, audio_path, mode='predict'):
        super().__init__()
        self.audio_path = audio_path
        self.mode = mode
        
    def run(self):
        try:
            if self.mode == 'predict':
                # 加载模型
                model, scaler = load_model()
                if model is None:
                    self.error.emit("模型加载失败")
                    return
                
                # 预测
                results = predict_food(self.audio_path, model, scaler)
                if results:
                    self.finished.emit({"results": results})
                else:
                    self.error.emit("预测失败")
            
            elif self.mode == 'train':
                # 训练模型
                model, scaler = train_model()
                if model is not None:
                    self.finished.emit({"message": "模型训练完成"})
                else:
                    self.error.emit("模型训练失败")
                    
        except Exception as e:
            self.error.emit(f"处理出错: {str(e)}")

class SpectrogramWidget(QWidget):
    """频谱图显示组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # 创建matplotlib图形
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # 初始化图形
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("频谱图")
        
    def clear_plot(self):
        """清除当前频谱图和颜色条"""
        self.figure.clear()  # 清除整个图形（包括颜色条）
        self.ax = self.figure.add_subplot(111)  # 重新创建坐标轴
        self.ax.set_title("频谱图")
        self.canvas.draw()
        
    def plot_spectrogram(self, audio_path):
        """显示频谱图"""
        try:
            # 清除旧图和颜色条
            self.clear_plot()
            
            # 加载音频
            y, sr = librosa.load(audio_path)
            
            # 计算频谱图
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            
            # 显示频谱图
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=self.ax)
            self.figure.colorbar(img, ax=self.ax, format='%+2.0f dB')
            
            # 调整布局以确保所有元素都能正确显示
            self.figure.tight_layout()
            
            # 更新画布
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"频谱图生成失败: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("食物咀嚼声分析系统")
        self.setMinimumSize(1000, 800)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        layout = QVBoxLayout(main_widget)
        
        # 添加控制按钮
        control_layout = QHBoxLayout()
        self.load_button = QPushButton("加载音频")
        self.train_button = QPushButton("训练模型")
        self.predict_button = QPushButton("开始预测")
        self.open_folder_button = QPushButton("训练用模型文件夹")
        self.predict_button.setEnabled(False)
        
        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.train_button)
        control_layout.addWidget(self.predict_button)
        control_layout.addWidget(self.open_folder_button)
        layout.addLayout(control_layout)
        
        # 添加频谱图显示区域
        self.spectrogram = SpectrogramWidget()
        layout.addWidget(self.spectrogram)
        
        # 添加结果显示区域
        self.result_label = QLabel("预测结果将在这里显示")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("QLabel { background-color : #f0f0f0; padding: 10px; }")
        layout.addWidget(self.result_label)
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # 连接信号和槽
        self.load_button.clicked.connect(self.load_audio)
        self.train_button.clicked.connect(self.train_model)
        self.predict_button.clicked.connect(self.predict_audio)
        self.open_folder_button.clicked.connect(self.open_training_folder)
        
        # 初始化变量
        self.current_audio_path = None
        self.processing_thread = None
        
    def load_audio(self):
        """加载音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            "",
            "Audio Files (*.wav *.mp3 *.m4a)"
        )
        if file_path:
            # 清除之前的音频路径
            if self.current_audio_path:
                self.spectrogram.clear_plot()
            
            self.current_audio_path = file_path
            self.predict_button.setEnabled(True)
            self.update_status(f"已加载: {os.path.basename(file_path)}")
            
            # 显示频谱图
            self.spectrogram.plot_spectrogram(file_path)
            
    def train_model(self):
        """训练模型"""
        reply = QMessageBox.question(
            self,
            "确认",
            "训练模型可能需要几分钟时间，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.update_status("开始训练模型...")
            self.train_button.setEnabled(False)
            self.processing_thread = AudioProcessThread(None, mode='train')
            self.processing_thread.finished.connect(self.handle_training_complete)
            self.processing_thread.error.connect(self.handle_error)
            self.processing_thread.start()
            
    def predict_audio(self):
        """预测音频类型"""
        if not self.current_audio_path:
            self.update_status("请先加载音频文件")
            return
            
        self.predict_button.setEnabled(False)
        self.processing_thread = AudioProcessThread(self.current_audio_path, mode='predict')
        self.processing_thread.finished.connect(self.handle_prediction_results)
        self.processing_thread.error.connect(self.handle_error)
        self.processing_thread.start()
        
    def handle_prediction_results(self, results):
        """处理预测结果"""
        if "results" in results:
            result_text = "预测结果:\n"
            total_segments = sum(count for _, count in results["results"])
            for food_type, count in results["results"]:
                percentage = (count / total_segments) * 100
                result_text += f"- {food_type}: {count}个片段 ({percentage:.1f}%)\n"
            self.result_label.setText(result_text)
        
        self.predict_button.setEnabled(True)
        
    def handle_training_complete(self, results):
        """处理训练完成"""
        self.train_button.setEnabled(True)
        self.update_status(results.get("message", "模型训练完成"))
        QMessageBox.information(self, "完成", "模型训练完成！")
        
    def handle_error(self, error_message):
        """处理错误"""
        self.train_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.update_status(f"错误: {error_message}")
        QMessageBox.warning(self, "错误", error_message)
        
    def update_status(self, message):
        """更新状态信息"""
        self.result_label.setText(message)

    def open_training_folder(self):
        """打开训练数据文件夹"""
        audio_folder = os.path.join(os.getcwd(), 'audio')
        
        # 如果文件夹不存在，创建它
        if not os.path.exists(audio_folder):
            os.makedirs(audio_folder)
            self.update_status("已创建训练数据文件夹")
        
        # 在Windows上使用explorer打开文件夹
        try:
            subprocess.run(['explorer', audio_folder], check=True)
        except subprocess.CalledProcessError:
            self.update_status("无法打开文件夹")
            QMessageBox.warning(self, "错误", "无法打开训练数据文件夹")

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 