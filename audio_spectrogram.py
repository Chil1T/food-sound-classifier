import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import traceback
from pydub import AudioSegment
import tempfile

# 忽略警告信息
warnings.filterwarnings('ignore')

def convert_to_wav(input_path):
    """
    将音频文件转换为WAV格式
    
    参数:
        input_path: 输入文件路径
    
    返回:
        str: 临时WAV文件的路径
    """
    try:
        # 获取文件扩展名
        ext = os.path.splitext(input_path)[1].lower()
        
        # 如果已经是wav格式，直接返回原路径
        if ext == '.wav':
            return input_path
            
        print(f"转换音频格式: {ext} -> .wav")
        
        # 创建临时文件
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        
        # 加载音频文件
        if ext == '.m4a':
            audio = AudioSegment.from_file(input_path, format='m4a')
        elif ext == '.mp3':
            audio = AudioSegment.from_mp3(input_path)
        else:
            audio = AudioSegment.from_file(input_path)
            
        # 导出为WAV格式
        audio.export(temp_wav.name, format='wav')
        print(f"格式转换完成: {temp_wav.name}")
        
        return temp_wav.name
        
    except Exception as e:
        print(f"转换音频格式时出错: {str(e)}")
        print("详细错误信息:")
        print(traceback.format_exc())
        return None

def load_audio(file_path):
    """
    加载音频文件并返回信号数据和采样率
    """
    try:
        print(f"尝试加载音频文件: {file_path}")
        print(f"文件是否存在: {os.path.exists(file_path)}")
        print(f"文件大小: {os.path.getsize(file_path)} bytes")
        
        # 首先转换为WAV格式
        wav_path = convert_to_wav(file_path)
        if wav_path is None:
            return None, None
            
        # 加载音频文件
        y, sr = librosa.load(wav_path, duration=60.0)  # 限制加载时长为60秒
        print(f"成功加载音频文件，采样率: {sr}Hz")
        
        # 如果是临时文件，则删除
        if wav_path != file_path:
            try:
                os.unlink(wav_path)
            except:
                pass
                
        return y, sr
    except Exception as e:
        print(f"加载音频文件时出错: {str(e)}")
        print("详细错误信息:")
        print(traceback.format_exc())
        return None, None

def plot_spectrogram(y, sr, title="Spectrogram"):
    """
    绘制音频的频谱图
    
    参数:
        y: 音频信号
        sr: 采样率
        title: 图表标题
    """
    try:
        # 创建新的图形
        plt.figure(figsize=(12, 8))
        
        # 计算梅尔频谱图
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        # 显示频谱图
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        
        return plt
    except Exception as e:
        print(f"绘制频谱图时出错: {str(e)}")
        print("详细错误信息:")
        print(traceback.format_exc())
        return None

def process_audio_file(audio_path, output_dir):
    """
    处理单个音频文件并保存频谱图
    
    参数:
        audio_path: 音频文件路径
        output_dir: 输出目录路径
    
    返回:
        bool: 处理是否成功
    """
    try:
        # 获取不带扩展名的文件名
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_sg.png")
        
        print(f"\n处理音频文件: {audio_path}")
        
        # 加载音频文件
        y, sr = load_audio(audio_path)
        
        if y is None or sr is None:
            return False
        
        # 绘制频谱图
        plt = plot_spectrogram(y, sr, title=f"Spectrogram - {base_name}")
        
        if plt is None:
            return False
            
        # 保存图像
        plt.savefig(output_path)
        plt.close()
        
        print(f"频谱图已保存为: {output_path}")
        return True
        
    except Exception as e:
        print(f"处理文件 {audio_path} 时出错: {str(e)}")
        print("详细错误信息:")
        print(traceback.format_exc())
        return False

def main():
    try:
        # 检查audio目录是否存在
        if not os.path.exists('audio'):
            os.makedirs('audio')
            print("已创建 'audio' 目录，请将音频文件放入该目录。")
            return
        
        # 创建输出目录
        output_dir = 'spectrograms'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")
        
        # 获取所有音频文件
        audio_files = [f for f in os.listdir('audio') if f.endswith(('.mp3', '.wav', '.m4a'))]
        
        if not audio_files:
            print("在 'audio' 目录中未找到音频文件。")
            print("请添加一些音频文件 (mp3, wav, 或 m4a) 到 'audio' 目录。")
            return
        
        print(f"找到 {len(audio_files)} 个音频文件")
        
        # 处理每个音频文件
        success_count = 0
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n处理第 {i}/{len(audio_files)} 个文件:")
            audio_path = os.path.join('audio', audio_file)
            if process_audio_file(audio_path, output_dir):
                success_count += 1
        
        # 输出处理结果统计
        print(f"\n处理完成！")
        print(f"成功处理: {success_count}/{len(audio_files)} 个文件")
        if success_count < len(audio_files):
            print(f"失败: {len(audio_files) - success_count} 个文件")
            
    except Exception as e:
        print(f"程序执行过程中出错: {str(e)}")
        print("详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 