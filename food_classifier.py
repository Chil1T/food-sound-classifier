import os
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import warnings
import traceback
from pydub import AudioSegment
import tempfile
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

# 忽略警告信息
warnings.filterwarnings('ignore')

def convert_to_wav(input_path):
    """
    将音频文件转换为WAV格式
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
        
        return temp_wav.name
        
    except Exception as e:
        print(f"转换音频格式时出错: {str(e)}")
        print(traceback.format_exc())
        return None

def normalize_audio(y):
    """
    对音频信号进行响度归一化
    
    参数:
        y: 音频信号
    
    返回:
        归一化后的音频信号
    """
    # 如果信号全为0，直接返回
    if np.all(y == 0):
        return y
        
    # 计算均方根值（RMS）作为响度的度量
    rms = np.sqrt(np.mean(y**2))
    
    # 如果RMS太小，可能是噪声，直接返回
    if rms < 1e-6:
        return y
        
    # 归一化到合适的响度范围
    target_rms = 0.1  # 目标RMS值
    scaling_factor = target_rms / rms
    return y * scaling_factor

def split_audio(y, sr, segment_duration=2.0):
    """
    将音频信号分割成固定长度的片段
    
    参数:
        y: 音频信号
        sr: 采样率
        segment_duration: 每个片段的持续时间（秒）
    
    返回:
        list: 音频片段列表
    """
    # 首先对整个音频进行响度归一化
    y_normalized = normalize_audio(y)
    
    # 计算每个片段的采样点数
    segment_length = int(segment_duration * sr)
    
    # 计算可以分割出多少个完整片段
    n_segments = len(y_normalized) // segment_length
    
    # 分割音频
    segments = []
    for i in range(n_segments):
        start = i * segment_length
        end = start + segment_length
        segment = y_normalized[start:end]
        
        # 计算片段的RMS值
        segment_rms = np.sqrt(np.mean(segment**2))
        
        # 使用相对阈值：与整个音频的平均RMS值比较
        full_rms = np.sqrt(np.mean(y_normalized**2))
        relative_threshold = full_rms * 0.1  # 使用平均RMS的10%作为阈值
        
        # 只保留有效的音频片段（去除静音片段）
        if segment_rms > relative_threshold:
            segments.append(segment)
            
        # 输出调试信息
        print(f"    片段 {i+1}: RMS = {segment_rms:.6f}, 阈值 = {relative_threshold:.6f}, "
              f"{'通过' if segment_rms > relative_threshold else '未通过'}")
    
    return segments

def extract_features(file_path):
    """
    从音频文件中提取特征，包含多层特征融合
    """
    try:
        # 转换为WAV格式
        wav_path = convert_to_wav(file_path)
        if wav_path is None:
            return None
            
        # 加载音频文件
        y, sr = librosa.load(wav_path)
        
        print(f"  音频信息:")
        print(f"    - 采样率: {sr} Hz")
        print(f"    - 时长: {len(y)/sr:.2f} 秒")
        print(f"    - RMS响度: {np.sqrt(np.mean(y**2)):.6f}")
        
        # 分割音频
        print(f"  分割音频片段:")
        segments = split_audio(y, sr)
        
        # 提取每个片段的特征
        features_list = []
        for i, segment in enumerate(segments, 1):
            # 1. 时频域基础特征
            # 1.1 MFCC特征及其统计量
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)  # 增加到20个系数
            mfccs_mean = np.mean(mfccs.T, axis=0)
            mfccs_std = np.std(mfccs.T, axis=0)
            mfccs_delta = librosa.feature.delta(mfccs)
            mfccs_delta_mean = np.mean(mfccs_delta.T, axis=0)
            mfccs_delta2 = librosa.feature.delta(mfccs, order=2)  # 二阶差分
            mfccs_delta2_mean = np.mean(mfccs_delta2.T, axis=0)
            
            # 1.2 梅尔频谱特征
            mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
            mel_spec_mean = np.mean(mel_spec.T, axis=0)[:20]  # 取前20个系数
            mel_spec_std = np.std(mel_spec.T, axis=0)[:20]
            
            # 2. 频域特征增强
            # 2.1 基础频谱特征
            spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)
            spectral_flatness = librosa.feature.spectral_flatness(y=segment)
            
            # 2.2 频谱统计特征
            spectral_stats = [
                np.mean(spectral_centroid),
                np.std(spectral_centroid),
                np.max(spectral_centroid),
                np.min(spectral_centroid),
                np.percentile(spectral_centroid, 25),  # 四分位数
                np.percentile(spectral_centroid, 75)
            ]
            
            # 2.3 频谱对比度增强
            contrast_stats = [
                np.mean(spectral_contrast),
                np.std(spectral_contrast),
                np.max(spectral_contrast),
                np.min(spectral_contrast),
                np.percentile(spectral_contrast, 25),
                np.percentile(spectral_contrast, 75)
            ]
            
            # 3. 时域特征增强
            # 3.1 基础时域特征
            zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)
            rms_energy = librosa.feature.rms(y=segment)
            
            # 3.2 时域统计特征
            temporal_stats = [
                np.mean(zero_crossing_rate),
                np.std(zero_crossing_rate),
                np.mean(rms_energy),
                np.std(rms_energy),
                np.max(rms_energy),
                np.min(rms_energy)
            ]
            
            # 4. 谐波特征增强
            # 4.1 色度图特征
            chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
            chroma_mean = np.mean(chroma.T, axis=0)
            chroma_std = np.std(chroma.T, axis=0)
            
            # 4.2 谐波特征
            harmonic, percussive = librosa.effects.hpss(segment)
            harmonic_mean = np.mean(harmonic)
            harmonic_std = np.std(harmonic)
            percussive_mean = np.mean(percussive)
            percussive_std = np.std(percussive)
            
            # 5. 特征融合
            combined_features = np.concatenate([
                # 时频域基础特征 (80维)
                mfccs_mean,                    # 20维
                mfccs_std,                     # 20维
                mfccs_delta_mean,              # 20维
                mfccs_delta2_mean,             # 20维
                
                # 梅尔频谱特征 (40维)
                mel_spec_mean,                 # 20维
                mel_spec_std,                  # 20维
                
                # 频域特征 (16维)
                spectral_stats,                # 6维
                contrast_stats,                # 6维
                [np.mean(spectral_rolloff)],   # 1维
                [np.mean(spectral_bandwidth)], # 1维
                [np.mean(spectral_flatness)],  # 1维
                [np.percentile(spectral_flatness, 75)], # 1维
                
                # 时域特征 (6维)
                temporal_stats,                # 6维
                
                # 谐波特征 (28维)
                chroma_mean,                   # 12维
                chroma_std,                    # 12维
                [harmonic_mean, harmonic_std,  # 4维
                 percussive_mean, percussive_std]
            ])
            
            features_list.append(combined_features)
        
        # 如果是临时文件，则删除
        if wav_path != file_path:
            try:
                os.unlink(wav_path)
            except:
                pass
                
        return features_list
        
    except Exception as e:
        print(f"特征提取出错: {str(e)}")
        print(traceback.format_exc())
        return None

def prepare_data():
    """
    准备训练数据
    """
    features = []
    labels = []
    
    # 从audio目录加载音频文件
    audio_dir = 'audio'
    for file_name in os.listdir(audio_dir):
        if file_name.endswith(('.mp3', '.wav', '.m4a')):
            file_path = os.path.join(audio_dir, file_name)
            
            # 提取特征
            print(f"正在处理: {file_name}")
            feature_list = extract_features(file_path)
            
            if feature_list is not None and len(feature_list) > 0:
                # 获取标签（从文件名中提取食物类型）
                label = file_name.split('.')[0]
                # 如果文件名中包含数字（如"白菜2"），去掉数字
                label = ''.join([i for i in label if not i.isdigit()])
                
                # 为每个音频片段添加相同的标签
                features.extend(feature_list)
                labels.extend([label] * len(feature_list))
                
                print(f"  - 提取了 {len(feature_list)} 个有效片段")
    
    return np.array(features), np.array(labels)

def train_model():
    """
    训练SVM模型，使用PCA进行特征降维
    """
    print("开始准备训练数据...")
    
    # 准备数据
    X, y = prepare_data()
    
    if len(X) == 0:
        print("没有找到有效的训练数据！")
        return
    
    print(f"\n原始数据集信息:")
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {len(X)}")
    print(f"标签类别: {np.unique(y)}")
    print(f"每个类别的样本数量:")
    for label in np.unique(y):
        count = np.sum(y == label)
        print(f"  - {label}: {count}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 使用PCA降维
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"\nPCA降维后的特征数量: {X_train_pca.shape[1]}")
    print(f"保留的方差比例: {np.sum(pca.explained_variance_ratio_):.3f}")
    print("\n各主成分的方差贡献率:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  - PC{i+1}: {ratio:.3f}")
    
    # 训练SVM模型，使用网格搜索找到最佳参数
    print("\n开始训练模型...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'poly']
    }
    
    grid_search = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_pca, y_train)
    
    print("\n最佳参数:")
    print(grid_search.best_params_)
    
    # 使用最佳参数的模型
    model = grid_search.best_estimator_
    
    # 在测试集上评估模型
    print("\n模型评估结果:")
    y_pred = model.predict(X_test_pca)
    print(classification_report(y_test, y_pred))
    
    # 返回模型、缩放器和PCA转换器
    return model, scaler, pca

def predict_food(audio_path, model, scaler, pca):
    """
    预测音频文件中的食物类型
    """
    try:
        print(f"\n分析音频文件: {audio_path}")
        
        # 提取特征
        features = extract_features(audio_path)
        
        if features is None or len(features) == 0:
            print("无法从音频中提取有效特征")
            return None
            
        # 特征缩放和PCA转换
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        # 预测
        predictions = model.predict(features_pca)
        
        # 统计每种食物类型的片段数量
        unique_predictions, counts = np.unique(predictions, return_counts=True)
        results = list(zip(unique_predictions, counts))
        
        # 按片段数量排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        print("\n预测结果:")
        total_segments = sum(count for _, count in results)
        for food_type, count in results:
            percentage = (count / total_segments) * 100
            print(f"- {food_type}: {count}个片段 ({percentage:.1f}%)")
            
        return results
        
    except Exception as e:
        print(f"预测过程出错: {str(e)}")
        print(traceback.format_exc())
        return None

def save_model(model, scaler, pca, model_path='model.joblib', 
              scaler_path='scaler.joblib', pca_path='pca.joblib'):
    """保存模型、缩放器和PCA转换器"""
    try:
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(pca, pca_path)
        print(f"\n模型已保存到: {model_path}")
        print(f"特征缩放器已保存到: {scaler_path}")
        print(f"PCA转换器已保存到: {pca_path}")
        return True
    except Exception as e:
        print(f"保存模型时出错: {str(e)}")
        return False

def load_model(model_path='model.joblib', scaler_path='scaler.joblib', 
              pca_path='pca.joblib'):
    """加载模型、缩放器和PCA转换器"""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        return model, scaler, pca
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None, None, None

def main():
    """
    主函数
    """
    try:
        # 检查命令行参数
        import sys
        if len(sys.argv) > 1:
            # 预测模式
            audio_path = sys.argv[1]
            if not os.path.exists(audio_path):
                print(f"错误：文件 {audio_path} 不存在")
                return
                
            # 加载模型
            model, scaler, pca = load_model()
            if model is None:
                print("错误：无法加载模型，请先训练模型")
                return
                
            # 预测
            predict_food(audio_path, model, scaler, pca)
        else:
            # 训练模式
            print("开始训练模型...")
            model, scaler, pca = train_model()
            if model is not None:
                save_model(model, scaler, pca)
                print("\n模型训练完成！")
                print("\n使用方法:")
                print("要预测新的音频文件，请运行:")
                print("python food_classifier.py <音频文件路径>")
                
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 