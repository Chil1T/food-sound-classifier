# 基于音频特征的食物咀嚼声分类系统

这是一个用于复试的极简陋机器学习项目，通过分析食物咀嚼声的声学特征来识别不同类型的食物。该系统利用先进的音频处理技术和机器学习算法，实现了对食物咀嚼声的自动分类。

## 功能特点

### 1. 图形用户界面
- **实时频谱图显示**：加载音频后自动显示频谱图
- **一键式操作**：
  - 音频加载与预览
  - 模型训练
  - 类型预测
  - 训练数据管理
- **直观的结果展示**：
  - 频谱图可视化
  - 预测结果百分比显示
  - 训练进度反馈

### 2. 音频处理技术
- **自适应分段**：将长音频自动分割为2秒的短片段
- **响度归一化**：使用RMS（均方根）归一化处理
- **智能静音过滤**：基于相对能量阈值自动过滤无效片段

### 3. 特征工程
- **多维特征提取**：
  - 13维MFCC（梅尔频率倒谱系数）特征
  - MFCC的统计特征（均值、标准差）
  - 频谱质心（Spectral Centroid）
  - 过零率（Zero Crossing Rate）
  - 频谱滚降（Spectral Rolloff）
  - 频谱带宽（Spectral Bandwidth）

### 4. 机器学习模型
- 使用SVM（支持向量机）分类器
- 自动特征缩放和标准化
- 模型持久化存储与加载

## 项目结构

```
project/
│
├── audio/                    # 训练用音频文件目录（皆为本人于2月24日吃晚饭外卖拼好饭手撕鸡饭时录制）
│   ├── 白菜.m4a             # 白菜咀嚼声样本1
│   ├── 白菜2.m4a            # 白菜咀嚼声样本2
│   ├── 米饭.m4a             # 米饭咀嚼声样本
│   ├── 豆皮.m4a             # 豆皮咀嚼声样本
│   ├── 鸡皮.m4a             # 鸡皮咀嚼声样本1
│   └── 鸡皮2.m4a            # 鸡皮咀嚼声样本2
│
├── audiofortest/            # 测试用音频文件目录
│   └── baicai_fortest.m4a  # 测试用白菜咀嚼声样本
│
├── spectrograms/            # 频谱图输出目录（自动创建）
│
├── gui_main.py              # 图形界面主程序
├── audio_spectrogram.py     # 频谱图生成模块
├── food_classifier.py       # 主分类器模块
├── requirements.txt         # 项目依赖
├── model.joblib            # 预训练模型文件
└── scaler.joblib          # 预训练特征缩放器
```

## 音频数据说明

### 训练数据
- 位于 `audio` 目录
- 包含四种食物的咀嚼声：白菜、米饭、豆皮、鸡皮
- 部分食物（如白菜、鸡皮）有多个样本以提高模型鲁棒性
- 每个音频文件约15-20秒长

### 测试数据
- 位于 `audiofortest` 目录
- 提供了示例测试文件 `baicai_fortest.m4a`
- 用户可以添加自己的测试音频到此目录
- 支持的音频格式：WAV、MP3、M4A

## 安装说明

### 1. 环境要求
- Python 3.7 或更高版本
- Windows/Linux/MacOS

### 2. 安装步骤

```bash
# 1. 克隆项目
git clone [项目地址]

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装额外依赖（针对音频格式转换）
# Windows用户：
# 确保已安装ffmpeg并添加到系统PATH
```

### 3. 特殊说明
- 对于Windows用户，需要安装ffmpeg用于音频格式转换：
  1. 下载ffmpeg: https://ffmpeg.org/download.html
  2. 将ffmpeg添加到系统PATH
  3. 验证安装：在命令行运行 `ffmpeg -version`

## 使用方法

### 1. 启动图形界面
```bash
python gui_main.py
```

### 2. 使用流程
1. **准备训练数据**：
   - 点击"训练用模型文件夹"按钮
   - 将训练音频文件放入打开的audio目录
   - 注意：项目已包含预训练数据，可直接使用

2. **训练模型**：
   - 点击"训练模型"按钮
   - 等待训练完成
   - 注意：项目已包含预训练模型，可跳过此步骤直接预测

3. **预测新样本**：
   - 点击"加载音频"选择要分析的音频文件
   - 查看频谱图
   - 点击"开始预测"进行预测
   - 可以使用 audiofortest/baicai_fortest.m4a 进行测试

### 3. 命令行方式（可选）
```bash
# 训练模型（如果需要重新训练）
python food_classifier.py

# 预测新样本
python food_classifier.py audiofortest/baicai_fortest.m4a
```

## 录音建议
1. **录音环境**：
   - 保持安静的环境
   - 避免背景噪声
   - 麦克风保持适当距离（建议10-15厘米）

2. **录音参数**：
   - 每个样本建议录制15-20秒
   - 保持稳定的咀嚼节奏
   - 避免其他杂音

3. **支持格式**：
   - WAV
   - MP3
   - M4A（自动转换为WAV）

## 性能指标

当前模型在测试集上的表现：
- 整体准确率：64%
- 各类别F1分数：
  - 白菜：0.67
  - 豆皮：0.86
  - 鸡皮：0.67
  - 米饭：待优化

## 优化方案总结

### 1. 特征融合层构建
为提升模型性能，我们实现了多层特征融合策略：

#### 1.1 时频域基础特征 (80维)
- MFCC特征扩展：从13维扩展到20维
- MFCC统计特征：均值、标准差
- MFCC差分特征：一阶差分和二阶差分
- 每个特征的统计量：均值、标准差

#### 1.2 梅尔频谱特征 (40维)
- 128个梅尔滤波器组
- 取前20个系数
- 计算均值和标准差

#### 1.3 频域特征增强 (16维)
- 频谱统计特征：质心、带宽、滚降
- 频谱对比度增强
- 四分位数统计
- 频谱平坦度分析

#### 1.4 时域特征增强 (6维)
- 过零率统计
- RMS能量分析
- 时域统计特征

#### 1.5 谐波特征增强 (28维)
- 色度图特征（12维）及其统计量
- 谐波-打击乐分离（HPSS）
- 谐波和打击乐成分的统计特征

### 2. PCA降维优化
为解决特征维度过高的问题，采用PCA进行降维：

#### 2.1 降维效果
- 原始特征维度：170维
- 降维后维度：29维
- 保留方差比例：95.4%

#### 2.2 主成分分析
- 第一主成分贡献率：17.1%
- 前5个主成分累计贡献率：50%
- 特征方差分布均匀，信息丰富

#### 2.3 优化结果
- 特征维度减少83%
- 计算效率显著提升
- 模型复杂度降低
- 某些类别识别效果提升（如白菜达到100%准确率）

### 3. 优化效果分析
- 白菜：F1分数从0.67提升至1.00
- 豆皮：F1分数保持在0.86
- 鸡皮：F1分数从0.67降至0.44
- 米饭：识别仍然困难（F1=0.00）

通过分析发现，米饭的识别困难可能源于其咀嚼声特征本身不够显著，这是一个物理特性导致的限制，而非模型性能的问题。

## 未来改进方向

1. **界面优化**：
   - 添加波形显示
   - 实时音频预览
   - 批量处理功能
   - 训练进度可视化

2. **算法优化**：
   - 深度学习模型支持
   - 在线学习功能
   - 自适应特征选择

3. **数据增强**：
   - 添加噪声增强
   - 时间拉伸/压缩
   - 音高变化

## 作者
[Chil1T]

## 许可证
MIT License 
