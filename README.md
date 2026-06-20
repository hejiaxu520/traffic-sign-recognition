# 交通标志识别 (Traffic Sign Recognition)

这是一个基于 PyTorch 实现的交通标志识别项目，主要用于识别限速类交通标志，支持多种常见标志类型的分类。

## ✨ 功能特点
- 支持 **40+ 种常见交通标志** 识别（以限速标志为主）
- 带可视化 GUI 界面，可直接上传图片进行识别
- 提供模型训练、评估代码，便于二次开发与优化
- 完整依赖配置，一键安装环境

## 📦 项目结构
```
pythonProject/
├── predict.py          # 主程序：GUI 界面 + 模型推理
├── train.py            # 模型训练代码
├── data_loader.py      # 数据加载与预处理
├── test_gpu.py         # GPU 可用性测试
├── test_evaluation.py  # 模型评估脚本
├── best_traffic_model.pth  # 训练好的模型权重文件
├── requirements.txt    # 项目依赖清单
└── README.md           # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备
确保已安装 Python 3.8+，然后执行以下命令安装依赖：
```bash
pip install -r requirements.txt
```

### 2. 运行识别程序
直接运行 `predict.py` 启动 GUI 界面：
```bash
python predict.py
```
在弹出的窗口中点击「选择交通标志图片」，上传 `.jpg/.png/.bmp` 格式的图片，即可查看识别结果。

### 3. 重新训练模型（可选）
如果需要用自己的数据集训练模型，可修改 `data_loader.py` 中的数据路径，然后运行：
```bash
python train.py
```
训练完成后，将新生成的模型权重替换 `best_traffic_model.pth` 即可。

## 📊 支持的标志类型
项目支持识别以下典型交通标志（部分示例）：
- 限速标志：限速 15km/h、30km/h、60km/h 等
- 禁令标志：禁止直行、禁止左转、禁止掉头等
- 指示标志：直行、左转、右转等
- 警告标志：注意行人、注意儿童、连续弯路等

完整类别映射可在 `predict.py` 中的 `TSRD_CLASSES` 字典查看。

## 📝 注意事项
- 模型目前**主要针对限速标志优化**，部分标志识别准确率可能随数据集完善而提升
- 模型文件 `best_traffic_model.pth` 较大，已直接上传至仓库，可直接使用
- 首次运行需确保已安装所有依赖，推荐使用虚拟环境隔离项目

## 🤝 贡献
欢迎提交 Issue 或 Pull Request 来完善项目！

---
