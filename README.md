# YOLO11 Forest Fire Detection

基于 Ultralytics YOLO11 的森林火灾协同检测项目，面向论文 A New Collaborative Detection Method for Forest Fires Under Degraded Image Conditions 的复现与实验整理。

## 项目简介

本项目围绕论文提出的 CoDeF-Net 框架展开，将 Retinex-BCE 图像增强与 YOLOv11 检测结合，用于提升雾霾、低照度和退化图像条件下的森林火灾识别能力。

项目重点解决的问题包括：

- 退化图像中烟雾特征模糊、难以分辨
- 火焰亮度衰减导致的小目标漏检
- 复杂森林背景下误检和稳定性不足

## 核心方法

- Retinex-BCE 增强：通过亮度分解与亮通道增强，尽量恢复火焰与烟雾的可见性。
- 协同检测：增强后的图像再输入 YOLOv11，提高复杂环境中的检测鲁棒性。
- 结果分析：仓库中的曲线脚本用于训练过程、精度指标和增强效果对比。

## 论文信息

- 论文题目：A New Collaborative Detection Method for Forest Fires Under Degraded Image Conditions
- 期刊：Remote Sensing
- 卷期：2026, 18(12)
- 文章号：1880
- DOI：10.3390/rs18121880

## 快速使用

```bash
python train.py
python val.py
python detect.py
```

如果你使用自己的权重，请在 [detect.py](detect.py) 中把 best.pt 路径改成你的模型文件。

## 仓库内容

- [train.py](train.py)：训练入口
- [val.py](val.py)：验证入口
- [detect.py](detect.py)：推理入口
- [plot_result.py](plot_result.py)：训练曲线与结果绘图
- [plot_result_detect.py](plot_result_detect.py)：检测结果绘图
- [plot_result_enh.py](plot_result_enh.py)：增强对比绘图
- [项目说明.md](项目说明.md)：更完整的中文项目说明

## GitHub 上传

如果本地代码已经提交过，后续更新只需要：

```bash
git add .
git commit -m "update"
git push
```

## 说明

本仓库是在 Ultralytics YOLO11 基础上整理的研究项目，但首页内容已经替换为你的论文项目说明，不再展示官方文档内容。
