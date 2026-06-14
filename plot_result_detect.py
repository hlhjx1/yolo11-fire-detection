import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d


pwd = os.getcwd()
names = ['RT-DETR', 'yolov7', 'yolov10', 'yolov11']

# 处理yolov7结果（txt文件），去掉第0列(epoch/total_epoch)和第1列(显存)Retinex-BCE'
def deal_yolov7_result(data_path):
    data_list = []
    with open(data_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # 去掉第0列(epoch/total_epoch)和第1列(显存)
            parts = [p for i, p in enumerate(parts) if i not in (0, 1)]
            data_list.append(parts)
    return np.array(data_list)

# 文件查找
def find_result_file(model_name):
    csv_path = f'runs/train/fire/{model_name}/results.csv'
    txt_path = f'runs/train/fire/{model_name}/results.txt'
    if os.path.exists(csv_path):
        return csv_path, 'csv'
    elif os.path.exists(txt_path):
        return txt_path, 'txt'
    else:
        raise FileNotFoundError(f"results file not found for {model_name}")

# YOLOv7列索引（去掉两列后调整）
yolov7_index_map = {
    'train/box_loss': 0,
    'train/obj_loss': 1,
    'train/cls_loss': 2,
    'val/box_loss': 3,
    'val/obj_loss': 4,
    'val/cls_loss': 5,
    'precision': 6,
    'recall': 7,
    'mAP50': 8,
    'mAP50-95': 9,
}

# YOLOv11列名
yolov11_col_map = {
    'precision': 'metrics/precision(B)',
    'recall': 'metrics/recall(B)',
    'mAP50': 'metrics/mAP50(B)',
    'mAP50-95': 'metrics/mAP50-95(B)',
    'train/box_loss': 'train/box_loss',
    'train/obj_loss': 'train/dfl_loss',  # YOLOv11用dfl_loss代替obj_loss
    'train/cls_loss': 'train/cls_loss',
    'val/box_loss': 'val/box_loss',
    'val/obj_loss': 'val/dfl_loss',
    'val/cls_loss': 'val/cls_loss'
}

# 开始绘图
plt.figure(figsize=(20, 5))
metrics = ['precision', 'recall', 'mAP50', 'mAP50-95']

for idx, metric in enumerate(metrics, 1):
    plt.subplot(1, 4, idx)
    for model_name in names:
        try:
            file_path, file_type = find_result_file(model_name)
        except FileNotFoundError as e:
            print(e)
            continue

        if file_type == 'csv':
            data = pd.read_csv(file_path)
            col = yolov11_col_map[metric]
            if col not in data.columns:
                print(f"Warning: {col} not found in {file_path}")
                continue
            y = data[col].astype(float).replace(np.inf, np.nan).interpolate()
            # y = y.head(80)
            if model_name == 'ours':
                y = data[col].astype(float).replace(np.inf, np.nan).interpolate().iloc[:80]
                f = interp1d(np.arange(80), y, kind='linear', fill_value="extrapolate")
                x_new = np.linspace(0, 79, 200)
                y = f(x_new)
        else:
            data = deal_yolov7_result(file_path)
            y = np.array(data[:, yolov7_index_map[metric]], dtype=float)
            # y = np.array(data[:80, yolov7_index_map[metric]], dtype=float)



        plt.plot(y, label=model_name)
    plt.xlabel('epoch')
    plt.title(metric)
    plt.legend()

plt.tight_layout()
plt.savefig('metrice_curve_detect.png')
print(f'metrice_curve.png saved in {pwd}/metrice_curve_detect.png')


plt.figure(figsize=(15, 10))
loss_metrics = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'val/box_loss', 'val/obj_loss', 'val/cls_loss']

for idx, metric in enumerate(loss_metrics, 1):
    plt.subplot(2, 3, idx)
    for model_name in names:
        try:
            file_path, file_type = find_result_file(model_name)
        except FileNotFoundError as e:
            print(e)
            continue

        if file_type == 'csv':
            data = pd.read_csv(file_path)
            col = yolov11_col_map[metric]
            if col not in data.columns:
                print(f"Warning: {col} not found in {file_path}")
                continue
            y = data[col].astype(float).replace(np.inf, np.nan).interpolate()
        else:
            data = deal_yolov7_result(file_path)
            y = np.array(data[:, yolov7_index_map[metric]], dtype=float)

        plt.plot(y, label=model_name)
    plt.xlabel('epoch')
    plt.title(metric)
    plt.legend()

plt.tight_layout()
plt.savefig('loss_curve_detect.png')
print(f'loss_curve.png saved in {pwd}/loss_curve_detect.png')
