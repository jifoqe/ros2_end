#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt

# CSV 路徑
csv_file = "/home/u/ros2_ws/src/ros2_end/esp32_receiver/data/dqn_parameters.csv"

# 儲存 (episode, reward) 配對，方便同步排序
data_points = []

# 開啟 CSV，讓 DictReader 自動讀標題列
with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            step = int(row["episode"])
            # status = row["reason"] # 不再需要，可以註釋掉
            reward = float(row["score"])
        except ValueError:
            # 如果 episode 或 score 不能轉成數字就跳過
            continue
        
        # 將 (episode, reward) 組合成一個元組存入列表
        data_points.append((step, reward))

## 關鍵的排序步驟
# 根據元組的第一個元素 (step/episode) 進行升序排序
# key=lambda x: x[0] 表示以 data_points 列表中每個元組的第0個元素 (episode) 作為排序依據
sorted_data = sorted(data_points, key=lambda x: x[0])

# 將排序後的資料重新拆分為 steps 和 scores 兩個列表
steps = [point[0] for point in sorted_data]
scores = [point[1] for point in sorted_data]


# 畫圖
plt.figure(figsize=(16,6))
# plt.scatter(steps, scores, c=colors, s=50, label="Data points") # 您的原代碼中沒有定義 colors
plt.plot(steps, scores)  # 連線方便觀察趨勢
plt.xlabel("episode")
plt.ylabel("reward")
plt.title("Robot Task Results (Sorted by Episode)")
plt.grid(True) # 加上網格線，視覺化效果更好
plt.show()
