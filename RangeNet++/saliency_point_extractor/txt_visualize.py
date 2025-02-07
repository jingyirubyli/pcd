import numpy as np
import matplotlib.pyplot as plt

# 加载点云
point_cloud = np.loadtxt("/Users/jingyili/Desktop/saliency_point_extractor/raw_txt/1.txt")
x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]

# 绘制点云
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='b', s=1)  # s 控制点的大小
plt.title("Point Cloud Visualization")
plt.show()