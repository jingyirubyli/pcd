from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def label_saliency_points(points, primitives, semantic_labels):
    """
    对显著点进行标注。

    参数:
    - points: 显著点的二维坐标列表 [(x1, y1), (x2, y2), ...]
    - primitives: 每个显著点拟合的几何形状 ["circle", "rectangle", "triangle", ...]
    - semantic_labels: 每个显著点的语义标签 ["traffic_sign", "vehicle", ...]

    返回:
    - labeled_points: 带有标签的显著点 [(x, y, s, p), ...]
    """
    labeled_points = []
    for i, point in enumerate(points):
        primitive_label = primitives[i]
        semantic_label = semantic_labels[i]
        if is_center_point(point):  # 判断是否是中心点
            primitive_label = "center"
        labeled_points.append((point[0], point[1], semantic_label, primitive_label))
    return labeled_points

def is_center_point(point):
    """
    判断给定点是否为中心点（可以基于点的特性实现）。
    """
    # 示例：假设中心点满足某些条件
    return True  # 替换为实际逻辑

def back_project_to_3D(points_2D, projection_matrix):
    """
    将2D点集投影到3D。

    参数:
    - points_2D: 二维点集 [(x1, y1), (x2, y2), ...]
    - projection_matrix: 投影矩阵 (3x4)

    返回:
    - points_3D: 三维点集 [(x, y, z), ...]
    """
    points_3D = []
    for point in points_2D:
        x, y = point
        # 将 (x, y, 1) 转为齐次坐标
        homogeneous_2D = np.array([x, y, 1]).reshape(3, 1)
        # 通过投影矩阵得到3D点
        homogeneous_3D = np.dot(projection_matrix, homogeneous_2D)
        # 转为非齐次坐标
        x_3D = homogeneous_3D[0] / homogeneous_3D[3]
        y_3D = homogeneous_3D[1] / homogeneous_3D[3]
        z_3D = homogeneous_3D[2] / homogeneous_3D[3]
        points_3D.append((x_3D, y_3D, z_3D))
    return points_3D

def generate_3D_labeled_points(points_2D, primitives, semantic_labels, projection_matrix):
    # 1. 标注2D显著点
    labeled_points_2D = label_saliency_points(points_2D, primitives, semantic_labels)
    
    # 2. 投影到3D空间
    points_3D = back_project_to_3D(points_2D, projection_matrix)
    
    # 3. 合并3D点和标签
    labeled_points_3D = []
    for i, (x, y, z) in enumerate(points_3D):
        semantic_label = labeled_points_2D[i][2]
        primitive_label = labeled_points_2D[i][3]
        labeled_points_3D.append((x, y, z, semantic_label, primitive_label))
    return labeled_points_3D


def visualize_3D_points(labeled_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for point in labeled_points:
        x, y, z, s, p = point
        ax.scatter(x, y, z, label=f"{s}-{p}")
    plt.legend()
    plt.show()

# 可视化
# visualize_3D_points(labeled_points)
