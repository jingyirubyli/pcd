import open3d as o3d
import numpy as np

# 1. 读取点云数据 (从 .txt 文件中加载)
point_cloud = np.loadtxt('/Users/jingyili/Desktop/saliency_point_extractor/raw_txt/1.txt')

# 2. 将 NumPy 数组转换为 Open3D 点云格式
def create_open3d_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

# 3. 可视化原始和过滤后的点云
def visualize_point_cloud(pcd1, pcd2=None):
    # 给原始点云设置颜色（灰色）
    pcd1.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色表示原始点云

    # 如果有过滤后的点云，设置不同的颜色（红色）
    if pcd2 is not None:
        pcd2.paint_uniform_color([1, 0, 0])  # 红色表示过滤后的点云

    # 在同一个窗口显示原始和过滤后的点云
    o3d.visualization.draw_geometries([pcd1, pcd2] if pcd2 is not None else [pcd1])

# 4. 可视化原始点云
pcd_original = create_open3d_point_cloud(point_cloud)

# 5. 平面拟合、计算距离并过滤
def fit_plane(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    covariance_matrix = np.cov(centered_points.T)
    _, _, vh = np.linalg.svd(covariance_matrix)
    normal = vh[-1, :]
    D = -np.dot(normal, centroid)
    return normal, D

def point_to_plane_distance(points, normal, D):
    distances = np.abs(np.dot(points, normal) + D) / np.linalg.norm(normal)
    return distances

def filter_outliers(points, normal, D, threshold=0.05):
    distances = point_to_plane_distance(points, normal, D)
    filtered_points = points[distances < threshold]
    return filtered_points

# 6. 进行平面拟合并过滤离群点
normal, D = fit_plane(point_cloud)
filtered_point_cloud = filter_outliers(point_cloud, normal, D, threshold=0.5)

# 7. 创建过滤后的点云对象
pcd_filtered = create_open3d_point_cloud(filtered_point_cloud)

# 8. 可视化原始和过滤后的点云
visualize_point_cloud(pcd_original, pcd_filtered)