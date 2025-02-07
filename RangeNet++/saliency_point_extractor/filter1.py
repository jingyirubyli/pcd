import open3d as o3d
import numpy as np
import os
import glob

# 1. 读取点云数据 (从 .txt 文件中加载)

input_dir = "/Users/jingyili/Desktop/saliency_point_extractor/raw_txt"  # 你的输入文件夹路径
output_dir = "filtered_point_cloud"  # 过滤后的点云保存文件夹

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取所有 .txt 文件
txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
# point_cloud = np.loadtxt('./1.txt')

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
# pcd_original = create_open3d_point_cloud(point_cloud)

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

# # 6. 进行平面拟合并过滤离群点
# normal, D = fit_plane(point_cloud)
# filtered_point_cloud = filter_outliers(point_cloud, normal, D, threshold=0.5)

# # 7. 创建过滤后的点云对象
# pcd_filtered = create_open3d_point_cloud(filtered_point_cloud)

# output_dir = "./filtered_point_cloud"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# input_file = './1.txt' 
# base_filename = os.path.splitext(os.path.basename(input_file))[0]
# output_file = os.path.join(output_dir, f"{base_filename}_filtered.ply")
# o3d.io.write_point_cloud(output_file, pcd_filtered)

# 8. 可视化原始和过滤后的点云
# visualize_point_cloud(pcd_original, pcd_filtered)

# 4. 对每个 .txt 文件进行处理
for input_file in txt_files:
    # 读取点云数据
    point_cloud = np.loadtxt(input_file)

    # 进行平面拟合并过滤离群点
    normal, D = fit_plane(point_cloud)
    filtered_point_cloud = filter_outliers(point_cloud, normal, D, threshold=0.05)

    # 创建过滤后的点云对象
    pcd_filtered = create_open3d_point_cloud(filtered_point_cloud)

    # 获取输入文件的基本文件名（不含路径和扩展名）
    base_filename = os.path.splitext(os.path.basename(input_file))[0]

    # 构建保存文件的完整路径
    output_file = os.path.join(output_dir, f"{base_filename}_filtered.ply")

    # 保存过滤后的点云
    o3d.io.write_point_cloud(output_file, pcd_filtered)

    print(f"Filtered point cloud saved to {output_file}")

# o3d.visualization.draw_geometries([point_cloud, pcd_filtered],
#                                     window_name="Original and Filtered Point Clouds",
#                                     width=800, height=600)
