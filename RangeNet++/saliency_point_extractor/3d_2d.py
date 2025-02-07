import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from plane import fit_plane
import cv2

# 定义将点云投影到平面的函数
def project_to_plane(point_cloud, plane_params):
    """
    将3D点云投影到给定平面上
    :param point_cloud: 3D点云 (N, 3)
    :param plane_params: 平面方程参数 (a, b, c, d)
    :return: 投影后的点云 (N, 3)
    """
    a, b, c, d = plane_params
    normal = np.array([a, b, c])
    normal_norm = np.linalg.norm(normal)  # 法向量的模

    # 投影每个点到平面
    projected_points = []
    for point in point_cloud:
        distance = (np.dot(normal, point) + d) / (normal_norm ** 2)
        projected_point = point - distance * normal
        projected_points.append(projected_point)

    return np.array(projected_points)

# 点云映射到图像网格
def convert_to_image(points_2d, alpha=1000, beta=100, img_size=(1000, 1000)):
    """
    将2D点云转换为二值图像
    :param points_2d: 2D点云
    :param alpha: 分辨率
    :param beta: 偏移量
    :param img_size: 图像尺寸
    :return: 二值图像
    """
    # 创建空白图像
    img = np.zeros(img_size, dtype=np.uint8)

    # 将2D点映射到图像网格
    for point in points_2d:
        u = int(np.round(alpha * point[0] + beta))
        v = int(np.round(alpha * point[1] + beta))

        # 确保坐标在图像尺寸内
        if 0 <= u < img_size[0] and 0 <= v < img_size[1]:
            img[u, v] = 1  # 将对应的像素设为1

    return img

# 可视化函数
def visualize_point_cloud_and_image(original_pc, projected_pc, img):
    """
    同时可视化原始点云、投影点云和二值图像
    """
    # 原始点云
    o3d_pc_orig = o3d.geometry.PointCloud()
    o3d_pc_orig.points = o3d.utility.Vector3dVector(original_pc)

    # 投影后的点云
    o3d_pc_proj = o3d.geometry.PointCloud()
    o3d_pc_proj.points = o3d.utility.Vector3dVector(projected_pc)

    # 创建颜色
    o3d_pc_orig.paint_uniform_color([1, 0, 0])  # 红色
    o3d_pc_proj.paint_uniform_color([0, 1, 0])  # 绿色

    # 可视化点云
    o3d.visualization.draw_geometries([o3d_pc_orig, o3d_pc_proj])

    # 可视化二值图像
    plt.imshow(img, cmap='gray')
    plt.title("Binary Image")
    plt.show()

def numpy_to_open3d(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    return pcd

def save_binary_image(binary_image, output_path):
    """
    保存二值图像到指定路径。

    Args:
        binary_image (numpy.ndarray): 二值图像数据。
        output_path (str): 保存文件的路径，例如 "binary_image.png"。
    """
    # 确保二值图像格式为 uint8 (0 和 255)
    if binary_image.max() <= 1:  # 如果二值图是 0 和 1
        binary_image = (binary_image * 255).astype(np.uint8)
    
    # 保存图像
    cv2.imwrite(output_path, binary_image)
    print(f"Binary image saved at: {output_path}")

# 示例
def example_usage():
    image_size = 1000
    beta = image_size // 10 
    # 读取点云文件路径
    # point_cloud_file = '/Users/jingyili/Desktop/saliency_point_extractor/raw_txt/1.txt'
    point_cloud = np.loadtxt('/Users/jingyili/Desktop/saliency_point_extractor/raw_txt/1.txt')
    # 加载点云数据
    # 转换为 Open3D 点云对象
    pcd = numpy_to_open3d(point_cloud)
    # point_cloud = load_point_cloud(point_cloud_file)
    plane_model, _ = fit_plane(pcd) 
    # 将点云投影到平面
    projected_pc = project_to_plane(point_cloud, plane_model)

    # 丢弃z坐标得到2D点云
    points_2d = projected_pc[:, :2]

    # # 转换为 Open3D 点云对象
    # pcd = numpy_to_open3d(point_cloud)
    # # point_cloud = load_point_cloud(point_cloud_file)
    # plane_model, _ = fit_plane(pcd) 
    # 示例平面方程 ax + by + cz + d = 0 (z = 5 平面)
    # 计算 alpha
    x_min, x_max = -71.00135028226528, 73.05716120640254
    y_min, y_max = -21.087425362414915, 53.815838585692056

    alpha_x = (image_size - 2 * beta) / (x_max - x_min)
    alpha_y = (image_size - 2 * beta) / (y_max - y_min)
    alpha = min(alpha_x, alpha_y)  # 取较小值，保证点云完全显示

    # 将点云映射到图像网格
    # u = np.round((points_2d[:, 0] - x_min) * alpha + beta).astype(int)
    # v = np.round((points_2d[:, 1] - y_min) * alpha + beta).astype(int)    
    u = np.round((points_2d[:, 0] - x_min) / alpha + beta).astype(int)
    v = np.round((points_2d[:, 1] - y_min) / alpha + beta).astype(int)

    # 设置图像大小
    image = np.zeros((image_size, image_size), dtype=np.uint8)

    # 过滤点，确保索引在图像范围内
    valid_mask = (u >= 0) & (u < image_size) & (v >= 0) & (v < image_size)
    u = u[valid_mask]
    v = v[valid_mask]

    # 映射到图像
    image[v, u] = 1


    # # 将点云投影到平面
    # projected_pc = project_to_plane(point_cloud, plane_model)

    # # 丢弃z坐标得到2D点云
    # points_2d = projected_pc[:, :2]
    print("Non-zero pixels in binary image:", np.count_nonzero(image))
    print("First few (u, v) points:", list(zip(u[:10], v[:10])))

    print("Mapped U range:", np.min(u), "to", np.max(u))
    print("Mapped V range:", np.min(v), "to", np.max(v))
    plt.scatter(u, v, s=1)  # 使用小点绘制
    plt.gca().invert_yaxis()  # 符合图像坐标
    plt.title("Scatter plot of projected points")
    plt.show()

    # 将2D点云转换为二值图像
    img = convert_to_image(points_2d)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.title("Binary Image with Correct Intensity Range")
    plt.show()
    # 检查投影到2D后的点的范围
    # print("Projected 2D points range:")
    # print("X range:", np.min(points_2d[:, 0]), "to", np.max(points_2d[:, 0]))
    # print("Y range:", np.min(points_2d[:, 1]), "to", np.max(points_2d[:, 1]))
    # 计算 U 和 V 的范围
    u_min, u_max = np.min(u), np.max(u)
    v_min, v_max = np.min(v), np.max(v)

    # 设置图像的分辨率
    image_width = 500  # 图像宽度，例如 300 像素
    image_height = 500  # 图像高度，例如 300 像素

    # 平移和缩放点云到新的坐标范围
    u_mapped = ((u - u_min) / (u_max - u_min) * (image_width - 1)).astype(int)
    v_mapped = ((v - v_min) / (v_max - v_min) * (image_height - 1)).astype(int)

    # 初始化二值图像
    binary_image = np.zeros((image_height, image_width), dtype=np.uint8)

    # 点大小 (radius)
    point_size = 2  # 半径为 2，表示 5x5 的区域

    # 将点映射到图像，并增加点的大小
    for i in range(len(u_mapped)):
        if 0 <= u_mapped[i] < image_width and 0 <= v_mapped[i] < image_height:
            # 在点的邻域范围内绘制
            for dx in range(-point_size, point_size + 1):
                for dy in range(-point_size, point_size + 1):
                    nx, ny = u_mapped[i] + dx, v_mapped[i] + dy
                    if 0 <= nx < image_width and 0 <= ny < image_height:
                        binary_image[ny, nx] = 1


    # # 将点映射到二值图像
    # valid_indices = (u_mapped >= 0) & (u_mapped < image_width) & (v_mapped >= 0) & (v_mapped < image_height)
    # binary_image[v_mapped[valid_indices], u_mapped[valid_indices]] = 1

    # 显示二值图像
    plt.imshow(binary_image, cmap='gray', vmin=0, vmax=1)
    plt.title("Binary Image Adjusted to Point Range")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
    save_binary_image(binary_image, "binary_image.png")
    # 可视化原始点云、投影点云和二值图像
    # visualize_point_cloud_and_image(point_cloud, projected_pc, img)


# 执行示例
example_usage()

# # 示例调用
# if __name__ == "__main__":
#     # 假设 binary_image 是之前生成的二值图
#     binary_image = np.zeros((500, 500), dtype=np.uint8)
#     binary_image[100:400, 100:400] = 1  # 测试数据

#     # 保存二值图像
#     save_binary_image(binary_image, "binary_image.png")