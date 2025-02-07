import open3d as o3d
import numpy as np

# 函数：加载点云
def load_point_cloud(file_path):
    point_cloud = np.loadtxt(file_path)  # 假设点云数据存储为txt文件
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud))

# 函数：拟合平面并返回平面模型
def fit_plane(point_cloud):
    points = np.asarray(point_cloud.points)
    
    # 使用最小二乘法拟合平面，平面模型为 ax + by + cz + d = 0
    # 使用open3d的segment_plane方法来拟合平面
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.02,
                                                     ransac_n=3,
                                                     num_iterations=1000)
    a, b, c, d = plane_model
    return plane_model, inliers

# 函数：可视化点云和半透明平面
def visualize_point_cloud_and_plane(point_cloud, plane_model):
    # 获取点云
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 添加点云
    vis.add_geometry(point_cloud)
    
    # 获取平面参数 a, b, c, d
    a, b, c, d = plane_model
    # 创建一个平面
    plane_size = 100  # 平面的大小
    plane_mesh = o3d.geometry.TriangleMesh.create_box(width=plane_size, height=plane_size, depth=0.1)
    plane_mesh.translate([0, 0, -d/c])  # 平面沿 Z 轴移动，以使其与拟合平面对齐
    plane_mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz([np.arctan2(b, c), 0, np.arctan2(a, c)]))  # 旋转平面
    plane_mesh.paint_uniform_color([0.8, 0.8, 0.8])  # 设置颜色为灰色
    plane_mesh.compute_vertex_normals()

    # 渲染选项
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.point_size = 3
    render_option.line_width = 1
    render_option.show_coordinate_frame = True

    # 通过修改 render_option 来实现半透明效果
    render_option.background_color = np.asarray([0.0, 0.0, 0.0])  # 黑色背景
    render_option.mesh_show_wireframe = True  # 显示线框
    # render_option.mesh_line_width = 1  # 设置线宽
    render_option.point_size = 1  # 点云大小
    # 创建半透明平面
    # 由于 open3d 不直接支持设置透明度，我们通常选择通过 mesh_show_wireframe 和其他方法来间接实现透明效果
    plane_mesh.paint_uniform_color([0.8, 0.8, 0.8])  # 灰色平面
    vis.add_geometry(plane_mesh)
    
    # 开始可视化
    vis.run()
    vis.destroy_window()

# 读取点云文件路径
point_cloud_file = '/Users/jingyili/Desktop/saliency_point_extractor/raw_txt/1.txt'

# 加载点云数据
point_cloud = load_point_cloud(point_cloud_file)

# 拟合平面
plane_model, inliers = fit_plane(point_cloud)

# 可视化点云和拟合的半透明平面
visualize_point_cloud_and_plane(point_cloud, plane_model)