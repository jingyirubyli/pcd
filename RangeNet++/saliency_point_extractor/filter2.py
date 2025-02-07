import open3d as o3d

# 函数：加载点云
def load_point_cloud(file_path):
    # 使用Open3D读取PLY文件
    point_cloud = o3d.io.read_point_cloud(file_path)
    return point_cloud

# 函数：统计离群点移除（Second Filtering）
def second_filtering(point_cloud, nb_neighbors=20, std_ratio=20.0):
    """
    执行统计离群点移除（Second Filtering）

    :param point_cloud: 需要处理的点云
    :param nb_neighbors: 邻域的点数
    :param std_ratio: 标准差倍数，决定离群点的阈值
    :return: 过滤后的点云
    """
    # 移除离群点
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    # 使用过滤后的点云
    filtered_point_cloud = point_cloud.select_by_index(ind)
    
    return filtered_point_cloud

# 可视化点云
def visualize_point_cloud(point_cloud):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 获取渲染选项
    render_option = vis.get_render_option()

    # 设置背景颜色为黑色
    render_option.background_color = [0, 0, 0]  # RGB为[0, 0, 0]表示黑色
    render_option.point_size = 2
    # 可视化点云
    vis.add_geometry(point_cloud)
    
    # 启动可视化
    vis.run()
    vis.destroy_window()

# 读取PLY点云文件路径
point_cloud_file = '/Users/jingyili/Desktop/saliency_point_extractor/filtered_point_cloud/1_filtered.ply'

# 加载点云数据
point_cloud = load_point_cloud(point_cloud_file)

# 第二步：统计离群点移除
filtered_point_cloud = second_filtering(point_cloud)

# 可视化过滤前后的点云
visualize_point_cloud(point_cloud)
visualize_point_cloud(filtered_point_cloud)

# 保存过滤后的点云
filtered_point_cloud_file = './filtered_point_cloud.ply'
o3d.io.write_point_cloud(filtered_point_cloud_file, filtered_point_cloud)