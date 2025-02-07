import numpy as np

def filter_points_by_label(bin_file, label_file, target_label, output_txt):
    # 加载点云和标签
    point_cloud = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)  # 包含 (x, y, z, intensity)
    labels = np.fromfile(label_file, dtype=np.uint32) & 0xFFFF  # 低 16 位是类别标签

    # 过滤出目标类别的点
    filtered_points = point_cloud[labels == target_label, :3]  # 保留 (x, y, z)

    # 保存过滤后的点云为 .txt
    np.savetxt(output_txt, filtered_points, fmt="%.6f")
    print(f"Filtered points saved to {output_txt}")

# 示例：提取类别 80（车辆）的点云
bin_file_path = "/Users/jingyili/Desktop/saliency_point_extractor/velodyne_bin/000000.bin"
label_file_path = "/Users/jingyili/Desktop/saliency_point_extractor/labels/000000.label"
output_txt_path = "./filtered_points.txt"
filter_points_by_label(bin_file_path, label_file_path, target_label=80, output_txt=output_txt_path)