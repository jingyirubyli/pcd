import numpy as np

def bin_to_txt(bin_file, txt_file):
    # 读取 .bin 文件 (每个点 4 个 float: x, y, z, intensity)
    point_cloud = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)

    # 保存为 .txt 文件，只保留 x, y, z 坐标 (如果需要保留 intensity，可以调整 fmt)
    np.savetxt(txt_file, point_cloud[:, :3], fmt="%.6f")
    print(f"Converted {bin_file} to {txt_file}")

# 示例：转换单帧点云
bin_file_path = "/Users/jingyili/Desktop/saliency_point_extractor/velodyne_bin/000001.bin"  # 修改为你的文件路径
txt_file_path = "/Users/jingyili/Desktop/saliency_point_extractor/1.txt"
bin_to_txt(bin_file_path, txt_file_path)