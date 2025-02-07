import cv2
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import math

def fill_convex_hull_area(binary_image):
    """
    填充凸包内的区域为白色。

    Args:
        binary_image (numpy.ndarray): 输入的二值图像。

    Returns:
        numpy.ndarray: 填充凸包区域后的图像。
    """
    # 找到所有非零点（点集）
    points = np.column_stack(np.where(binary_image > 0))

    # 如果点数太少，跳过凸包计算
    if points.shape[0] < 3:
        print("Not enough points to calculate a convex hull.")
        return binary_image

    # 计算凸包
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # 使用 OpenCV 填充凸包区域
    filled_image = np.zeros_like(binary_image)  # 初始化空白图像
    cv2.fillPoly(filled_image, [hull_points.astype(np.int32)], 255)

    return filled_image

# def extract_hull

def draw_convex_hull_contour(binary_image):
    """
    提取二值图像的凸包轮廓并将其显示在图像上。

    Args:
        binary_image (numpy.ndarray): 输入的二值图像（像素值应为0或255）。

    Returns:
        contour_image (numpy.ndarray): 带有凸包轮廓的图像。
        hull_points (numpy.ndarray): 凸包的点集。
    """
    # 检查输入图像是否为二值图
    if binary_image.max() > 1:
        _, binary_image = cv2.threshold(binary_image, 127, 1, cv2.THRESH_BINARY)

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("Error: No contours found in the image.")
        return None, None

    # 获取最大的轮廓（假设目标区域是最大的轮廓）
    largest_contour = max(contours, key=cv2.contourArea)

    # 计算凸包
    hull = cv2.convexHull(largest_contour)

    # 绘制凸包轮廓
    contour_image = np.zeros_like(binary_image, dtype=np.uint8)
    cv2.drawContours(contour_image, [hull], -1, 255, thickness=2)

    # 返回凸包的点集
    hull_points = hull[:, 0, :]  # 提取凸包点集，形状为 (N, 2)

    return contour_image, hull_points

def add_black_border(filled_image, border_thickness=5):
    """
    在已填充凸包的图像周围添加黑色边框。
    
    参数：
    - filled_image: np.ndarray，凸包内部填充为白色的二值图像。
    - border_thickness: int，黑色边框的宽度，默认值为 5 像素。
    
    返回：
    - bordered_image: np.ndarray，带有黑色边框的新图像。
    """
    # 获取图像的高度和宽度
    height, width = filled_image.shape

    # 创建一个新图像，比原图大，黑色填充
    bordered_image = np.zeros(
        (height + 2 * border_thickness, width + 2 * border_thickness), dtype=np.uint8
    )

    # 将原图嵌入到新的带边框图像中
    bordered_image[
        border_thickness : border_thickness + height,
        border_thickness : border_thickness + width,
    ] = filled_image

    return bordered_image

def fit_circle(hull_points):
    """拟合圆"""
    center, radius = cv2.minEnclosingCircle(hull_points)
    return center, radius

def fit_rectangle(hull_points):
    """拟合矩形"""
    rect = cv2.minAreaRect(hull_points)  # 最小面积矩形
    box = cv2.boxPoints(rect)
    box = np.int0(box)  # 转换为整数点
    return box

def fit_triangle(hull_points):
    """拟合三角形"""
    hull_points = np.array(hull_points).reshape(-1, 2)
    _, triangle = cv2.minEnclosingTriangle(hull_points)  # 最小包围三角形
    triangle = np.int0(triangle.reshape(-1, 2))
    return triangle

def plot_fitted_shapes(binary_image, hull_points):
    """
    使用凸包点集分别拟合圆、矩形和三角形。
    返回拟合形状的数据，供外部可视化使用。
    """
    # 圆拟合
    circle_data = fit_circle(hull_points)

    # 矩形拟合
    rectangle_data = fit_rectangle(hull_points)

    # 三角形拟合
    triangle_data = fit_triangle(hull_points)
    # print(circle_data, rectangle_data, triangle_data)
    return circle_data, rectangle_data, triangle_data

def calculate_area_of_circle(radius):
    """计算圆的面积"""  
    return math.pi * radius ** 2

def calculate_area_of_rectangle(box_points):
    """计算矩形的面积"""
    width = np.linalg.norm(box_points[0] - box_points[1])
    height = np.linalg.norm(box_points[1] - box_points[2])
    return width * height

def calculate_area_of_triangle(triangle_points):
    """计算三角形的面积"""
    # 使用海伦公式计算三角形面积
    a = np.linalg.norm(triangle_points[0] - triangle_points[1])
    b = np.linalg.norm(triangle_points[1] - triangle_points[2])
    c = np.linalg.norm(triangle_points[2] - triangle_points[0])
    s = (a + b + c) / 2  # 半周长
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))  # 海伦公式
    return area


def calculate_extent_score(hull_points, hull_area):
    """计算圆、矩形和三角形的 Extent Score"""
    
    # 1. 计算圆拟合的面积
    circle_center, circle_radius = cv2.minEnclosingCircle(hull_points)
    circle_area = calculate_area_of_circle(circle_radius)
    extent_circle = 100 * hull_area / circle_area 

    # 2. 计算矩形拟合的面积
    rectangle_box = cv2.minAreaRect(hull_points)
    box_points = cv2.boxPoints(rectangle_box)
    box_points = np.int0(box_points)
    rectangle_area = calculate_area_of_rectangle(box_points)
    extent_rectangle = 100 * hull_area / rectangle_area

    # 3. 计算三角形拟合的面积
    _, triangle_points = cv2.minEnclosingTriangle(hull_points)
    triangle_points = np.int0(triangle_points)
    triangle_area = calculate_area_of_triangle(triangle_points)
    extent_triangle = 100 * hull_area / triangle_area
    # extent_triangle = triangle_area /hull_area
    # print(hull_area, circle_area, rectangle_area, triangle_area)
    # print(extent_circle, extent_rectangle, extent_triangle)
    return extent_circle, extent_rectangle, extent_triangle
    
def main():
    # 1. 加载二值图像
    input_image_path = "binary_image.png"  # 替换为你的图像路径
    binary_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    if binary_image is None:
        print("Error: Could not load the image. Check the file path.")
        return
    
    filled_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    if filled_image is None:
        print("Error: Could not load the image. Check the file path.")
        return
    border_thickness = 100  # 边框宽度（可调整）
    bordered_image = add_black_border(filled_image, border_thickness)

    # 将图像转换为二值（确保像素值为 0 或 1）
    _, binary_image = cv2.threshold(binary_image, 127, 1, cv2.THRESH_BINARY)

    # 2. 计算凸包并填充区域
    filled_hull_image = fill_convex_hull_area(bordered_image)
    contour_image, hull_points = draw_convex_hull_contour(binary_image)
    # 3. 显示结果
    # cv2.imshow("Original Binary Image", binary_image * 255)
    # cv2.imshow("Filled Convex Hull Area", filled_hull_image)
    # 提取凸包轮廓
    contours, _ = cv2.findContours(filled_hull_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("Error: No contours found in the image.")
        return

    # 获取最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    hull_points = hull[:, 0, :]  # 提取凸包点集

    # # 可视化拟合结果
    # plot_fitted_shapes(binary_image, hull_points)
# 执行形状拟合
    circle_data, rectangle_data, triangle_data = plot_fitted_shapes(filled_hull_image, hull_points)

    # 计算凸包面积
    hull_area = ConvexHull(np.column_stack(np.where(filled_hull_image > 0))).area
    print(f"Convex Hull Area: {hull_area}")

    # 计算 Extent Score
    extent_circle, extent_rectangle, extent_triangle = calculate_extent_score(hull_points, hull_area)
    scores = extent_circle, extent_rectangle, extent_triangle
    shape_keys = ["Circle", "Rectangle", "Triangle"]
    # 输出结果
    print(f"Extent Score for Circle: {extent_circle}")
    print(f"Extent Score for Rectangle: {extent_rectangle}")
    print(f"Extent Score for Triangle: {extent_triangle}")
    # print(type(scores), scores)
    scores_dict = dict(zip(shape_keys, scores))
    # print(scores_dict)
    # print(f"凸包点数: {len(hull_points)}")

    best_shape = max(scores_dict, key=scores_dict.get)
    print(f"Best shape: {best_shape}")
    # 创建彩色图像用于可视化
    shape_image = cv2.cvtColor(filled_hull_image, cv2.COLOR_GRAY2BGR)

    # 可视化圆形
    center, radius = circle_data
    if not np.isnan(center).any() and not np.isnan(radius):
        cv2.circle(shape_image, (int(center[0]), int(center[1])), int(radius), (255, 0, 0), 2)

    # 可视化矩形
    box = rectangle_data
    cv2.drawContours(shape_image, [box], 0, (0, 255, 0), 2)

    # 可视化三角形
    triangle = triangle_data
    cv2.drawContours(shape_image, [triangle], 0, (0, 0, 255), 2)


    plt.figure(figsize=(5, 5))
    plt.subplot(2, 2, 1)
    plt.title("Original Binary Image")
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Filled Convex Hull Area")
    plt.imshow(filled_hull_image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(bordered_image, cmap='gray')
    plt.title("Full Convex Hull")
    # plt.title("Convex Hull Contour")
    plt.axis('off')

    # shape_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(shape_image, cv2.COLOR_BGR2RGB))
    plt.title("Fitted Contours")
    plt.axis('off')

    plt.show()
    # 4. 保存结果
    output_image_path = "fitted_contour.png"
    cv2.imwrite(output_image_path, shape_image)
    print(f"Filled convex hull image saved to: {output_image_path}")

if __name__ == "__main__":
    main()