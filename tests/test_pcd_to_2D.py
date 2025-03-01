import sys
import os

# 获取项目根目录路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到 Python 路径，确保可以正确导入 utils 模块
sys.path.append(project_root)

import numpy as np
import open3d as o3d
import pandas as pd
import cv2
from utils.coordinate_transformation import CoordinateTransformer
from utils.dimensional_transformation import DimensionalTransformation

def test_pcd_to_2D():
    # 读取点云数据
    pcd_path = r"D:\DataVolume\TestDataForPanoramicLidar-master\processed_data\pts.pcd"
    pcd = o3d.io.read_point_cloud(pcd_path)  # 使用 open3d 读取点云文件
    points = np.asarray(pcd.points)  # 将点云数据转换为 numpy 数组

    # 读取相机位姿信息
    pose_path = r"D:\DataVolume\TestDataForPanoramicLidar-master\processed_data\camera_pos.csv"
    pose_df = pd.read_csv(pose_path)  # 使用 pandas 读取 CSV 文件
    
    # 从 CSV 文件中提取相机位姿信息
    x, y, z = pose_df['X'].values[8], pose_df['Y'].values[8], pose_df['Altitude'].values[8]  # 平移向量
    roll, pitch, yaw = pose_df['Roll'].values[8], pose_df['Pitch'].values[8], pose_df['Heading'].values[8]  # 旋转角度
    Width,Height =pose_df['Width'].values[8],pose_df['Height'].values[8]

    # 创建坐标变换器
    transformer = CoordinateTransformer()
    transformer.set_translation(x, y, z)  # 设置平移向量
    transformer.set_rotation(roll, pitch, yaw)  # 设置旋转矩阵

    # 进行坐标变换
    transformed_points = transformer.transform(points)  # 将点云从原坐标系转换到目标坐标系

    # 创建变换后的点云
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)  # 将变换后的点云数据赋值给新点云对象

    # 实现等距投影
    dimensional_transformer = DimensionalTransformation()
    pixel_points = dimensional_transformer.from_3D_to_2D(transformed_points,Width,Height)
    
    # 创建一个空的图像
    image = np.zeros((Height, Width), dtype=np.uint8)

    # 将像素点绘制到图像上
    image[np.int_(pixel_points[:, 1]), np.int_(pixel_points[:, 0])] = 255

    # 显示图像
    cv2.imshow("等距投影后的图像",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_pcd_to_2D()  # 运行测试函数
