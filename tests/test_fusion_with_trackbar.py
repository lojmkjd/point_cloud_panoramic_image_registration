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

def test_fusion():
    # 读取点云数据
    pcd_path = r"D:\DataVolume\TestDataForPanoramicLidar-master\processed_data\pts.pcd"
    pcd = o3d.io.read_point_cloud(pcd_path)
    # 滤波
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    points = np.asarray(pcd.points)

    # 读取相机位姿信息
    pose_path = r"D:\DataVolume\TestDataForPanoramicLidar-master\processed_data\camera_pos.csv"
    pose_df = pd.read_csv(pose_path)
    
    # 从 CSV 文件中提取相机位姿信息
    # 行数
    row_num = 2
    x, y, z = pose_df['X'].values[row_num], pose_df['Y'].values[row_num], pose_df['Altitude'].values[row_num]
    roll, pitch, yaw = 1+pose_df['Roll'].values[row_num], -4+pose_df['Pitch'].values[row_num], -122+pose_df['Heading'].values[row_num]
    Width, Height = pose_df['Width'].values[row_num], pose_df['Height'].values[row_num]

    # 创建坐标变换器并进行坐标变换
    transformer = CoordinateTransformer()
    transformer.set_translation(x, y, z)
    transformer.set_rotation(roll, pitch, yaw)
    transformed_points = transformer.transform(points)

    # 实现等距投影
    dimensional_transformer = DimensionalTransformation()
    pixel_points = dimensional_transformer.from_3D_to_2D(transformed_points, Width, Height)
    
    # 创建一个空的图像
    lidar_image = np.zeros((Height, Width, 3), dtype=np.uint8)

    # 将像素点绘制到图像上（绿色）
    lidar_image[np.int_(pixel_points[:, 1]), np.int_(pixel_points[:, 0])] = [0, 255, 0]

    # 读取全景图像，并获取实际尺寸
    pano_image_path = r"D:\DataVolume\TestDataForPanoramicLidar-master\processed_data\images\LB3-20140924-033636-000000000902.jpg"
    pano_image = cv2.imread(pano_image_path)
    if pano_image is None:
        raise ValueError("无法读取全景图像，请检查文件路径是否正确")
    # 获取全景图的实际宽高
    Width_pano = pano_image.shape[1]
    Height_pano = pano_image.shape[0]

    # 创建空的lidar_image，直接使用全景图尺寸
    lidar_image = np.zeros((Height_pano, Width_pano, 3), dtype=np.uint8)

    # 创建滑块窗口
    cv2.namedWindow("融合后的图像", cv2.WINDOW_NORMAL)

    # 修改投影部分的代码，使用全景图尺寸
    def update_orientation():
        nonlocal yaw, roll, pitch
        transformer.set_rotation(roll, pitch, yaw)
        transformed_points = transformer.transform(points)
        # 使用全景图尺寸进行投影
        pixel_points = dimensional_transformer.from_3D_to_2D(transformed_points, Width_pano, Height_pano)
        lidar_image.fill(0)
        valid_indices = (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] < Width_pano) & (pixel_points[:, 1] >= 0) & (pixel_points[:, 1] < Height_pano)
        pixel_points = pixel_points[valid_indices]
        lidar_image[np.int_(pixel_points[:, 1]), np.int_(pixel_points[:, 0])] = [0, 255, 0]
        fused_image = cv2.addWeighted(pano_image, 0.7, lidar_image, 0.3, 0)
        cv2.imshow("融合后的图像", fused_image)
        
    def update_yaw(value):
        nonlocal yaw
        yaw = value-180
        update_orientation()
        
    def update_roll(value):
        nonlocal roll
        roll = value-180
        update_orientation()
        
    def update_pitch(value):
        nonlocal pitch
        pitch = value-180
        update_orientation()

    # 创建滑块
    cv2.createTrackbar("Yaw", "融合后的图像", int(yaw)+180, 360, update_yaw)
    cv2.createTrackbar("Roll", "融合后的图像", int(roll)+180, 360, update_roll)
    cv2.createTrackbar("Pitch", "融合后的图像", int(pitch)+180, 360, update_pitch)

    # 初始显示
    update_orientation()

    # 等待用户操作
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 按下 ESC 键退出
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_fusion()
