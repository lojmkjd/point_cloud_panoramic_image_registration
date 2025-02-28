import os
import numpy as np
import open3d as o3d
import pandas as pd
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.coordinate_transformation import CoordinateTransformer
from utils.dimensional_transformation import DimensionalTransformation

def batch_fusion(data_dir, result_dir):
    # 获取点云文件路径列表
    pcd_file = os.path.join(os.path.join(data_dir, "pts.pcd"))

    # 读取点云数据
    pcd = o3d.io.read_point_cloud(pcd_file)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    points = np.asarray(pcd.points)
    
    # 获取相机位姿文件路径
    pose_file = os.path.join(os.path.join(data_dir, "camera_pos.csv"))
    
    # 读取相机位姿信息
    pose_df = pd.read_csv(pose_file)
    
    # 遍历相机位姿信息
    for i in range(len(pose_df)):
        
        # 从相机位姿信息中提取对应的位姿数据
        x, y, z = pose_df['X'].values[i], pose_df['Y'].values[i], pose_df['Altitude'].values[i]
        roll, pitch, yaw = 1+pose_df['Roll'].values[i], -4+pose_df['Pitch'].values[i],-119+pose_df['Heading'].values[i] 
        
        # 创建坐标变换器并进行坐标变换
        transformer = CoordinateTransformer()
        transformer.set_translation(x, y, z)
        transformer.set_rotation(roll, pitch, yaw)
        transformed_points = transformer.transform(points)
        
        # 获取对应的全景图像文件路径
        image_name = pose_df['Filename'].values[i]
        image_file = os.path.join(data_dir, "images", image_name)
        pano_image = cv2.imread(image_file)
        Width_pano = pano_image.shape[1]
        Height_pano = pano_image.shape[0]
        
        # 使用全景图尺寸重新投影
        dimensional_transformer = DimensionalTransformation()
        pixel_points = dimensional_transformer.from_3D_to_2D(transformed_points, Width_pano, Height_pano)
        lidar_image = np.zeros((Height_pano, Width_pano, 3), dtype=np.uint8)
        
        # 过滤超出图像范围的点
        valid_indices = (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] < Width_pano) & (pixel_points[:, 1] >= 0) & (pixel_points[:, 1] < Height_pano)
        pixel_points = pixel_points[valid_indices]
        
        # 绘制点云到图像
        lidar_image[np.int_(pixel_points[:, 1]), np.int_(pixel_points[:, 0])] = [0, 255, 0]
        
        # 图像融合 
        fused_image = cv2.addWeighted(pano_image, 0.5, lidar_image, 0.5, 0)
        
        # 保存融合结果
        result_file = os.path.join(result_dir, f"fused_{i}.jpg")
        cv2.imwrite(result_file, fused_image)
        
if __name__ == "__main__":
    data_dir = r"D:\DataVolume\TestDataForPanoramicLidar-master\processed_data"
    result_dir = r"D:\DataVolume\TestDataForPanoramicLidar-master\results"
    os.makedirs(result_dir, exist_ok=True)
    batch_fusion(data_dir, result_dir)
