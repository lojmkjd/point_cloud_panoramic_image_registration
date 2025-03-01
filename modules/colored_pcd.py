import os
import numpy as np
import open3d as o3d
import pandas as pd
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.coordinate_transformation import CoordinateTransformer
from utils.dimensional_transformation import DimensionalTransformation

def get_pcd_colors(data_dir):
    # 获取点云文件路径
    pcd_file = os.path.join(os.path.join(data_dir, "pts.pcd"))

    # 读取点云数据
    pcd = o3d.io.read_point_cloud(pcd_file)
    # 素体滤波
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    points = np.asarray(pcd.points)
    print(points.shape)
    
    # 获取相机位姿文件路径
    pose_file = os.path.join(os.path.join(data_dir, "camera_pos.csv"))
    
    # 读取相机位姿信息
    pose_df = pd.read_csv(pose_file)
    
    colors_list = []
    
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
        
        # 过滤超出图像范围的点
        valid_indices = (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] < Width_pano) & (pixel_points[:, 1] >= 0) & (pixel_points[:, 1] < Height_pano)
        pixel_points = pixel_points[valid_indices].astype(int)
        
        # 从图像中获取点云对应的颜色
        colors = pano_image[pixel_points[:, 1], pixel_points[:, 0]]
        colors_list.append(colors)
        
    # 计算颜色的平均值
    color_list = np.mean(colors_list, axis=0)

    # 颜色归一化
    color_list = color_list / 255.0
    return points,color_list

if __name__ == "__main__":
    data_dir = r"D:\DataVolume\TestDataForPanoramicLidar-master\processed_data"
    points,colors = get_pcd_colors(data_dir)
    print(colors.shape)

    # 将颜色赋值给点云
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.points = o3d.utility.Vector3dVector(points)

    # 创建显示窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

