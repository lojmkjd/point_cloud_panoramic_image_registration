import sys
import os

# 获取项目根目录路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到 Python 路径，确保可以正确导入 utils 模块
sys.path.append(project_root)

import numpy as np
import open3d as o3d
import pandas as pd
from utils.coordinate_transformation import CoordinateTransformer

def test_pcd_transform():
    # 读取点云数据
    pcd_path = r"D:\DataVolume\TestDataForPanoramicLidar-master\processed_data\pts.pcd"
    pcd = o3d.io.read_point_cloud(pcd_path)  # 使用 open3d 读取点云文件
    points = np.asarray(pcd.points)  # 将点云数据转换为 numpy 数组

    # 读取相机位姿信息
    pose_path = r"D:\DataVolume\TestDataForPanoramicLidar-master\processed_data\camera_pos.csv"
    pose_df = pd.read_csv(pose_path)  # 使用 pandas 读取 CSV 文件
    
    # 从 CSV 文件中提取相机位姿信息
    x, y, z = pose_df['X'].values[0], pose_df['Y'].values[0], pose_df['Altitude'].values[0]  # 平移向量
    roll, pitch, yaw = pose_df['Roll'].values[0], pose_df['Pitch'].values[0], pose_df['Heading'].values[0]  # 旋转角度

    # 创建坐标变换器
    transformer = CoordinateTransformer()
    transformer.set_translation(x, y, z)  # 设置平移向量
    transformer.set_rotation(roll, pitch, yaw)  # 设置旋转矩阵

    # 进行坐标变换
    transformed_points = transformer.transform(points)  # 将点云从原坐标系转换到目标坐标系

    # 创建变换后的点云
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)  # 将变换后的点云数据赋值给新点云对象

    # 设置点云颜色
    pcd.paint_uniform_color([1, 0, 0])  # 原始点云为红色
    transformed_pcd.paint_uniform_color([0, 1, 0])  # 变换后的点云为绿色

    # 创建两个独立的可视化窗口
    vis1 = o3d.visualization.Visualizer()  # 创建第一个可视化器
    vis1.create_window(window_name="原始点云", width=800, height=600)  # 创建窗口，设置窗口名称和大小
    vis1.add_geometry(pcd)  # 将原始点云添加到可视化器中

    vis2 = o3d.visualization.Visualizer()  # 创建第二个可视化器
    vis2.create_window(window_name="变换后点云", width=800, height=600)  # 创建窗口，设置窗口名称和大小
    vis2.add_geometry(transformed_pcd)  # 将变换后的点云添加到可视化器中

    # 保持窗口打开
    while True:
        vis1.update_geometry(pcd)  # 更新原始点云的几何数据
        vis2.update_geometry(transformed_pcd)  # 更新变换后点云的几何数据
        
        if not vis1.poll_events() or not vis2.poll_events():  # 检查窗口是否关闭
            break
        
        vis1.update_renderer()  # 更新第一个窗口的渲染
        vis2.update_renderer()  # 更新第二个窗口的渲染

    # 关闭窗口
    vis1.destroy_window()  # 销毁第一个窗口
    vis2.destroy_window()  # 销毁第二个窗口

if __name__ == "__main__":
    test_pcd_transform()  # 运行测试函数
