import numpy as np
from typing import Tuple

class DimensionalTransformation:
    """
    维度变换基类，提供坐标转换相关方法
    """
    
    @staticmethod
    def cartesian_to_polar(points: np.ndarray) -> np.ndarray:
        """
        将三维笛卡尔坐标转换为球坐标
        
        参数:
            points (np.ndarray): 形状为[N,3]的点云数据
            
        返回:
            np.ndarray: 形状为[N,3]的极坐标数据
                第一列: 径向距离
                第二列: 方位角（与x轴的夹角）
                第三列: 极角（与z轴的夹角）
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)  # 计算方位角
        if r.any() != 0:
            theta = np.arccos(z / r)
        else:
            theta = 0
        
        return np.column_stack([r, phi, theta])
    
    def from_3D_to_2D(self, points: np.ndarray, Width, Height) -> np.ndarray:
        """
        将三维点云数据转换为二维平面数据

        参数:
            points (np.ndarray): 形状为[N,3]的点云数据
            Width (int): 图像宽度
            Height (int): 图像高度

        返回:
            np.ndarray: 形状为[N,2]的二维平面数据
                第一列: 平面坐标x
                第二列: 平面坐标y
        """ 
        # 将三维点云数据转换为球坐标
        polar_points = self.cartesian_to_polar(points)
        # 根据等距投影模型,计算像素坐标
        pixel_x = (- polar_points[:, 1] + np.pi) / (2 * np.pi) * Width
        pixel_y = polar_points[:, 2] / np.pi * Height
        pixel_points = np.column_stack([pixel_x, pixel_y])
        return pixel_points
