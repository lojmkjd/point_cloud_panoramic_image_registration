import numpy as np

class CoordinateTransformer:
    def __init__(self):
        # 初始化平移向量、旋转矩阵、齐次矩阵及其逆矩阵
        self.translation_vector = np.zeros(3)  # 平移向量，默认值为 [0, 0, 0]
        self.rotation_matrix = np.eye(3)  # 旋转矩阵，默认值为单位矩阵
        self.homogeneous_matrix = np.eye(4)  # 齐次矩阵，默认值为单位矩阵
        self.inv_homogeneous_matrix = np.eye(4)  # 齐次矩阵的逆矩阵，用于缓存
    
    def set_translation(self, x, y, z):
        """设置平移向量"""
        self.translation_vector = np.array([x, y, z])  # 将输入的 x, y, z 转换为 numpy 数组
        self.update_matrices()  # 更新齐次矩阵及其逆矩阵
    
    def set_rotation(self, roll, pitch, yaw):
        """设置旋转矩阵（欧拉角）"""
        # 将欧拉角从度转换为弧度
        roll, pitch, yaw = np.radians([roll, pitch, yaw])
        
        # 计算绕 X 轴的旋转矩阵
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        Rx = np.array([[1, 0, 0],
                      [0, cos_r, -sin_r],
                      [0, sin_r, cos_r]])
        
        # 计算绕 Y 轴的旋转矩阵
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        Ry = np.array([[cos_p, 0, sin_p],
                      [0, 1, 0],
                      [-sin_p, 0, cos_p]])
        
        # 计算绕 Z 轴的旋转矩阵
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cos_y, -sin_y, 0],
                      [sin_y, cos_y, 0],
                      [0, 0, 1]])
        
        # 组合旋转矩阵：Rz * Ry * Rx
        self.rotation_matrix = Rz @ Ry @ Rx
        self.update_matrices()  # 更新齐次矩阵及其逆矩阵
    
    def update_matrices(self):
        """更新齐次矩阵及其逆矩阵"""
        self.homogeneous_matrix = np.eye(4)  # 初始化齐次矩阵为单位矩阵
        self.homogeneous_matrix[:3, :3] = self.rotation_matrix  # 设置旋转部分
        self.homogeneous_matrix[:3, 3] = self.translation_vector  # 设置平移部分
        self.inv_homogeneous_matrix = np.linalg.inv(self.homogeneous_matrix)  # 计算逆矩阵并缓存
    
    def transform(self, points):
        """从原坐标系转换到目标坐标系"""
        points = np.asarray(points)  # 将输入的点云数据转换为 numpy 数组
        if points.ndim == 1:  # 如果输入是单个点，将其转换为二维数组
            points = points[np.newaxis, :]
        
        # 将点云数据转换为齐次坐标形式（添加一列 1）
        homogeneous_points = np.column_stack([points, np.ones(points.shape[0])])
        # 使用逆矩阵进行坐标变换
        transformed_points = homogeneous_points @ self.inv_homogeneous_matrix.T
        return transformed_points[:, :3].squeeze()  # 返回变换后的点云数据，去除齐次坐标
    
    def inverse_transform(self, points):
        """从目标坐标系转换到原坐标系"""
        points = np.asarray(points)  # 将输入的点云数据转换为 numpy 数组
        if points.ndim == 1:  # 如果输入是单个点，将其转换为二维数组
            points = points[np.newaxis, :]
        
        # 将点云数据转换为齐次坐标形式（添加一列 1）
        homogeneous_points = np.column_stack([points, np.ones(points.shape[0])])
        # 使用齐次矩阵进行逆变换
        transformed_points = homogeneous_points @ self.homogeneous_matrix.T
        return transformed_points[:, :3].squeeze()  # 返回逆变换后的点云数据，去除齐次坐标