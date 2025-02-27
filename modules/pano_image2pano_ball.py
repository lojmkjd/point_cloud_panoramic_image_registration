import numpy as np

class PanoramicImageToSphere:
    def __init__(self, dpi=72):
        """
        初始化
        :param dpi: 每英寸像素数
        """
        self.dpi = dpi

    def calculate_spherical_coordinates(self, image_shape):
        """
        计算球面坐标
        :param image_shape: 图像形状 (H, W)
        :return: phi (经度), theta (极角)
        """
        h, w = image_shape
        pixel_y, pixel_x = np.indices(image_shape)
        phi = pixel_x * 2 * np.pi / w - np.pi   # 经度 [-π, π]
        theta = pixel_y * np.pi / h            # 极角 [0, π]
        return phi, theta

    def spherical_to_cartesian(self, phi, theta, radius):
        """
        将球面坐标转换为笛卡尔坐标
        :param phi: 经度
        :param theta: 极角
        :param radius: 球体半径
        :return: x, y, z 坐标
        """
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return x, y, z

    def map_coordinates_to_image(self, x, y, radius):
        """
        将坐标映射到图像空间
        :param x: x 坐标
        :param y: y 坐标
        :param radius: 球体半径
        :return: 映射后的坐标和有效索引
        """
        x_mapped = np.int_(x + radius)
        y_mapped = np.int_(y + radius)
        valid_indices = (x_mapped >= 0) & (x_mapped < 2*radius) & (y_mapped >= 0) & (y_mapped < 2*radius)
        return x_mapped, y_mapped, valid_indices

    def create_sphere_image(self, pixel_radius):
        """
        创建球面图像
        :param pixel_radius: 球体半径(单位:像素)
        :return: 球面图像
        """
        return np.zeros((2*pixel_radius, 2*pixel_radius, 3), dtype=np.uint8)

    def convert_image(self, image, radius_mm):
        """
        将全景图像转换为球面图像
        :param image: 输入的全景图像
        :param radius_mm: 球体半径(单位:毫米)
        :return: 转换后的球面图像
        """
        # 计算像素半径
        pixel_radius = np.int_(radius_mm * self.dpi / 25.4)
        
        # 生成球面坐标
        phi, theta = self.calculate_spherical_coordinates(image.shape[:2])
        x, y, _ = self.spherical_to_cartesian(phi, theta, pixel_radius)
        
        # 创建新图像
        new_image = self.create_sphere_image(pixel_radius)
        
        # 映射坐标并填充图像
        x_mapped, y_mapped, valid_indices = self.map_coordinates_to_image(x, y, pixel_radius)
        new_image[y_mapped[valid_indices], x_mapped[valid_indices]] = image[valid_indices]
        
        return new_image