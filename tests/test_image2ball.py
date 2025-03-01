import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from modules.pano_image2pano_ball import PanoramicImageToSphere

def test_image2ball():
    # 读取图像
    image = cv2.imread('D:\DataVolume\TestDataForPanoramicLidar-master\processed_data\images\LB3-20140924-033636-000000000900.jpg')
    
    # 创建转换器实例
    converter = PanoramicImageToSphere(dpi=72)
    
    # 转换图像
    radius_mm = 100  # 单位为毫米
    new_image = converter.convert_image(image, radius_mm)
    
    # 显示结果
    cv2.imshow('new_image', new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image2ball()