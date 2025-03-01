## 使用说明

### 目录结构
- `modules/`
  - `batch_fusion.py`: 批量融合点云数据的模块
  - `colored_pcd.py`: 处理彩色点云数据的模块
  - `pano_image2pano_ball.py`: 将全景图像转换为球面图像的模块
- `tests/`
  - `test_image2ball.py`: 测试将全景图像转换为球面图像的脚本
- `utils/`
  - `coordinate_transformation.py`: 坐标变换相关的工具类
  - `dimensional_transformation.py`: 维度变换相关的工具类

### 安装依赖
确保已经安装了以下依赖库：
- `opencv-python`
- `numpy`
- 其他依赖库请根据代码中的导入语句自行安装

### 使用示例

#### 配置
在运行脚本之前，需要根据数据目录进行相应的配置。例如，在 `modules/pano_image2pano_ball.py` 中配置输入图像的路径和输出图像的路径。

```python
def __init__(self, input_dir, output_dir):
    """初始化
    :param input_dir: 输入图像的目录
    :param output_dir: 输出图像的目录
    """
    self.input_dir = input_dir
    self.output_dir = output_dir