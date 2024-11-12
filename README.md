# SPU-PMD: Self-Supervised Point Cloud Upsampling via Progressive Mesh Deformation (CVPR 2024)

This repository is for SPU-PMD introduced in the following paper:  

[**SPU-PMD: Self-Supervised Point Cloud Upsampling via Progressive Mesh Deformation**](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_SPU-PMD_Self-Supervised_Point_Cloud_Upsampling_via_Progressive_Mesh_Deformation_CVPR_2024_paper.html)

Yanzhe Liu, Rong Chen, Yushi Li, Yixi Li, Xuehou Tan

The IEEE/CVF Conference on Computer Vision and Pattern Recognition 2024

<img src="./img/Overview_v2.png" style="zoom:80%;" />

# Usage
The code is tested under Pytorch 1.6.0 and Python 3.6 / pytorch 1.12.1（py3.7_cuda11.3_cudnn8.3.2_0）python 3.7.13. Pretrained weights are available in [here](https://drive.google.com/drive/folders/1pMtT6xVw617xGcBWEcL9icTPKc7NUVPr?usp=drive_link).

1. Install python denpendencies.
   ```shell
   pip install -r requirements.txt
   ```
2. Compile pyTorch extensions.
   ```shell
   cd pointnet2_ops_lib
   python setup.py install
    
   cd ../losses
   python setup.py install
   ```
3. Install uniformloss
   ```shell
   pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
   ```
4. Compile evaluation code
   ```shell
   cd evaluation_code
   cmake .
   make
   ```

# Results

You can download the **test results** of [PU1K](https://drive.google.com/drive/folders/1-Q1_xtnvBqGEfXQKhJOHQN__-OZN_xcY?usp=drive_link) and [PUGAN](https://drive.google.com/drive/folders/1rYZ0__Z1ZiZAJBlBNRk-p01Qd2mYeyRo?usp=drive_link) here.

You can download the **pretrained models** of [PU1K](https://drive.google.com/file/d/1v26YqHQ3CKZjSFOS3zW9F_SooAg5iyI9/view?usp=drive_link) and [PUGAN](https://drive.google.com/file/d/1576rtdgEoaO9D-6EbmwEJbgWbqEY7taf/view?usp=drive_link) here.

# Upsampling Demo

<img src="./img/result-pu1k.png" style="zoom:80%;" />

<img src="./img/result-scannet.png" style="zoom:80%;" />


# Codes
To be updated soon.

# Citation
If you find our code or paper useful, please consider citing
```
@InProceedings{Liu_2024_CVPR,
    author    = {Liu, Yanzhe and Chen, Rong and Li, Yushi and Li, Yixi and Tan, Xuehou},
    title     = {SPU-PMD: Self-Supervised Point Cloud Upsampling via Progressive Mesh Deformation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {5188-5197}
}
```
