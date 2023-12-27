# FasterLIVO
仅供自己学习使用。
参考fast-livo中的视觉部分，增加到faster-lio中并更新相关配置文件

目前已完成平面不确定性的引入和自适应体素地图表征方式的修改，已更新在分支livo-voxel_octree。
已完成std描述子的加入以检测回环，更新到分支fastlivo_voxel_std中。
## 使用说明

### 参数配置说明

### 使用说明


## 数据集测试
通过手持设备录制，livox mid70 + realsenceD455（内置IMU）
### 手持设备图：
![image](https://github.com/hr2894235132/faster-livo/blob/fastlivo_voxel_std/doc/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20230710091310.jpg)
![image](https://github.com/hr2894235132/faster-livo/blob/fastlivo_voxel_std/doc/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20230710091334.jpg)


### 室内建图测试
#### 点云地图
![1](https://user-images.githubusercontent.com/86639226/230259773-f66e0d96-3cc2-4694-8e0a-1f5ec3576c07.png)
![2](https://user-images.githubusercontent.com/86639226/230259774-5ae492e9-d399-4954-aaed-0b977ac359e9.png)
#### 不同分辨率自适应体素表征效果
![ezgif com-gif-maker](https://user-images.githubusercontent.com/86639226/230259661-26307a5a-6185-4948-a236-1c11a430375f.gif)
![ezgif com-video-to-gif](https://user-images.githubusercontent.com/86639226/230259665-2444fc81-9946-4657-acb4-19822264ed82.gif)
### 室外建图测试
#### 点云地图



#### 不同分辨率自适应体素表征效果



### 室内回环修正测试
+ 数据集说明：1,2两个场景分别为同一场景下黑夜和白天。**（这里的黑夜在录制的时候出现了灯突然熄灭再打开的情况，所以导致出现了很大的漂移现象）**
#### 场景1
![1](https://github.com/hr2894235132/faster-livo/blob/fastlivo_voxel_std/doc/1.png)

#### 场景2

#### 场景3

### 室外回环修正测试





## TODO
- [x] 通过体素的方式表征地图
- [x] 增加平面确定性
- [x] 修改发布函数，支持在工程运行过程中，任意切换想要观察的layer下对应的平面特征，不再需要不断在运行前修改配置文件
- [x] 加入回环检测方案
- [ ] 考虑加入std描述子的离线保存与加载功能，为后续复用离线地图模块提供特征

## Acknowledgments
Thanks for:

FAST-LIO2：https://github.com/hku-mars/FAST_LIO.git

FASTER-LIO：https://github.com/gaoxiang12/faster-lio.git

STD：https://github.com/hku-mars/STD.git

VoxelMap:https://github.com/hku-mars/VoxelMap.git




