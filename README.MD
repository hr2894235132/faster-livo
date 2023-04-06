# FasterLIVO
仅供自己学习使用。
参考fast-livo中的视觉部分，增加到faster-lio中并更新相关配置文件

目前已完成平面不确定性的引入和自适应体素地图表征方式的修改，已更新在分支livo-voxel_octree。
## 数据集测试
通过手持设备录制，livox mid70 + realsenceD455
### 室内数据集测试
#### 点云地图
![1](https://user-images.githubusercontent.com/86639226/230259773-f66e0d96-3cc2-4694-8e0a-1f5ec3576c07.png)
![2](https://user-images.githubusercontent.com/86639226/230259774-5ae492e9-d399-4954-aaed-0b977ac359e9.png)

#### 自适应体素表征效果
![ezgif com-gif-maker](https://user-images.githubusercontent.com/86639226/230259661-26307a5a-6185-4948-a236-1c11a430375f.gif)
![ezgif com-video-to-gif](https://user-images.githubusercontent.com/86639226/230259665-2444fc81-9946-4657-acb4-19822264ed82.gif)


## TODO
- [x] 通过体素的方式表征地图
- [x] 增加平面确定性
- [x] 修改发布函数，支持在工程运行过程中，任意切换想要观察的layer下对应的平面特征，不再需要不断在运行前修改配置文件


