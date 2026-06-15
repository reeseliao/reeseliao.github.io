---
title: "如何判断点是否在 Box 中"
date: 2025-07-26
description: 记录两种在 DistantCount 项目中判断 3D 点是否位于边界框内的方法：向量投影法与坐标系逆变换法。
tags: [PointCloud, KITTI, Python, V2X]
---

在处理 KITTI 等自动驾驶数据集时，我们经常需要进行数据清洗。其中一个最基础但也最重要的操作，就是判断一个 3D 点（Point）是否位于某个 3D 边界框（Bounding Box）的内部。

例如，在训练 PointNet++ 进行分类或计数之前，我需要根据标签把前景点（在vehicles，pedestrians，cyclist内的点）提取出来。这里记录两种在我的 `DistantCount` 项目中用到的实现方法。

### 方法一：向量投影法 (Vector Projection)

这种方法的思路来源于数学中的向量投影。只要我们知道 Box 的 8 个角点，我们就可以利用点积（Dot Product）将目标点投影到 Box 的三个主轴上，判断投影长度是否在轴长范围内。

这种方法比较通用，适用于任意方向的 Box，前提是你已经算出了 Box 的 8 个角点。

**代码实现：**

```python
import numpy as np

def points_in_box(corners, points):
    """
    Checks whether points are inside the box.
    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    
    :param corners: The 8 corners of the box <np.float: 8, 3>
    :param points: The points to check <np.float: 3, n>
    :return: mask <np.bool: n, > indicating which points are inside
    """
    
    # 选取一个角点作为原点 p1
    # 选取相邻的三个角点来确定 Box 的三个局部坐标轴方向
    p1 = corners[0]
    p_x = corners[1]
    p_y = corners[2]
    p_z = corners[3]

    # 计算三个轴的方向向量
    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    # 计算目标点到基准点 p1 的向量 v
    # 注意：这里假设 points 是 (3, N) 的形状，如果 (N, 3) 需要相应调整转置
    v = (points - np.expand_dims(p1, axis=0)).T

    # 将 v 投影到三个轴向量上 (利用点积)
    iv = np.dot(i, v)
    jv = np.dot(j, v)
    kv = np.dot(k, v)

    # 判断投影长度是否在 0 和轴长之间
    # np.dot(i, i) 就是轴长的平方
    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
    
    # 只有三个方向都满足条件，点才在 Box 内
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    return mask
```

### 方法二：坐标系逆变换法 (V2X / LiDAR 坐标系)

在 V2X 或纯点云任务中，我们通常使用 LiDAR 坐标系（Z 轴向上）。这与 KITTI 的相机坐标系（Y 轴向下）不同，因此旋转矩阵是绕 Z 轴 进行的。

这种方法的核心思想是：与其把 Box 的 8 个角点算出来（比较麻烦），不如把所有点云变换到 Box 的局部坐标系中。一旦在这个局部坐标系下，判断点是否在 Box 内就变成了简单的 `abs(x) < length/2` 的范围判断。

**代码实现：**

我的 `DistantCount` 项目是基于 V2X 数据的，因此采用如下的 Z 轴旋转逻辑：

```python
import numpy as np

def in_box_mask(pts, o):
    """
    判断点 pts 是否在由对象 o 定义的 3D Box 中 (V2X/LiDAR Coordinate)
    :param pts: 点云数据 <np.array: N, 3>
    :param o: 包含 box 信息的字典 (cx, cy, cz, l, w, h, ry)
    :return: boolean mask
    """
    
    # 1. 平移 (Translation)
    # 将点云的坐标原点平移到 Box 的中心
    rel = pts - np.array([o['x'], o['y'], o['z']])
    
    # 2. 旋转 (Rotation)
    # V2X 场景下，物体通常绕 Z 轴旋转
    # 我们要把点云"反向"转回与坐标轴对齐的状态，所以使用 -ry
    ry = o['ry']
    c, s = np.cos(-ry), np.sin(-ry)
    
    # 构建绕 Z 轴旋转的矩阵
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
    # 执行旋转变换
    loc = rel.dot(R.T) 
    
    # 3. 范围判断 (Check Boundaries)
    # 在局部坐标系下，Box 的中心就是 (0,0,0)
    return (
        (np.abs(loc[:, 0]) <= o['l'] / 2) &
        (np.abs(loc[:, 1]) <= o['w'] / 2) &
        (np.abs(loc[:, 2]) <= o['h'] / 2)
    ) 
