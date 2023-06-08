# 致谢

https://github.com/isantesteban/snug 

The repository of article **[Snug: Self-supervised neural dynamic garments](http://openaccess.thecvf.com/content/CVPR2022/html/Santesteban_SNUG_Self-Supervised_Neural_Dynamic_Garments_CVPR_2022_paper.html)**. This repository provides physical loss function. 

My repository is already include `LISENCE.md` of this repository.

https://github.com/zihui-debug/snug-recurrence 提供了模型训练的主要框架，本仓库是其的一个变体。

版权相关问题请及时联系。

If there is any copyright issue, please contact me. 1729924874@qq.com

# 如何进行训练

进行训练

```sh
python train.py
```

## 设置Batch Size

这个模型训练时占用显存较大。6G显存的显卡是不能设置Batch Size为8的。

修改Batch Size时，需要修改以下2个地方：

1 `train.py`

```python
Line233: batch_size = 2
```

2 `Data.py`

```python
Line49: Batch_size = 2
```

## 用于训练和预测的数据

即`Data.py`中的`_get_pose`函数

```python
Line30: self._pose_path = "assets/CMU"
```

即AMASS数据集的路径，具体请查看https://github.com/isantesteban/snug

## 选择不同的损失函数

本仓库设置了两种不同的`L_bend`函数

如果要使用原SNUG论文的`L_bend`函数

则在`train.py`

```python
Line388: loss = L_inertia + L_strain + L_bend + L_gravity + L_collision
```

如果要使用新的`L_bend`函数，即Willmore泛函

则在`train.py`

```python
Line388: loss = L_inertia + L_strain + L_bend_New + L_gravity + L_collision
```

# 使用训练后的模型进行预测

```sh
python run_snug.py
```

在训练中的每个epoch都会生成一个对应的模型，模型会保存在`./train_model_save`中。

在`snug_utils.py`文件中的`get_model_path()`中设置每种服装将要使用的模型文件路径。

# 渲染

预测后会导出用于blender的`.obj`文件。

具体保存路径可以设置，请参考https://github.com/isantesteban/snug 

后续的步骤请参考https://github.com/isantesteban/snug 





