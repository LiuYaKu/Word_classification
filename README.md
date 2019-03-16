# 领域词分类
这份代码用于区分领域词和非领域词，目前的预训练模型只适用于人力资源领域。

### 环境需求
- python 3.6
- tensorflow 1.4.0
- jieba 0.39

### 推荐配置
- Linux with Tensorflow GPU edition + cuDNN

### 用法

```sh
# 训练模型
python train.py
# 测试模型
python test.py --test_data_path ./data/test.txt --threshold 0.999
test_data_path是要测试的文件的路径。
threshold是阈值最大值为1，越大挑选出来的词越好，但是数量越少，可以调试找出最佳的阈值。
测试结果输出在result文件夹下，以_result结尾的是领域词，以_f_result结尾的是非领域词。
# 更多的参数可以在config.py中自行配置
```