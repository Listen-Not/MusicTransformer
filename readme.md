# Music Transformer
学习练手项目，实现了论文方法，代码整体基于下述仓库精简重写
- 原文地址：[Music Transformer: Generating Music with Long-Term Structure](https://arxiv.org/abs/1809.04281) 
- 参考仓库：[MusicTransformer-pytorch](https://github.com/jason9693/MusicTransformer-pytorch)
## 使用方法
使用了python3.12.0解释器，需要python三方库
* pretty_midi 0.2.10
* pytorch 2.6.0
* torchmetrics==1.7.1
* tqdm==4.67.1


克隆仓库
```powershell
$ git clone https://github.com/Listen-Not/MusicTransformer.git
```
下载midi文件，也可以自己准备文件
```powershell
$ cd Trainfiles
$ python downfiles.py
```
把midi文件转成离散事件
```powershell
$ cd ..
$ cd processor
$ python preprocessor.py
```
训练模型
```powershell
$ cd ..
$ python train.py
```
生成midi音乐
```powershell
$ python generate.py
```