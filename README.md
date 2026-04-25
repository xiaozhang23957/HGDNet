<div align="center">
  <h1 align="center">Hybrid Gaussian Deformation for Efficient
Remote Sensing Object Detection</h1>
  <div align="center">
        <div class="is-size-4 publication-authors">
              <span class="author-block"> <a href="https://ieeexplore.ieee.org/author/37086125479" target="_blank">Wenda Zhao</a>, </span>
              <span class="author-block"> <a href="https://github.com/undyingjoker" target="_blank">Xiao Zhang</a>, </span>
              <span class="author-block"><a href="https://ieeexplore.ieee.org/author/37089921660" target="_blank">Haipeng Wang</a>, </span>
              <span class="author-block"><a href="https://scholar.google.com/citations?hl=en&user=D3nE0agAAAAJ" target="_blank">Huchuan Lu</a>, </span>
        </div>
        <div class="is-size-4 publication-authors">
        </div>
    </div>
    <h2 align="center">📖 IEEE Transactions on Pattern Analysis and Machine Intelligence 2025</h2>
</div>
<div>
    <h3 style="text-align: center;">Abstract</h3>
    <p>
        Large-scale high-resolution remote sensing images (LSHR) are increasingly adopted for object detection, since they capture finer details. However, LSHR imposes a substantial computational cost. Existing methods explore lightweight backbones and advanced oriented bounding box regression mechanisms. Nevertheless, they still rely on high-resolution inputs to maintain detection accuracy. We observe that LSHR comprise extensive background areas that can be compressed to reduce unnecessary computation, while object regions contain details that can be reserved to improve detection accuracy. Thus, we propose a hybrid Gaussian deformation module that dynamically adjusts the sampling density at each location based on its relevance to the detection task, i.e., high-density sampling preserves more object regions and better retains detailed features, while low-density sampling diminishes the background proportion. Further, we introduce a bilateral deform-uniform detection framework to exploit the potential of the deformed sampled low-resolution images and original high-resolution images. Specifically, a deformed deep backbone takes the deformed sampled images as inputs to produce high-level semantic information, and a uniform shallow backbone takes the original high-resolution images as inputs to generate precise spatial location information. Moreover, we incorporate a deformation-aware feature registration module that calibrates the spatial information of deformed features, preventing regression degenerate solutions while maintaining feature activation. Subsequently, we introduce a feature relationship interaction fusion module to balance the contributions of features from both deformed and uniform backbones. Comprehensive experiments on three challenging datasets show that our method achieves superior performance compared with the state-of-the-art methods.
    </p>
</div>


<div align="center">
<h3>Hybrid Gaussian Deformation Module</h3>
    <img src="docs/fig3.jpg?_t=202506181741" alt="HGD Module" width="60%">
</div>


<div align="center">
    <h3>Comparison of various methods</h3>
    <table width="100%">
        <tr>
            <td align="left" style="width: 50%;">
                <img src="docs/FPS.svg?_t=202506181400" alt="FPS1" width="100%">
            </td>
            <td align="right" style="width: 50%;">
                <img src="docs/FPS2.svg?_t=202505061400" alt="FPS2" width="100%">
            </td>
        </tr>
    </table>
</div>
<div align="center">
    <h3>Qualitative results</h3>
    <img src="docs/fig8.jpg?_t=202505061400" alt="Results" width="100%">
</div>


## 📦 安装 (Installation)
本项目基于 [MMDetection](https://github.com/open-mmlab/mmdetection) 框架开发，环境配置请严格参考官方文档：[🔗 MMDetection 安装指南](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation)

推荐基础安装步骤：
```bash
# 1. 创建虚拟环境
conda create -n mmdet python=3.9 -y
conda activate mmdet

# 2. 安装 PyTorch (请根据你的 CUDA 版本调整)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. 使用 MIM 安装核心依赖
pip install -U openmim
mim install mmdet
mim install mmcv
```
> 💡 完整版本对应表、Windows/Linux 差异及源码编译方式，请查阅官方 [Installation 文档](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation)。

## 🗂️ 数据集准备 (Dataset)
配置文件使用 DOTA 1.0 数据集。MMDetection 默认采用 COCO 格式标注，需先进行格式转换：
- 转换工具参考：[🔗 DOTA 数据集说明](https://mmdetection.readthedocs.io/en/latest/advanced_guides/datasets/dota.html)
- 推荐目录结构：
```
data/
├── dota1_0/
│   ├── trainval/
│   │   ├── images/
│   │   ├── labelTxt/
│   │   └── trainval.json
│   └── test/
│       ├── images/
│       └── test.json
```

## 🚀 训练 (Training)
训练流程完全遵循 MMDetection 官方规范。详细说明请参考：[🔗 MMDetection 训练指南](https://mmdetection.readthedocs.io/en/latest/user_guides/train.html)

### 单卡训练
```bash
python tools/train.py configs/deformable/deformable_faster_rcnn_orpn_r50_fpn_1x_ms_rr_dota10.py
```

### 多卡训练 (推荐)
```bash
# Linux / WSL
bash tools/dist_train.sh configs/deformable/deformable_faster_rcnn_orpn_r50_fpn_1x_ms_rr_dota10.py 4

# Windows (PowerShell)
python -m torch.distributed.run --nproc_per_node=4 tools/train.py configs/deformable/deformable_faster_rcnn_orpn_r50_fpn_1x_ms_rr_dota10.py
```

**常用参数：**
| 参数 | 说明 |
|------|------|
| `--work-dir <path>` | 指定日志与权重保存目录 |
| `--resume <path>` | 从指定 checkpoint 恢复训练 |
| `--cfg-options 'train_dataloader.batch_size=2'` | 运行时覆盖配置参数 |

## 📊 测试与评估 (Testing & Evaluation)
模型评估与推理流程参考官方文档：[🔗 MMDetection 测试指南](https://mmdetection.readthedocs.io/en/latest/user_guides/test.html)

### 官方指标评估
```bash
python tools/test.py configs/deformable/deformable_faster_rcnn_orpn_r50_fpn_1x_ms_rr_dota10.py work_dirs/deformable_faster_rcnn_orpn_r50_fpn_1x_ms_rr_dota10/latest.pth
```
默认输出 COCO 格式 `bbox mAP`。若配置文件启用了旋转框评估（`metric='dota'`），请确保已安装 `dota-devkit` 或按官方说明配置评估器。

### 单图推理可视化
```bash
python demo/image_demo.py demo/demo.jpg configs/deformable/deformable_faster_rcnn_orpn_r50_fpn_1x_ms_rr_dota10.py work_dirs/.../latest.pth --score-thr 0.5
```

> ⚠️ **注意事项**
> 1. 配置文件包含 `ms` (多尺度) 和 `rr` (随机旋转)，测试时请确保 `test_pipeline` 与训练时的数据增强逻辑一致，以保证指标可复现。
> 2. 所有命令默认在项目根目录执行。若配置文件路径不同，请使用绝对路径或调整相对路径。


