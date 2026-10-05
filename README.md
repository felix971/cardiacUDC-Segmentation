# CardiacUDC 四腔室超声分割

本项目面向四腔心超声图像的逐像素分割，目标是建立可复现的数据处理、二维分割基线与评估流程，并在此基础上比较视频建模方法。

参考起点为[腾讯云文章](https://cloud.tencent.com/developer/article/2399789)中的 VNet2D/VNet3D 路线。文章未提供完整训练代码、精确网络定义及可复用的划分清单，本仓库目前不宣称复现其模型或结果。

## 当前状态

- 已实现：基于数据索引读取单个影像与标签，检查数组形状、空间变换和类别值，统计含非零标签的帧，导出可视化与 JSON 报告。
- 已验证：默认样本的检查脚本可以运行；依赖由 `uv.lock` 固定。
- 尚未实现：PyTorch 数据管线、模型训练、患者分组划分、独立测试与视频分割。

当前可视化展示原始影像和已有标签，不是模型预测。仓库尚无训练权重或分割性能结果。

## 任务定义

单帧输入为灰度图像，输出为与图像位置对应的类别。按[GraphEcho 数据加载器的 A4C 定义](https://github.com/xmed-lab/GraphEcho/blob/main/datasets/cardiac_uda.py)，候选类别映射如下：

| ID | 类别 |
| ---: | --- |
| 0 | 背景 |
| 1 | 左心室（LV） |
| 2 | 左心房（LA） |
| 3 | 右心房（RA） |
| 4 | 右心室（RV） |

该映射用于当前检查脚本支持的 A4C 样本；不代表全部原始标签均已完成语义核实。脚本遇到 `0..4` 以外的标签会报错，不会自动重编号。

计划中的二维模型接口为：

```text
images: [B, 1, H, W]  float32
logits: [B, 5, H, W]  float32
targets:[B, H, W]     int64
pred:   [B, H, W]     argmax(logits, dim=1)
```

这里的接口尚未实现为训练代码。GraphEcho 的跨医院域适应方法与文章中的 VNet 路线也不能视为同一实现。

## 快速开始

环境要求：Python 3.12、[uv](https://docs.astral.sh/uv/)。当前运行依赖为 NumPy、NiBabel 和 Matplotlib，不需要 GPU 或 PyTorch。

原始影像不包含在 Git 仓库中。先从[数据发布页](https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset)获取数据，按[数据目录说明](data/README.md)保留原始子目录，并确保 [cases.csv](data/metadata/cases.csv) 中相对于仓库根目录的路径能定位本地文件。默认检查需要：

```text
data/raw/cardiacUDC_dataset/Site_G_100/patient1-4_image.nii.gz
data/raw/cardiacUDC_dataset/Site_G_100/patient1-4_label.nii.gz
```

在仓库根目录运行：

```bash
uv sync --locked --python 3.12
uv run --locked python scripts/inspect_case.py
```

指定其他记录或帧：

```bash
uv run --locked python scripts/inspect_case.py --record-id Site_G_100/patient1-4 --frame 4
uv run --locked python scripts/inspect_case.py --help
```

`--frame` 从 0 开始计数；未指定时选择非零标签像素最多的一帧。脚本只读取原始数据，输出到 `outputs/inspection/<source_folder>/<record>/`：

- `frame_XXX.png`：原图、已有标签及叠加图。
- `frame_XXX.json`：形状、存储类型、标签计数、候选标注帧、所选帧信息与检查边界。

## 数据检查范围

现有索引包含 364 个影像记录，其中 294 个有同名前缀标签文件、70 个无标签文件，共涉及 658 个 NIfTI 文件。记录数不等于独立患者数；存在标签文件不等于每帧都有完整标注。

2026-10-02 抽样检查结果：

| 记录 | 观察 |
| --- | --- |
| `Site_G_100/patient1-4` | 数组形状 `(800, 600, 94)`，标签值 `0..4`，6 帧含非零标签；检查的第 4 帧呈区域标签 |
| `label_all_frame/normal-23-4` | 65 帧均有非零标签；检查的第 7 帧呈轮廓形式 |

以上帧号均从 0 开始。区域标签与轮廓标签不能直接混为同一种监督目标；这些观察仅适用于所检查的样本和帧。没有完成全量解码、标注语义、跨目录重复或患者身份审计，也没有据此证明整个数据集完整可训练。

作者数据加载器将最后一个轴作为视频帧处理，但 NIfTI 格式和三个数组轴本身不能证明第三维是时间或空间。当前示例的 NIfTI 单位未知，不能据此推算帧率。全零标签可能表示未标注，非零标签也不足以证明标注完整。

数据结构与索引字段见[数据说明](data/README.md)；固定划分的前提见[划分约定](data/splits/README.md)。目前 `patient_group_id` 尚未核实、`split` 均未分配，不能报告独立测试集性能。

## 后续实验

1. 核实标签语义与标注覆盖、患者或检查分组、跨目录关联记录。
2. 建立二维分割数据管线，先用可靠标注完成小样本过拟合检查。
3. 固定患者分组划分，明确逐类别 Dice 及聚合口径，以验证集选择模型，再进行独立测试。
4. 保存配置、权重和实验记录，在相同数据协议下比较二维与视频模型。

## 参考资料

- [腾讯云文章：CardiacUDC 分割项目](https://cloud.tencent.com/developer/article/2399789)
- [数据集作者说明](https://github.com/XiaoweiXu/CardiacUDA-dataset)
- [Kaggle 数据发布页](https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset)
- [GraphEcho 数据加载器](https://github.com/xmed-lab/GraphEcho/blob/main/datasets/cardiac_uda.py)
- [NiBabel 文档](https://nipy.org/nibabel/gettingstarted.html)
