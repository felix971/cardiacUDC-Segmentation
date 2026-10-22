# CardiacUDC 四腔室超声分割

本项目面向四腔心超声图像的逐像素分割，目标是建立可复现的数据处理、二维分割基线与评估流程，并在此基础上比较视频建模方法。

参考起点为[腾讯云文章](https://cloud.tencent.com/developer/article/2399789)中的 VNet2D/VNet3D 路线。文章未提供完整训练代码、精确网络定义及可复用的划分清单，本仓库目前不宣称复现其模型或结果。

## 当前状态

- 已实现：基于数据索引读取单个影像与标签，检查数组形状、空间变换和类别值，统计含非零标签的帧，导出可视化与 JSON 报告。
- 已实现：全量数据审计，包括逐文件的标注帧与标签形式（`scripts/audit_labels.py`）、同一录像检测（`scripts/find_same_videos.py`）和标签编号核对（`scripts/check_label_codes.py`）。结果在 `data/metadata/`，结论见下文“全量审计”。
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

该映射来自 GraphEcho，本仓库没有独立核实每个编号对应哪个心腔。已核实的是不同版本之间编号一致：同一录像的两个标签版本对照时，出现的编号 1–4 都落在对方的同号区域上。这种对照只覆盖 16 个文件：`label_all_frame` 的 7 个和 `Site_R_126` 的 9 个（含 `Site_R_126` 内部的 `patient-89-4` 与 `patient-90-4`）；其余文件，包括 `Site_R_126` 的另外 76 个、`label_all_frame` 的另外 3 个和全部 `Site_G_*`、`Site_R_52`，都没有同一录像可对照。两个文件是例外，`label_all_frame/patient-5-4` 用 5、6 代替 1、4，`label_all_frame/normal-27-4` 用 6 代替 4，重编号见 [label_code_remap.csv](data/metadata/label_code_remap.csv)。`inspect_case.py` 遇到 `0..4` 以外的标签仍会报错，不会自动重编号。

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

以上帧号均从 0 开始。区域标签与轮廓标签不能直接混为同一种监督目标；这些观察仅适用于所检查的样本和帧。

### 全量审计（2026-10）

| 检查 | 脚本与结果 | 结论 |
| --- | --- | --- |
| 读得出来吗 | `audit_labels.py` → [label_audit.csv](data/metadata/label_audit.csv) | 294 个标签文件中 293 个可读；`Site_R_126/patient-73-4` 的标签文件被截断，读取时报 `EOFError`。 |
| 哪些帧有标注 | 同上 | 一帧至少有 100 个非零标签像素才算有标注；更少的是标注工具留下的零星笔迹（最多 60 像素），完整的标注每帧至少 3000 像素。稀疏视频（5 个带标签的 `Site_*` 目录里每个只标少数几帧的视频，共 284 个）中，除去读不出的和全空的各 1 个，282 个有标注，其中 280 个各标 5 帧；`Site_G_20/patient-5-4` 标 7 帧（其中 2 帧各约 300 像素，只画了 4 号），`Site_R_126/patient-69-4` 标 3 帧；`label_all_frame` 的 10 个视频每帧都有标注。`Site_R_126/patient-60-4` 的标签全为 0。 |
| 区域还是轮廓 | 同上 | 稀疏视频是填满的区域（内部像素比例，即上下左右都是同一类的前景像素占比，≥ 0.95），`label_all_frame` 是轮廓线（0.45–0.66），阈值取 0.8。轮廓要先填充才能当作区域监督。 |
| 每帧四类齐不齐 | 同上 | 6 个标注帧缺类：`Site_G_100/patient40-4` 第 6 帧缺 4；`Site_G_100/patient44-4` 第 12、22 帧缺 2；`Site_G_20/patient-5-4` 第 13、16 帧只画了 4；`label_all_frame/patient-44-4` 第 50 帧（缺哪类未记录）。两个需要重编号的文件，重编号后还没有逐帧复查。 |
| 同一段录像 | `find_same_videos.py` → `same_video_pairs.csv` | 形状相同的 558 对视频中有 9 对是同一段录像：`label_all_frame/X` = `Site_R_126/X`（X 为 patient-5-4、17-4、35-4、39-4、44-4、60-4、64-4、67-4），以及 `Site_R_126/patient-89-4` = `Site_R_126/patient-90-4`。同一录像的冲突比例都是 0；不同录像中最接近的一对（`label_all_frame/normal-27-4` 与 `label_all_frame/patient-5-4`）也有 0.649，远高于阈值 0.01。 |
| 编号是什么意思 | `check_label_codes.py` → `label_code_crosstab.csv`；[label_code_remap.csv](data/metadata/label_code_remap.csv) | 8 对能对照的同一录像中（7 对在 `label_all_frame` 与 `Site_R_126` 之间，1 对在 `Site_R_126` 内部），出现的编号 1–4 都对应对方的同号区域。`label_all_frame/patient-5-4` 的 5 对应 1、6 对应 4（像素对照）；`label_all_frame/normal-27-4` 的 6 对应 4（没有对照版本，按缺失的类别和位置推断，证据较弱）。 |

使用这些数据时：

- 同一段录像的记录在划分时必须放在同一组，否则测试集里会有训练时见过的视频。
- `Site_R_126/patient-60-4` 的稀疏标签为空，但同一录像的 `label_all_frame/patient-60-4` 逐帧都有轮廓标注。
- `label_all_frame` 的影像画面保留着病人姓名和检查日期，同一录像的 `Site_R_126` 版本把这些文字置为 0。不要发布 `label_all_frame` 的原始帧。

仍未完成患者身份审计：不同目录里同名的记录不一定是同一患者，`patient_group_id` 仍为空。同一录像检测只比较形状相同的视频，被剪短或缩放过的同一录像找不出来，这种情况没有查。整个数据集是否完整可训练，也还要在数据管线里验证。

作者数据加载器将最后一个轴作为视频帧处理，但 NIfTI 格式和三个数组轴本身不能证明第三维是时间或空间。当前示例的 NIfTI 单位未知，不能据此推算帧率。全零标签可能表示未标注，非零标签也不足以证明标注完整。

数据结构与索引字段见[数据说明](data/README.md)；固定划分的前提见[划分约定](data/splits/README.md)。目前 `patient_group_id` 尚未核实、`split` 均未分配，不能报告独立测试集性能。

## 后续实验

1. 核实患者或检查分组。标注覆盖、标签编号和同一录像（只比较了形状相同的视频）已经审计，见“全量审计”。
2. 建立二维分割数据管线，先用可靠标注完成小样本过拟合检查。
3. 固定患者分组划分，明确逐类别 Dice 及聚合口径，以验证集选择模型，再进行独立测试。
4. 保存配置、权重和实验记录，在相同数据协议下比较二维与视频模型。

## 参考资料

- [腾讯云文章：CardiacUDC 分割项目](https://cloud.tencent.com/developer/article/2399789)
- [数据集作者说明](https://github.com/XiaoweiXu/CardiacUDA-dataset)
- [Kaggle 数据发布页](https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset)
- [GraphEcho 数据加载器](https://github.com/xmed-lab/GraphEcho/blob/main/datasets/cardiac_uda.py)
- [NiBabel 文档](https://nipy.org/nibabel/gettingstarted.html)
