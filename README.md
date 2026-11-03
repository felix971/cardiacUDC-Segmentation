# CardiacUDC 四腔室超声分割

本项目面向四腔心超声图像的逐像素分割，目标是建立可复现的数据处理、二维分割基线与评估流程，并在此基础上比较视频建模方法。

参考起点为[腾讯云文章](https://cloud.tencent.com/developer/article/2399789)中的 VNet2D/VNet3D 路线。文章未提供完整训练代码、精确网络定义及可复用的划分清单，本仓库目前不宣称复现其模型或结果。

## 当前状态

- 已实现：基于数据索引读取单个影像与标签，检查数组形状、空间变换和类别值，统计含非零标签的帧，导出可视化与 JSON 报告。
- 已实现：全量数据审计，包括逐文件的标注帧与标签形式（`scripts/audit_labels.py`）、同一录像检测（`scripts/find_same_videos.py`）和标签编号核对（`scripts/check_label_codes.py`）。结果在 `data/metadata/`，结论见下文“全量审计”。
- 已验证：默认样本的检查脚本可以运行；依赖由 `uv.lock` 固定。
- 已实现：固定的 train/val/test 划分（`scripts/make_split.py` → [baseline_v1.csv](data/splits/baseline_v1.csv)）。同一录像和同一站点同号的记录分在同一组，患者身份没有核实，规则见[划分说明](data/splits/README.md)。
- 进行中：nnU-Net v2 二维基线。模型、预处理和训练都交给 nnU-Net，仓库不写自己的模型或训练循环。`scripts/export_nnunet.py` 把 `baseline_v1` 中 `region` 记录的标注帧导出成 nnU-Net 数据集，并按 `baseline_v1` 写好划分。已在真实数据上训练，val 上逐帧平均 Dice：1000 轮 0.920，500 轮 0.918，见下文“nnU-Net 二维基线”。
- 进行中：VNet2D 与 VNet3D。网络用 [MONAI](https://pypi.org/project/monai/) 现成的 `VNet`，训练仍交给 nnU-Net（`scripts/nnunet_vnet_trainer.py`）；三维数据把整段视频作为一个病例（`scripts/export_nnunet_3d.py`）。两者已在真实数据上各训练 500 轮，VNet3D 在 val 上按视频平均 Dice 为 0.792；和二维模型逐帧对比用 `scripts/compare_models.py`，见下文“VNet2D 与 VNet3D”。改进版 VNet3D（8 帧、整帧视野、浅层只在平面内卷积）没有在 Dataset502 上单独训练，而是用作下文邻帧 A/B 实验的网络。
- 已完成一个配对种子：邻帧 A/B 实验。同一个 8 帧 VNet3D、同一套训练协议，A 组输入重复 8 次的目标帧，B 组输入真实邻帧，只监督目标帧。seed2026 在 val 218 帧上的前景宏平均 Dice：A 85.49%，B 85.80%，B − A 为 +0.31 个百分点，按视频配对检验没有显著差异（p = 0.553）。固定 B 的权重、推理时打乱邻帧顺序，Dice 下降 1.52 个百分点。见下文“邻帧 A/B 实验”。
- 尚未实现：独立测试（test 集还没有用过）。

当前可视化展示原始影像和已有标签，不是模型预测。训练权重和预测都在 `outputs/`，不进 Git。

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

GraphEcho 的跨医院域适应方法与文章中的 VNet 路线不能视为同一实现。

## 快速开始

环境要求：Python 3.12、[uv](https://docs.astral.sh/uv/)。运行依赖为 NumPy、NiBabel、Matplotlib、[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) 和 MONAI，PyTorch 随 nnU-Net 一起安装。检查、审计和划分脚本不需要 GPU。在 Linux 上 `uv sync` 会同时下载 PyTorch 的 CUDA 13 运行库（约 3 GB）；用 GPU 训练需要 NVIDIA 驱动 580 或更新，可用 `uv run --locked python -c "import torch; print(torch.cuda.is_available())"` 确认。

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

- 同一段录像的记录在划分时必须放在同一组，否则测试集里会有训练时见过的视频。`baseline_v1` 还把同一站点里患者编号相同的记录放在同一组，见[划分说明](data/splits/README.md)。
- `Site_R_126/patient-60-4` 的稀疏标签为空，但同一录像的 `label_all_frame/patient-60-4` 逐帧都有轮廓标注。
- `label_all_frame` 的影像画面保留着病人姓名和检查日期，同一录像的 `Site_R_126` 版本把这些文字置为 0。不要发布 `label_all_frame` 的原始帧。

仍未完成患者身份审计：不同目录里同名的记录不一定是同一患者，反过来同一患者也可能用了不同编号。`baseline_v1` 的做法是保守地把同一站点同号的记录放进一组；编号不同的同一患者仍然分不出来。同一录像检测只比较形状相同的视频，被剪短或缩放过的同一录像找不出来，这种情况没有查。整个数据集是否完整可训练，也还要在数据管线里验证。

作者数据加载器将最后一个轴作为视频帧处理，但 NIfTI 格式和三个数组轴本身不能证明第三维是时间或空间。当前示例的 NIfTI 单位未知，不能据此推算帧率。全零标签可能表示未标注，非零标签也不足以证明标注完整。

数据结构与索引字段见[数据说明](data/README.md)；固定划分见[划分说明](data/splits/README.md)。患者身份没有核实，报告测试集结果时要注明这个前提。

## nnU-Net 二维基线

nnU-Net 自动规划网络、预处理和训练，但它要求一帧一个文件的固定目录结构，也不知道哪些帧来自同一段视频。`scripts/export_nnunet.py` 负责这两件事：

- 每个标注帧存成一张二维 NIfTI 图像和一张标签，train、val 放进 `imagesTr`/`labelsTr`，test 放进 `imagesTs`/`labelsTs`，共 970 + 218 + 217 帧。只取 `region` 记录的标注帧，去掉缺类的帧。
- 写 `splits_final.json`，只有一折：fold 0 的训练集和验证集就是 `baseline_v1` 的 train 和 val。没有这个文件，nnU-Net 会把所有帧随机分成 5 折，同一段视频的帧会同时出现在训练和验证里。
- `Site_R_52` 的 8 条 1024×768 和 640×480 记录按像素大小存，nnU-Net 预处理时把它们重采样到 800×600，相当于假设每帧的视野相同；真实像素大小未知。
- 写 `frame_manifest.csv`，每个病例一行：病例名、划分、记录、分组、帧号和原始尺寸。nnU-Net 按病例名报告结果，用这张表能查回是哪段视频的哪一帧。

输出在 `data/processed/nnunet/` 和 `outputs/nnunet/`，都不进 Git。重新导出时加 `--overwrite`，它会先删掉上一次的导出：

```bash
uv run --locked python scripts/export_nnunet.py

# nnU-Net 通过这三个环境变量找目录，每开一个新终端都要重新设置
export nnUNet_raw="$PWD/data/processed/nnunet/raw"
export nnUNet_preprocessed="$PWD/data/processed/nnunet/preprocessed"
export nnUNet_results="$PWD/outputs/nnunet/results"

uv run --locked nnUNetv2_plan_and_preprocess -d 501 -c 2d --verify_dataset_integrity
uv run --locked nnUNetv2_train 501 2d 0
```

默认训练 1000 轮，单卡也要很多小时。可以先加 `-tr nnUNetTrainer_5epochs` 只跑 5 轮，确认流程没问题再正式训练。训练结束时 nnU-Net 会在 val 上算一次 Dice。

选定模型后才在 test 上评估，且只评估一次：

```bash
uv run --locked nnUNetv2_predict -i "$nnUNet_raw/Dataset501_CardiacUDC2D/imagesTs" -o outputs/nnunet/test_pred -d 501 -c 2d -f 0
uv run --locked nnUNetv2_evaluate_folder "$nnUNet_raw/Dataset501_CardiacUDC2D/labelsTs" outputs/nnunet/test_pred -djfile outputs/nnunet/test_pred/dataset.json -pfile outputs/nnunet/test_pred/plans.json
```

`outputs/nnunet/test_pred/summary.json` 里有逐类和平均的 Dice。

## VNet2D 与 VNet3D

VNet（Milletari 等，2016）用 MONAI 的实现 `monai.networks.nets.VNet`，不手写网络。`scripts/nnunet_vnet_trainer.py` 里的 `nnUNetTrainerVNet` 只把 nnU-Net 搭网络的那一步换成 VNet，预处理、裁块、增强、loss、预测和评分都不变，所以 VNet 和 nnU-Net 默认 U-Net 的差别只来自网络本身，包括 VNet 没有 deep supervision（U-Net 低层的额外输出和 loss）。同一个 trainer 按 patch 的维数决定建 VNet2D 还是 VNet3D。VNet 把输入缩小 4 次，每条边长必须是 16 的倍数；nnU-Net 规划出的 patch 满足这一点。

VNet 训练时会随机丢掉整张特征图（默认 50%），位置在最深的两个下采样块和每条 skip 连接上。MONAI 1.6.1 不管二维还是三维都把这些层建成 `Dropout3d`：`VNet` 有 `dropout_dim` 参数，但没有传给内部的模块。二维特征的形状是 (batch, 通道, 高, 宽)，`Dropout3d` 会把它当成一个不带 batch 的 (通道, 深, 高, 宽) 体积，把 batch 维当成通道维，于是一次清掉约一半图像的全部通道。`nnUNetTrainerVNet` 建好 VNet2D 后，把这些层换成同样比例的 `Dropout2d`，每次只丢单张图像的单个通道。2026-10-29 之前训练的 VNet2D 没有这个修正。

nnU-Net 训练和预测时都只在自己的安装目录里按名字找 trainer，所以先把这个文件链接进去。重装环境后要重新链接：

```bash
ln -s "$PWD/scripts/nnunet_vnet_trainer.py" \
    .venv/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/nnunet_vnet_trainer.py
```

VNet2D 直接用二维基线的 Dataset501：

```bash
uv run --locked nnUNetv2_train 501 2d 0 -tr nnUNetTrainerVNet
```

默认训练 1000 轮。想只跑一半，用 `-tr nnUNetTrainerVNet_500epochs`，并和同样跑 500 轮的 U-Net（nnU-Net 自带的 `-tr nnUNetTrainer_500epochs`）比较。不能拿 1000 轮训练中的第 500 轮来比：学习率按总轮数从 0.01 降到 0，1000 轮训练到第 500 轮时学习率还有约 0.0054，而且 nnU-Net 不保存那一轮的权重。

VNet3D 需要另一份数据。`scripts/export_nnunet_3d.py` 把每段 `region` 视频存成一个三维病例（高 × 宽 × 帧），写出 `Dataset502_CardiacUDC3D`，划分同样沿用 `baseline_v1`：train 194 段、val 44 段、test 44 段，参与评分的标注帧和二维相同（970、218、217 帧）。

- 每段视频只有约 5% 的帧有标注。没标注的帧和缺类的帧在标签里记为 5，`dataset.json` 里叫 `ignore`；nnU-Net 训练和评分时跳过这些像素，但网络仍能看到它们的图像。
- 时间方向每帧的间距记为 4。nnU-Net 把第三维当成空间，这个间距让它规划出时间方向较短的 patch（真实尺寸下约 32 帧 × 256 × 192），并且只在图像平面内旋转做增强。
- `frame_manifest.csv` 每个评分帧一行，用来和二维结果逐帧对照。

```bash
uv run --locked python scripts/export_nnunet_3d.py
uv run --locked nnUNetv2_plan_and_preprocess -d 502 -c 3d_fullres --verify_dataset_integrity
uv run --locked nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainerVNet
```

三维数据要存 train 和 val 全部约 2.4 万帧，预处理后占用几十 GB。

### 和二维模型逐帧对比

nnU-Net 给二维模型逐帧打分：val 的 218 帧各算一个 Dice，再取平均。给 VNet3D 却是按视频打分：一段视频里所有评分帧合起来算一个 Dice，再对 44 段视频取平均。两个平均的权重不同，不能直接比较。`scripts/compare_models.py` 把 VNet3D 对每段视频的预测切回那几个评分帧，用 nnU-Net 自己的计数方法逐帧打分，这样每个模型在同样的 218 帧上都有逐类 Dice。

二维模型给 val 的 `summary.json`，三维模型给 val 的预测文件夹（里面有每段视频的预测和 nnU-Net 的 `summary.json`）。脚本会打印每个模型的逐类平均 Dice、按来源文件夹的平均 Dice 和得分最低的几帧，并把每帧每个模型的 Dice 写到 `outputs/nnunet/compare_val/frame_dice.csv`：

```bash
uv run --locked python scripts/compare_models.py \
    --model-2d U-Net=outputs/nnunet/results/Dataset501_CardiacUDC2D/nnUNetTrainer_500epochs__nnUNetPlans__2d/fold_0/validation/summary.json \
    --model-2d VNet2D=outputs/nnunet/colab/summary.json \
    --model-3d VNet3D=outputs/nnunet/results/Dataset502_CardiacUDC3D/nnUNetTrainerVNet_500epochs__nnUNetPlans__3d_fullres/fold_0/validation
```

把逐帧的计数按视频加起来，应当正好得到 nnU-Net 自己的按视频 Dice。脚本会打印两者的最大差值，用来确认帧是沿时间轴切的；差值应为 0。

### 改进版 VNet3D：8 帧、整帧视野

现在的 VNet3D 有两个问题：
- **视野小**：patch 是 32 帧 × 256 × 192，只看到约 14% 的画面（裁剪后的帧中位尺寸是 699 × 517）。
- **时间轴和平面一样处理**：MONAI 的 VNet3D 从第一层起就把每帧和前后各 2 帧混在一起，4 次下采样也都把帧数减半。

改进版分别处理这两点：

- **窗口**：patch 改成 8 帧，平面放大。`scripts/add_vnet_plans.py` 把 Dataset502 的 `nnUNetPlans.json` 复制成 `nnUNetPlans_vnet.json`，加两个配置：
  - `3d_t8_768x512`：8 × 768 × 512，和二维模型的 patch 一样大，几乎是整帧。
  - `3d_t8_512x384`：8 × 512 × 384，约半帧，显存约为前者的一半。

  两者都继承 `3d_fullres`，直接用已经预处理好的三维数据。结果目录名里带 `nnUNetPlans_vnet`，不会和旧结果混在一起。
- **时间轴**：`nnUNetTrainerVNetAniso_500epochs` 建好 MONAI 的 VNet3D 后，替换其中一部分卷积层：
  - 最浅的两级用 1×5×5 卷积，只在平面内下采样（步长 (1, 2, 2)），每帧单独处理。
  - 后三级用 3×5×5 卷积，三个方向一起下采样。

  8 帧的 patch 在各级依次是 8 → 8 → 8 → 4 → 2 帧，所以帧数必须是 4 的倍数。nnU-Net 自己给 Dataset502 规划的三维 U-Net 也是前两级只在平面内卷积。通道数、残差连接、dropout 和 BatchNorm 都不变，参数从 4561 万降到 2733 万。

```bash
uv run --locked python scripts/add_vnet_plans.py
uv run --locked nnUNetv2_train 502 3d_t8_768x512 0 -tr nnUNetTrainerVNetAniso_500epochs -p nnUNetPlans_vnet
```

在 CPU 上逐层估算，和旧 VNet3D 相比：

| 配置 | 激活值 | 计算量 |
|---|---|---|
| `3d_t8_768x512` | 约 3 倍 | 约 2.1 倍 |
| `3d_t8_512x384` | 约 1.5 倍 | 约 1 倍 |

正式训练前，先在显卡上跑几分钟，量出显存和每轮秒数。

## 邻帧 A/B 实验

前面的二维和三维模型同时改变了网络、输入窗口、监督方式和训练配置，分数高低回答不了“邻帧有没有帮助”。这个实验把其他条件都固定，只改输入内容：在同一个 8 帧 VNet3D、同一套训练协议下，把“重复的目标帧”换成“真实邻帧”，看目标帧的分割会不会变好。它不比较二维和三维哪个更好，也不检验运动顺序的机制。冻结的协议见 [neighbor-ab-protocol.md](docs/experiments/neighbor-ab-protocol.md)。

### 设计

- **数据**：`Dataset503_CardiacUDCNeighborAB`，每个病例是一个标注目标帧。8 个通道依次是偏移 −4、−3、−2、−1、0、+1、+2、+3 的帧，越过视频边界时取该视频最近的有效帧，目标帧固定在第 5 个通道（索引 4）。train 970 帧、val 218 帧，和二维基线是同一批目标帧，划分沿用 `baseline_v1`。每帧单独做 z-score 归一化，不用标签做掩膜，也不裁掉零值边框。
- **A、B 两组**：两组的数据、裁块和增强完全相同，8 帧共用同一套空间增强。B 组把 8 张真实帧送进网络；A 组在送进网络前，把已经增强好的目标帧复制 8 次，丢掉邻帧。两组输入形状相同，差别只在是否保留真实邻帧的内容。
- **网络**：上文的改进版 VNet3D，外面包一层 `TargetFrameVNet`：把 [N,8,H,W] 变成 [N,1,8,H,W]，只输出第 5 帧，只有目标帧的标签进入 loss。配置名里的 “2d” 指缓存、loss 和空间推理接口，模型本身是三维的。
- **训练**：patch 768×512，batch 2，500 轮 × 每轮 250 次更新，SGD（学习率 0.01，动量 0.99，Nesterov），weight decay 3e-5，多项式学习率衰减，Dice + 交叉熵。交叉熵换成了数学上等价、能在 CUDA 上严格确定性运行的写法。两组共享初始参数和样本、增强序列，启用 PyTorch 严格确定性算法。
- **推理**：Gaussian 滑窗，步长 0.5，不做镜像，也不在时间上平均。B 组读到了目标之后的 3 帧，所以这是离线分割实验，不能当作实时预测的结果。
- **主指标**：第 500 轮的最终检查点，在全部 218 个 val 目标帧上的前景宏平均 Dice：每类先对 218 帧取平均，再对 4 个前景类取平均。

### seed2026 结果

seed2026 在同一台 Ubuntu 主机上先训练 A、再训练 B，各约 9 小时，通过了冻结协议的严格配对验收：初始参数、样本、增强参数和随机状态逐项核对一致。

| 指标 | A：重复目标帧 | B：真实邻帧 | B − A（百分点） |
| --- | ---: | ---: | ---: |
| 前景宏平均 Dice | 85.4896% | 85.7954% | +0.3058 |
| 前景宏平均 IoU | 77.5536% | 78.0731% | +0.5195 |
| LV Dice | 84.4372% | 85.9400% | +1.5028 |
| LA Dice | 83.4671% | 83.3495% | −0.1176 |
| RA Dice | 89.2586% | 89.4638% | +0.2052 |
| RV Dice | 84.7957% | 84.4284% | −0.3672 |

换一种权重汇总，差值的符号会变：

| 汇总方式 | 单位数 | A | B | B − A（百分点） |
| --- | ---: | ---: | ---: | ---: |
| 帧等权（主指标） | 218 | 85.4896% | 85.7954% | +0.3058 |
| 视频等权 | 44 | 85.5420% | 85.8445% | +0.3025 |
| 候选组等权 | 29 | 86.3067% | 86.2151% | −0.0916 |

- 逐帧看，B 更高的有 121 帧，A 更高的有 92 帧，5 帧持平。按视频是 23 比 20，1 段持平。按候选组是 15 比 14。
- 视频等权的探索性配对 t 检验 p = 0.553，B − A 的 95% 区间为 −0.7172 到 +1.3223 个百分点。
- 候选组是按记录名线索合并的疑似同一患者，没有核实身份，不能当作患者层面的结果。

### 固定 B 模型的乱序推理

这一步检查训练好的 B 模型是否依赖邻帧的排列。用 seed2026 第 500 轮的 B 检查点，在同样的 218 帧上做 1 次原序推理和 5 次乱序推理。乱序时目标帧固定在第 5 个通道，其余 7 帧换位。每帧每次的排列由字符串 `seed2026-frame-order-v1:<次数>:<病例名>` 的 sha256 决定，可以复现。原序推理得到的 218 张预测，和原验证结果逐像素一致。

| 条件 | 前景宏平均 Dice |
| --- | ---: |
| 原序 | 85.7954% |
| 乱序（5 次平均） | 84.2767% |
| 原序 − 乱序 | +1.5187 个百分点 |

- 5 次乱序的 Dice 依次为 84.3376%、84.1160%、84.3286%、84.3807%、84.2210%，都低于原序。
- 44 段视频中有 40 段原序更高。视频等权的差值为 +1.5263 个百分点，探索性 95% 区间为 0.8563 到 2.1963。
- 这说明这个 B 检查点依赖邻帧的原始排列。但它和 A/B 是两种不同的干预，不能当作同一个效应的两次估计。要知道“有序输入比乱序输入在训练上好多少”，需要另训一个训练时就打乱邻帧的 C 组。

脚本在 `scripts/neighbor_ab_shuffle/`。它们是当时实际运行的版本，原样保存，sha256 与运行记录一致，所以路径和常数都写死在代码里：

- `probe.py`：在仓库根目录运行。它先核对 `outputs/neighbor_ab/formal/seed_2026/` 下 B 组 `checkpoint_final.pth` 的 sha256，然后把 `ordered/`、`shuffle_1/` … `shuffle_5/` 的预测、`permutations.json` 和 `complete.json` 写在**脚本自己所在的目录**。所以要先把它复制到仓库外的一个新目录，再从仓库根目录运行那个副本。每个病例对应哪个排列，依赖 nnU-Net 验证时按排好序的病例名逐个预测，这是 nnU-Net 的默认行为。
- `summarize.py`：读整理过的结果目录（每个条件一个 `summary.json` 和 `permutations.json`），以及 seed2026 验收目录里的 `paired_frame_metrics.csv`，写出 `paired_dice.csv` 和 `dice_comparison.json`。

### seed2027：严格配对验收没有通过

seed2027 原计划在 Colab 上先训练 B、再训练 A。

- B 完成了 500 轮和 218 帧验证。
- A 经历了会话中断和更换 Colab 主机，最后迁到 Ubuntu，从第 482 轮的检查点续训了最后 18 轮。

两组的预测都能独立逐像素复算，A 为 85.8078%，B 为 85.1186%，但两组的累计训练审计摘要不一致，严格配对验收没有通过。诊断发现，在不同主机上，CPU 预处理（DEFAULT 与 AVX2 两条数值路径、双线性 `grid_sample`）可能给出不逐位相同的增强结果，但根因没有完全定位。所以 seed2027 只作描述，不和 seed2026 合并成两个配对种子的平均效应。

### 结论与局限

- 在通过严格验收的这一个配对种子里，真实邻帧让主指标高了 0.31 个百分点，但没有检出统计显著差异；按候选组等权时，差值转为负。
- 只有一个种子，无法估计种子之间的波动。
- 乱序推理说明 B 模型在推理时用到了邻帧的排列。有序输入相对于乱序输入的训练收益，要靠 C 组来回答。
- val 已经用于观察训练过程和做诊断，不能再当作没碰过的测试集；test 至今没有用过。
- 验证帧来自 44 段视频，患者身份没有核实，不能把 218 帧当作 218 个独立患者。
- 训练权重、预测，以及验收和审计记录都不进 Git。

### 运行

`scripts/register_neighbor_ab.py` 会把 `nnunet_vnet_trainer.py`、`nnunet_neighbor_ab_trainer.py` 和 `neighbor_ab_preprocessor.py` **复制**进已安装的 nnU-Net 包，替换掉上文 `nnunet_vnet_trainer.py` 的符号链接。所以最好在单独的 checkout 和环境里运行。下面的命令按各脚本的参数整理：

```bash
# 从二维基线的导出（Dataset501 及其 splits_final.json）生成 8 通道数据集
uv run --locked python scripts/export_neighbor_ab.py

# 注册 trainer、预处理器和冻结的 plans（configs/neighbor_ab_base_plans.json），再预处理
uv run --locked python scripts/register_neighbor_ab.py --nnunet-root data/processed/nnunet
export nnUNet_raw="$PWD/data/processed/nnunet/raw"
export nnUNet_preprocessed="$PWD/data/processed/nnunet/preprocessed"
uv run --locked nnUNetv2_preprocess -d 503 -c 2d

# 在一张卡上依次训练一对 A、B，并在 val 上验证；结果在 outputs/neighbor_ab/formal/seed_2026/
uv run --locked python scripts/run_neighbor_ab_pair.py \
    --nnunet-root data/processed/nnunet --output-root outputs/neighbor_ab/formal \
    --seed 2026 --host-label <主机名> --order A B
```

其余脚本：

- `smoke_neighbor_ab.py`：正式训练前的有界验收，用真实缓存跑少量 A、B 更新，并做全尺寸导出。
- `benchmark_neighbor_ab.py`：测量显存和速度。
- `prepare_neighbor_continuation.py`：在新的输出目录里，从一个确定的检查点续训，并记录来源。
- `validate_neighbor_prefetch.py`：验证 CPU 预取（`NEIGHBOR_AB_PREFETCH=2`）和串行版本逐位等价，并测量吞吐。

单元测试只需要 CPU：

```bash
uv run --locked python -m unittest discover -s tests -p 'test_neighbor*.py'
```

## 后续实验

1. 跑通 nnU-Net 二维基线：按上文导出、预处理并训练 fold 0，看 val 上的 Dice。标注覆盖、标签编号、同一录像和分组划分已经完成，见“全量审计”和[划分说明](data/splits/README.md)。
2. 明确逐类别 Dice 及聚合口径，以验证集选择模型，再进行独立测试。
3. 训练 VNet2D 与 VNet3D，在 val 的同一批标注帧上与二维基线逐类比较 Dice（`scripts/compare_models.py`）。
4. 保存配置、权重和实验记录，在相同数据协议下比较二维与视频模型。
5. 邻帧 A/B 的后续：
   - 先核实患者标识和划分，并事先定好主指标和统计单位。
   - 如果要检验有序输入的训练收益，增加 C 组：用和 B 相同的真实帧，固定目标帧，训练时随机打乱其余邻帧，并在相应的乱序条件下评估。
   - 补足能通过严格配对验收的种子。

## 参考资料

- [腾讯云文章：CardiacUDC 分割项目](https://cloud.tencent.com/developer/article/2399789)
- [数据集作者说明](https://github.com/XiaoweiXu/CardiacUDA-dataset)
- [Kaggle 数据发布页](https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset)
- [GraphEcho 数据加载器](https://github.com/xmed-lab/GraphEcho/blob/main/datasets/cardiac_uda.py)
- [NiBabel 文档](https://nipy.org/nibabel/gettingstarted.html)
- [MONAI](https://pypi.org/project/monai/)
