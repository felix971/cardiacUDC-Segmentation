# CardiacUDC 数据目录使用说明

## 核心原则

把数据分成三件事：**原始事实、可重建的处理结果、描述数据的索引/划分**。原始数据只有一份且不原地修改；训练代码通过索引读取文件，而不是依赖手动搬动病例。

## 目录结构

```text
data/
├── raw/
│   └── cardiacUDC_dataset/        # 发布者原始文件，保留名称、字节内容和子目录
│       ├── Site_G_100/
│       ├── Site_G_20/
│       ├── Site_G_29/
│       ├── Site_R_126/
│       ├── Site_R_52/
│       ├── Site_R_73/
│       └── label_all_frame/
├── processed/                    # 以后生成的缩放、归一化、抽帧等结果；当前为空
├── metadata/
│   ├── cases.csv                 # 按原始文件名前缀配对的影像/标签索引
│   ├── files.csv                 # 逐文件大小、CRC32、SHA256、NIfTI 头信息
│   ├── inventory.json            # 盘点汇总及检查范围
│   ├── unpack_report.json        # 首次解压时的历史校验记录
│   └── cleanup_report.json       # 原下载 ZIP 的清理记录
├── splits/
│   └── README.md                 # 后续划分约定；当前没有训练/验证/测试划分
└── README.md
```

原始数据现路径：`data/raw/cardiacUDC_dataset/`，相对于项目根目录。
原先的 `data/cardiacUDC_dataset/` 已整体移动，未保留第二份副本或修改内部病例文件。

## 实际盘点结果

这里的“影像”指文件，不等于独立患者；“有标签”仅表示存在同名前缀的 `_label.nii.gz` 文件。

| 原始目录 | 影像文件 | 标签文件 |
|---|---:|---:|
| Site_G_100 | 97 | 97 |
| Site_G_20 | 21 | 21 |
| Site_G_29 | 29 | 29 |
| Site_R_126 | 85 | 85 |
| Site_R_52 | 52 | 52 |
| Site_R_73 | 70 | 0 |
| label_all_frame | 10 | 10 |
| 合计 | 364 | 294 |

共 658 个 `.nii.gz` 文件；大小共 4,547,904,911 字节。复核通过原解压 CRC32；整理时另生成了 SHA256，并在删除 ZIP 前再次复核全部文件。

- 294 个影像有同名前缀标签，影像/标签的 NIfTI shape 一致。
- 70 个影像没有同名前缀标签文件，均在 `Site_R_73`。
- 目录名末尾数字是发布者命名，**不要把它直接当成本地实际文件数**。
- 未发现压缩文件字节完全相同的影像；这只是 `.nii.gz` 文件 SHA256 检查，**不能排除同一患者或相同影像数组的不同导出版本**。
- 没有修改、重采样、归一化、抽帧、重命名、删除任何原始 NIfTI 文件。

## 如何使用索引

`metadata/cases.csv` 每行是一个“来源目录 + 原始文件名前缀”的记录，共 364 行。
所有 `image_path`、`label_path` 都是**相对于项目根目录**的路径，而非 CSV 所在目录。

主要字段：

- `record_id`：原始来源目录加原始前缀，例如 `Site_G_100/…`；只用于唯一定位文件，不宣称它就是患者 ID。
- `source_folder`：发布者原始目录名称。
- `site_from_folder_name`：仅由 `Site_G_…` / `Site_R_…` 字面提取 G/R；`label_all_frame` 留空，不猜测来源。
- `image_path` / `label_path`：影像与标签路径。无标签时 `label_path` 为空。
- `has_label_file`：`true` / `false`，不表示逐帧标注完整性。
- `image_shape` / `label_shape`：直接来自 NIfTI 头的维度数组；没有重新解释坐标轴。
- `image_datatype_code` / `label_datatype_code`：NIfTI 头中的数据类型编号。
- `image_sha256` / `label_sha256`：原始压缩文件的校验值。
- `patient_group_id`：当前为空，待依据可靠患者/检查对应关系核实。
- `label_coverage`：有标签文件的记录为 `not_audited`，无标签文件为 `no_label_file`。
- `split`：全部为 `unassigned`，没有擅自划分训练集。

CSV 布尔字段是文本。Python 的 `bool('false')` 仍为真，不要这样解析，应使用 `row['has_label_file'] == 'true'`。

## 后续工作约定

1. **raw 只读使用**：不要在这里覆盖原始影像。`.nii.gz` 是需要保留的实际医学影像/标签格式，不是这次应清理的下载 ZIP。
2. **处理结果写入 processed**：例如 `processed/baseline_v1/`；同时保存处理参数、代码版本和输入校验值，以便重建。不强制现在转换格式。
3. **先核实标注，再选训练样本**：有标签文件不意味着每一帧都有有效标注；不要把未标注帧擅自当作背景真值。无标签记录不要伪造全零标签来凑配对。
4. **划分写清单，不搬原始数据**：确认患者/检查对应关系和研究目标后，在 `splits/` 固定 train/val/test 或交叉验证清单。
5. **同一患者、同一检查/视频的关联帧不跨集合**：否则测试集可能不再代表独立样本。不能把 364 行直接理解为 364 个独立患者，也不能仅凭局部文件名前缀认定跨目录患者身份。
6. **保留来源信息**：做普通监督分割还是跨站点泛化，会影响划分方案；当前不把 G/R 目录直接指定为训练/测试。

## Git 与记录

项目 `.gitignore` 已排除原始数据、预处理大文件及数据目录中的压缩文件。索引、划分、说明文档可以版本控制；本次没有执行 commit 或 push。

`unpack_report.json` 是历史记录：其中旧 destination、原 ZIP 路径和 `original_downloads_preserved` 描述的是首次解压时状态。当前路径看 `inventory.json`，ZIP 是否已删除看 `cleanup_report.json`。

原始发布来源记录：
- 数据集页面：`https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset`
- 作者仓库：`https://github.com/XiaoweiXu/CardiacUDA-dataset`

本索引不替代作者的患者映射、标注说明、官方划分或数据使用条件；这些尚未在本次整理中审计。
