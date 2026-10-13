# TACO 第一阶段数据准备审计报告

**项目**：`<repo-root>`
**目标模型**：`yolov8m-seg.pt`（Ultralytics 8.4.176 / PyTorch 2.14.1+cu130 / RTX 5090）
**本轮范围**：仅 TACO 第一阶段**数据准备与审计**，不启动训练、不修改主模型
**报告日期**：2026-10-13

---

## 0. 交付标志

```
TACO_DOWNLOAD        = PASS
TACO_IMAGES          = 1500
COCO_ANNOTATIONS     = PASS
TACO_CLASS_MAPPING   = PROVISIONAL
YOLO_SEG_CONVERSION  = BLOCKED
READY_FOR_STAGE1     = NO
```

| 标志 | 判定 | 依据 |
|---|---|---|
| `TACO_DOWNLOAD` | **PASS** | 官方 Zenodo DOI 归档下载完成并通过官方 MD5 校验（§1.2）。未使用任何第三方重新标注版本 |
| `TACO_IMAGES` | **1500** | 官方归档全量图片 1500 张，磁盘 1500 个文件，与 COCO `images` **一一对应，缺失 0** |
| `COCO_ANNOTATIONS` | **PASS** | 官方 `annotations.json` 结构完整、引用无孤儿、无未知类别；与仓库版逐字节相同（§1.3） |
| `TACO_CLASS_MAPPING` | **PROVISIONAL** | 60 类已逐类给出候选映射与 UNRESOLVED 标记，但存在 **6 项需人工裁决**的决定（§2.4） |
| `YOLO_SEG_CONVERSION` | **BLOCKED** | 按任务约束，类别映射未确认时**不生成最终 YOLO 训练标签**。方案已就绪（§5），`yolo_seg/` 刻意为空 |
| `READY_FOR_STAGE1` | **NO** | 阻塞项：D1–D7 未裁决 + `Unlabeled litter` 处理策略未定（§6） |

**未做且明确不做的事**：未启动训练；未修改 `yolov8m-seg.pt` 或 `weights/`；
未修改 `datasets/garbage_seg_merged_full1444/`（§7 已验证未被触碰）；
未使用室内数据集 test 划分来调 TACO 超参数。

---

## 1. TACO 数据来源

### 1.1 服务器既有 TACO 排查

`datasets/taco/` **不存在**，无既有数据可复用（避免了重复下载）。

服务器上唯一已存在的 TACO 相关文件是 `~/Desktop/TACO(DST1609).zip`
（2.97 GB / 18733 个文件）。经检查其为 **Roboflow 第三方重新导出**：
内含 `*.rf.<hash>.jpg` 重命名图片、混排的 YOLO `images/{train,valid,test}`
+ `data.yaml` + COCO JSON + VOC XML 三套格式。

**判定：不使用。** 它是未经确认的第三方重新标注版本，违反任务约束。

### 1.2 图片获取：两次尝试

**尝试 1 —— 官方 `download.py` 的 Flickr 逐图下载：FAILED（HTTP 429）**

按官方方式从 `annotations.json` 的 `flickr_url`（original 分辨率）下载：

| 并发 | 结果 |
|---|---|
| 12 线程 | 前 125 张中 **仅 7 张成功，118 张 HTTP 429** |
| 单线程 + 1.5 s 间隔 | **连续 10 张全部 429** |
| 同时刻 `flickr_640_url`（`_z`） | HTTP 200 |

即：Flickr 对本机针对 **original-size（`_o`）** 接口限流，`mib` 头显示限流票据，
且不返回 `Retry-After`。**这不是网络故障，也不是并发过高的偶发问题。**

使用 640px 版本**不可接受**：其分辨率与标注 `width/height` 不一致，
会让分割坐标与像素错位（§3.3）。因此放弃该路径。

> 过程中的失败**未被静默忽略**：脚本 `tools/taco/download_taco_official.py`
> 具备逐图失败记录、断点续传、尺寸校验与 MD5 能力；失败日志保存为
> `raw/flickr_per_image_attempt_FAILED_HTTP429.log`。

**尝试 2 —— 官方 Zenodo DOI 镜像：PASS**

官方 `README.md` 明确提供替代渠道：

> Alternatively, download from **[DOI 10.5281/zenodo.3587843]**

该 DOI 由 TACO 作者本人发布（v1.0），属**官方来源**：

| 项 | 值 |
|---|---|
| 归档 | `TACO.zip`，2 719 810 615 字节 |
| 官方 MD5 | `e9149407d883e21a8d224feef8210920` → **校验通过** |
| 本地 SHA256 | `51cd6b009cac1e94decf058e470b1fef64148f9c89923d96dcd54d4db61ceae9` |
| 下载方式 | `tools/taco/fetch_zenodo_taco.sh`（16 段并行 range，避免单连接 ~240 kB/s 的 3 小时耗时） |

### 1.3 来源交叉验证

| 校验 | 结果 |
|---|---|
| 归档内 `TACO/data/annotations.json` MD5 | `7f54502e321b51332576771beb87d977` |
| 仓库 `data/annotations.json` MD5 | `7f54502e321b51332576771beb87d977` → **完全相同** |
| 归档 SHA256（审计口径） | `ac1ec605c9e1fcda767970de3457c69b4a43b62c02aca8c4c80f51d1b5ae8449` |

即归档与 GitHub 仓库携带的是**同一份官方标注**，来源可靠性得到双向确认。

### 1.4 下载结果统计

| 项 | 数值 |
|---|---|
| 标注来源 | `github.com/pedropro/TACO` @ `29de1a9` + 官方 Zenodo DOI 镜像 |
| 实际图片数量 | **1500**（磁盘 1500，`batch_1..15`） |
| 成功下载 | **1500** |
| **失败/缺失** | **0**（缺失清单为空；无静默丢弃） |
| 图片总体积 | 2.5 GiB |
| COCO annotations | `images` 1500 / `annotations` 4784 / `categories` 60 —— **完整** |
| 额外模块 | `scene_annotations` 4296（1496 图）+ `scene_categories` 7 —— **非实例分割，本阶段不使用** |
| 图片格式 | 833 `*.jpg` + 667 `*.JPG`（大小写混合，转换时须按扩展名不敏感处理） |

---

## 2. 类别分析

### 2.1 TACO 原生类别与实例分布

60 个类别、4784 个实例。完整逐类实例数见 `taco_audit.json`
（`category_instance_counts`）。分布极端不均：

| 排名 | 类别 | 实例 | 图 |
|---|---|---|---|
| 1 | Cigarette | 667 | 227 |
| 2 | Unlabeled litter | 517 | 269 |
| 3 | Plastic film | 451 | 310 |
| 4 | Clear plastic bottle | 285 | 225 |
| 5 | Other plastic | 273 | 171 |
| … | … | … | … |
| 59 | Carded blister pack | 1 | 1 |
| 60 | Plastified paper bag | **0** | 0 |

- **19 个类别实例数 < 10**
- **1 个类别实例数为 0**（`Plastified paper bag`）
- 最大/最小实例比 = **667 : 0**（含零类），非零类间最大/最小 = 667 : 1
- 单图最多实例 = **90**

### 2.2 与项目现有 9 类的逐类比较

**方法**：按"材质 + 形态"映射，而非名称字面相似。理由是项目 9 类是多源合并
（`waste_ds_c3_neg_yolo` 5 类 + `half_dataset_yolo` 9 类）的产物，其语义边界
必须从 TACO 的 supercategory 结构推断，不能靠字符串匹配。

完整逐类结果见 `tools/taco/taco_class_mapping_candidate.json`。分五档：

| 档位 | 类别数 | 实例数 | 含义 |
|---|---|---|---|
| `CONFIRMED_CANDIDATE` | 20 | **2277** | 高置信候选，可直接对应 |
| `PROVISIONAL` | 10 | **787** | 语义可用但必需人工裁决 |
| `OTHER_CANDIDATE` | 27 | **1198** | 杂项垃圾，**仅在 `other` 被正式批准后**才可用 |
| `UNRESOLVED` | 1 | **5** | 无可靠对应，**禁用**（Six pack rings） |
| `EXCLUDE` | 1 | **517** | 结构性禁用（Unlabeled litter） |
| `EMPTY` | 1 | 0 | 零实例 |
| 合计 | 60 | 4784 | 完全对账 |

**高置信候选（`CONFIRMED_CANDIDATE`）**：

| TACO 类别 | → 9 类 | 实例 |
|---|---|---|
| Other plastic bottle / Clear plastic bottle | `bottle` | 50 / 285 |
| Other carton / Egg carton / Corrugated carton / Meal carton / Pizza box | `box` | 201 |
| Food Can / Drink can | `can` | 34 / 229 |
| Cigarette | `cigarette_butt` | 667 |
| Food waste | `food_waste` | 8 |
| Magazine paper / Tissues / Normal paper | `paper` | 12 / 42 / 82 |
| Garbage bag / Single-use carrier bag / Polypropylene bag | `plastic_bag` | 31 / 61 / 3 |
| Other plastic wrapper / Crisp packet | `wrapper` | 260 / 39 |
| Other plastic | `other` | 273 |

### 2.3 语义不确定 ⇒ UNRESOLVED / 不擅自归并

严格遵守以下三条，**未**把无法匹配的垃圾一律归入 `other`：

1. **`Unlabeled litter`（517 实例 / 269 图）标记 `EXCLUDE`**。
   该类的含义是"已定位但未定类的垃圾"。两种错误处理都必须避免：
   - 映射为**背景** ⇒ 把真实垃圾当负样本监督，制造标签错误；
   - 映射为 `other` ⇒ 引入语义噪声并伪造 `other` 的规模。
2. **`other` 未获批准前，`OTHER_CANDIDATE` 层整体保持 UNMAPPED**，
   不以任何形式写入标签。
3. **未映射实例绝不当背景**：若某实例未映射，其所在区域必须在方案中被
   显式处理（§5.6），不能默认落入背景。

**明确的语义冲突（必须人工裁决，不得默认）**：

| 冲突 | 说明 |
|---|---|
| `Plastic film` (451 实例) | TACO supercategory 为 "Plastic bag & wrapper"，**同时**跨越项目 `plastic_bag` 与 `wrapper`；且是 TACO 最大类之一 → 单一决定影响巨大 |
| `Wrapping paper` | 名称含 wrapping，但 supercategory 为 **Paper** → `paper` 还是 `wrapper`？ |
| `Paper bag` / `Paper cup` / `Paper straw` | 材质=纸，形态=袋/杯/吸管 → 项目 `paper` 是否包含这些形态？ |
| `Drink carton` | 纸盒：`box` 还是 `paper`？ |
| `Glass bottle` | 形态=瓶但材质=玻璃 → 项目 `bottle` 是否为塑料专用？ |
| 瓶盖 / 拉环 / 盖 (`Plastic bottle cap` 209、`Pop tab` 99、`Lid` 87) | 容器**配件**而非容器本体，并入 `can`/`bottle` 会污染其几何定义 |

### 2.4 两种训练路线可行性评估

**（A）TACO 原生 60 类训练**

| 维度 | 评估 |
|---|---|
| 可行性 | **低** |
| 类不均衡 | 极严重：667 : 1（非零类），19 类 < 10 实例，1 类为 0 |
| 零实例类 | `Plastified paper bag` 无任何监督信号 |
| 污染类 | `Unlabeled litter` 作为独立类会与真实垃圾类混淆（同一物体既可能标为具体类也可能标为该类） |
| 与项目对接 | **不兼容**：部署需要 9 类输出。原生训练的模型不能直接与室内数据集联合，也无法直接推理交付 |
| 结论 | 仅可作为**独立预训练/表征学习实验**，需单独的 60 类头；对第一阶段 9 类目标**无直接交付价值**，且引入额外风险 |

**（B）统一 9 类训练**

| 维度 | 评估 |
|---|---|
| 可行性 | **中**（需先完成映射裁决） |
| 前提 | D1–D7 全部裁决；映射冻结并版本化 |
| 标签噪声风险 | 中：约 **787 实例（PROVISIONAL）+ 1198 实例（OTHER_CANDIDATE）** 的归并依赖未裁决假设 |
| 漏检/误检风险 | 高：`Plastic film` 与 `Plastic bag & wrapper` 的边界主观，跨类一致性差，容易在 `plastic_bag` / `wrapper` 间产生系统性混淆 |
| 对项目最直接的价值 | **补短板**：室内数据集最缺的 `can`(9) 与 `food_waste`(26) 正好可被 TACO 的 `Food Can`/`Drink can`(263) 显著补充；`cigarette_butt` 亦有 667 实例 |
| 结论 | **推荐路线**，但必须先裁决映射；建议第一阶段只启用 `CONFIRMED_CANDIDATE` 层（2277 实例）以控制噪声，把 `PROVISIONAL` 作为第二阶段增量 |

### 2.5 与室内 9 类的互补性（实例数对比）

| 类 | 室内 1444 | TACO 场景 A（高置信） | TACO 场景 B（+临时） | TACO 场景 C（+other 批准） |
|---|---|---|---|---|
| bottle | 99 | 335 | 439 | 439 |
| box | 584 | 201 | 251 | 251 |
| **can** | **9** | **263** | 273 | 273 |
| cigarette_butt | 121 | 667 | 667 | 667 |
| **food_waste** | **26** | **8** | 8 | 8 |
| other | 1313 | 273 | 273 | 1471 |
| paper | 674 | 136 | 246 | 246 |
| plastic_bag | 774 | 95 | 546 | 546 |
| wrapper | 215 | 299 | 361 | 361 |
| **合计** | 3815 | **2277** | **3064** | **4262** |

**结论**：TACO 对 `can`（9 → +263）补充最为关键；`bottle`、`cigarette_butt`
亦有实质增益。但 **`food_waste` 无法由 TACO 补充**（仅 8 实例），
该类缺失问题必须在阶段二另寻数据源。

---

## 3. 实例分割格式审计

### 3.1 分割格式总体

| 项 | 结果 |
|---|---|
| **polygon**（list of list） | **4784 / 4784（100%）** |
| RLE（dict，含 `counts`/`size`） | **0** |
| 空 segmentation | **0** |
| 非 polygon 非 RLE | **0** |
| `iscrowd` | **全部为 0**（无 crowd 区域） |
| 点数 < 3 的多边形 | **0** |
| 退化多边形（< 3 个不同顶点） | **0** |
| `area` 为 0 / 负 | 0 / 0 |
| `bbox` 非法（宽或高 ≤ 0） | **0** |
| `area` 与多边形面积偏差 > 5% | **1** |
| 多边形点数 | min 3 / max 543 / 平均 23.5 |

**有利结论**：100% 纯 polygon 且无 `iscrowd` ⇒ 转换**不需要** pycocotools，
也**不涉及** RLE 解码，显著降低"标注静默丢失"的风险面。

### 3.2 引用完整性

| 检查 | 结果 |
|---|---|
| 引用了不存在 image 的 annotation | **0** |
| 引用了不存在 category 的 annotation | **0** |
| 无任何 annotation 的 image | **0**（1500 张全部有标注） |
| image `id` / `file_name` / `flickr_url` 重复 | **0 / 0 / 0** |

### 3.3 图像宽高与标注对应关系 ⚠️ 本审计最重要的发现

**用 PIL 的 `Image.size` 对比，560 / 1500（37.3%）张图片尺寸与 COCO 元数据不一致。**

进一步定位为**单一根因：EXIF 方向标签**。

| EXIF Orientation (274) | 图片数 | 说明 |
|---|---|---|
| 无该标签 | 625 | 尺寸一致 |
| 1（正常） | 292 | 尺寸一致 |
| 3（180°） | 13 | **尺寸一致但内容倒置** —— 静默陷阱 |
| **6（旋转 90° CW）** | **558** | 尺寸 w/h 互换 |
| **8（旋转 90° CCW）** | **4** | 尺寸 w/h 互换 |
| 0（非法值） | 8 | 尺寸一致 |

不一致模式高度规整：`(1.333, 0.75)` 共 505 张 + `(1.778, 0.562)` 51 张，
**恰为 90° 旋转导致的宽高互换**，而非缩放。

**决定性证据 —— 标注在哪个坐标系？**

| 证据 | 结果 |
|---|---|
| 所有多边形坐标**均未超出** COCO `width/height` | **0 例外 / 1500** |
| EXIF-6/8 图片中，多边形坐标**超出原始存储像素尺寸**的 | **196** 张 |

196 张图片的多边形坐标**不可能**存在于原始存储像素空间中 ——
证明 **TACO 多边形位于 EXIF 校正后（显示态）坐标系，即等于 COCO `width/height`**。

**加载器一致性验证**：

| 加载方式 | 对 EXIF-6 图片的结果 | 与 COCO 尺寸一致？ |
|---|---|---|
| `PIL.Image.open().size` | 3264×2448（原始，未校正） | ✗ |
| `PIL.ImageOps.exif_transpose()` | 2448×3264 | ✓ |
| `cv2.imread` | 2448×3264（自动校正） | ✓ |
| **`cv2.imdecode`（Ultralytics 实际所用）** | 2448×3264 | ✓ **1500/1500 全部一致** |

并核对 Ultralytics 源码：`data/utils.py::exif_size()` 对
Orientation ∈ {5,6,7,8} 执行 `s = s[::-1]` 换轴，
`utils/patches.py::imread()` 走 `cv2.imdecode` —— **两者一致地应用 EXIF**。

**视觉验证**（`batch_1/000104.JPG`，EXIF-6，7 个实例）：

- 应用 EXIF：多边形**紧密贴合**每个物体（电池、泡罩、喷雾罐、软管、瓶盖）
- 忽略 EXIF：多边形相对物体**整体旋转 90°**，完全错位

**结论与风险**：

> 按 COCO `width/height` 归一化是**正确**的，前提是加载器应用 EXIF ——
> Ultralytics 满足该前提。但**任何忽略 EXIF 的环节都会静默地把 562 张
> （37.5%）图片的掩码旋转 90°/180°**，而且其中 13 张 EXIF-3 的图片
> **尺寸校验会通过**（宽高未变），无法靠尺寸断言发现。

⇒ 转换方案必须在**转换时把 EXIF 方向烘焙进像素**并对齐坐标系（§5.1）。

### 3.4 重复图片

| 检查维度 | 结果 |
|---|---|
| 完全相同的文件内容（SHA256） | **0 组** |
| 相同 `flickr_url` | **0 组** |
| 相同 Flickr photo id | **0 组** |
| 近重复（8×8 dHash, Hamming ≤ 5） | **6 对** |

TACO 的 1500 张图片**没有跨图重复**，这与室内数据集（343 个重复组）形成鲜明对比。
仅 6 对近重复需要在划分时绑定到同一 split。

### 3.5 可能发生标注丢失的转换情况 ⚠️

审计出 **4 类**会造成"静默"损失或畸变的情况：

**（1）重复的 annotation `id` —— 2 对 / 4 条标注**

| 重复 id | 冲突内容 |
|---|---|
| `309` | 图 93 `Cigarette` ↔ 图 101 `Clear plastic bottle` |
| `4040` | 图 1281 `Cigarette` ↔ 图 1300 `Unlabeled litter` |

两者**内容不同**，是**真实存在的两个不同物体被赋了同一个 id**，属官方标注缺陷。
**风险**：任何以 `annotation.id` 为键构建字典的转换器会**静默丢掉每对中的一条**。
其中 `309` 的两条都落在 `CONFIRMED_CANDIDATE` 层 ⇒ 会造成真实标注丢失。
**对策**：转换器必须按**列表顺序**遍历，禁止以 `id` 为键。

**（2）多部件多边形 —— 114 条标注 / 81 张图**

| 部件数 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 14 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|
| 标注数 | 64 | 18 | 16 | 5 | 4 | 1 | 3 | 1 | 1 | 1 |

**100 条**标注的"最大部件"占总面积 **< 90%**（即非最大部件携带实质面积）。
YOLOv8-Seg 标签格式**一行一个实例、一个多边形**，因此必须显式选择策略：

| 方案 | 效果 | 风险 |
|---|---|---|
| 每个部件写独立行 | 保留全部面积 | ❌ 把一个实例拆成多个实例，污染实例计数与指标 |
| 只保留最大部件 | 行数正确 | ❌ 丢失面积（100 条标注实质受影响） |
| 栅格化后取外轮廓 | 面积与实例数都正确 | 可能填洞/合并邻近部件，改变语义 |

属**必须裁决项 D7**，不得默认。

**（3）坐标越界 —— 6 条标注**

`id` 371 / 387 / 707 / 859 / 1621 / 1777，**全部恰好越界 -1 px**（顶点位于 `-1.0`）。
归一化后会出现负值，触发 Ultralytics `verify_image_label` 的
"非归一化坐标"断言。**对策**：转换时裁剪到 `[0, w-1] / [0, h-1]` 并记录。
影响极小，但**必须显式处理而非放任**。

**（4）EXIF 未烘焙（见 §3.3）**

不烘焙 EXIF 时，562 张图片的坐标与像素错位 —— 属最严重的潜在丢失。

### 3.6 其他审计结论

| 项 | 结果 |
|---|---|
| `scene_annotations` | 4296 条 / 1496 图，字段仅 `{image_id, background_ids}`，指向 7 个场景类（Clean / Indoor, Man-made / Pavement / Sand,Dirt,Pebbles / Trash / Vegetation / Water）。**场景级上下文，不是实例分割标签，本阶段不使用** |
| `images/` 下非图片文件 | 15 个官方 per-batch `annotations.json`（官方发布内容，予以保留）；已清理归档带来的 `.DS_Store` |
| 不可读图片 | 0 |
| 图片格式大小写 | `*.jpg` 833 + `*.JPG` 667，转换须大小写不敏感 |
| 单图最大实例数 | 90 |

---

## 4. 数据划分建议

### 4.1 独立性分组

TACO 无内容重复，但仍有 6 对近重复（dHash ≤ 5）。划分前将
**完全相同内容 ∪ dHash≤5 近重复 ∪ 相同 flickr_url / photo id** 合并为
"独立单位"，同一单位内所有图片强制进入同一 split。

结果：**1500 张 → 1495 个独立单位**（6 对近重复合并掉 5 个）。

### 4.2 建议划分

| split | 图片数 | 比例 |
|---|---|---|
| train | **1200** | 80% |
| val | **300** | 20% |

**分层策略**：对最稀缺类别（`food_waste`、`can`、`bottle`、`cigarette_butt`）
做贪心配额，优先把含有稀缺类的独立单位放入 val，保证 val 中稀缺类有足够样本。
本例中 val 覆盖：food_waste 4 图、can 72 图、bottle 97 图、cigarette_butt 71 图
（均达到或超过 20% 配额目标）。

### 4.3 约束遵守

| 约束 | 遵守情况 |
|---|---|
| 不使用室内数据集 test 划分选择 TACO 超参数 | ✅ 遵守。TACO 使用**自己的** train/val；室内 test 独立保留 |
| 不修改已合并的 1444 张室内数据集 | ✅ 遵守（§7 已验证） |
| 重复/近重复不跨 split | ✅ 由独立单位机制保证 |
| 不引入跨数据集泄漏 | ✅ TACO 与室内数据图片内容零重叠（TACO 0 重复组；且来源不同） |

### 4.4 类别不均衡的应对建议（不构成超参调优）

- val 中 `food_waste` 仅 ~4 图，**val 指标对该类不可靠**，不应据此做类别级决策；
- 建议在报告中按类分别汇报，避免被 `cigarette_butt`(667) 与 `other`(1471) 主导总体指标；
- TACO val 与室内 test **分开汇报**，不混合评估。

---

## 5. COCO → YOLOv8-Seg 可复现转换方案

> **本方案已设计完成但未执行**（`YOLO_SEG_CONVERSION = BLOCKED`），
> 原因是 D1–D7 未裁决。以下为待解锁后实现的规范。

### 5.1 坐标系与 EXIF（最高优先级）

1. **转换时把 EXIF 方向烘焙进像素**：对每张图片执行
   `ImageOps.exif_transpose()`，并**清除 Orientation 标签**后重新编码。
   - 效果：转换后**存储像素尺寸 == COCO `width/height`**；
   - 收益：彻底消除加载器差异，任何加载方式（PIL / cv2 / 忽略 EXIF）
     都得到同一结果 ⇒ 560 张的"尺寸不一致"告警归零，562 张的旋转风险归零。
2. **归一化一律使用 COCO `width/height`**，不使用文件实际尺寸：
   `x_norm = x / image.width`，`y_norm = y / image.height`。
3. 转换后**断言** `存储尺寸 == COCO 尺寸`，不一致即报错中止（不静默）。

### 5.2 多边形归一化

- 输出格式：`<class_id> x1 y1 x2 y2 ... xn yn`，坐标归一化到 `[0, 1]`；
- 精度：保留 6 位小数（与 Ultralytics 默认一致，避免量化误差）；
- 多边形**不重采样**（不插值、不简化），保持原始顶点，确保面积不失真。

### 5.3 越界裁剪（6 条标注）

- 顶点裁剪到 `[0, width-1]` / `[0, height-1]`；
- 裁剪后若顶点数 < 3 或面积 < 1 px² ⇒ 丢弃该实例并**计入日志**；
- 已核实这 6 条越界量均为 -1 px，裁剪后不会产生退化。

### 5.4 多部件多边形（114 条，待 D7 裁决）

实现须**参数化**（`--multipart-policy {largest,split,union-contour}`），
并按策略输出差异统计。默认**不预设**，由 D7 决定。
无论选哪种，都须输出受影响标注清单以便复核。

### 5.5 防静默丢失的三条硬规则

1. **禁止以 `annotation.id` 为键**（存在 2 对重复 id，§3.5(1)）；按列表顺序遍历。
2. **逐条对账**：输出 `written + dropped_unmapped + dropped_excluded +
   dropped_degenerate + dropped_multipart_adjust == 输入实例数`，
   以及对 `written <= ` 映射后应写实例数，任一等式不成立即报错。
3. **输出审计清单**：每条标注的 `(annotation_id, image_id, image_file,
   category_id, tier, 目标类, 处理动作)`，便于逐条追溯。

### 5.6 未映射实例的处理（关键，防"变背景"）

YOLOv8-Seg **没有 ignore 区域机制**，因此未映射实例若简单地不写入标签，
其区域会**隐式变成背景负样本**。处理规则：

| 情况 | 处理 |
|---|---|
| 图片含 `Unlabeled litter`（269 图） | **整图排除**（保守方案），避免 517 个真实垃圾区域被当成背景；次选方案：仅排除"只含 Unlabeled litter"的 19 图，但须接受 250 张图上的残余风险 —— 属 D6 |
| 图片含 `UNRESOLVED` 类实例（Six pack rings，5 实例 / 4 图） | **整图排除** |
| 图片含 `OTHER_CANDIDATE` 实例而 `other` 未批准 | **整图排除**（或整图排除至 `other` 批准） |

**淘汰规模必须在报告中显式给出**，不得静默缩减数据集。

### 5.7 输出结构

```
datasets/taco/yolo_seg/
├── data.yaml            # 9 类，绝对路径
├── images/{train,val}/
├── labels/{train,val}/
└── conversion_manifest.json   # 与 §5.5(3) 对应的逐条审计清单
```

- **不生成** `test/`：TACO 第一阶段不使用测试划分，避免与室内 test 混淆；
- 文件名保留 `batch_N__xxxxxx.JPG` 形式前缀化，避免同名覆盖；
- 标签**类别顺序固定**为
  `0 bottle / 1 box / 2 can / 3 cigarette_butt / 4 food_waste / 5 other / 6 paper / 7 plastic_bag / 8 wrapper`。

### 5.8 可复现性

- 固定随机种子；划分结果**落盘为清单文件**而非运行时随机生成；
- 转换脚本幂等（重复运行结果按 SHA256 一致）；
- 记录输入 `annotations.json` 的 SHA256 与映射文件版本；
- 不依赖 pycocotools（本数据集无 RLE），仅需 numpy + Pillow + pandas。

---

## 6. 阻塞项与建议的下一步

| 编号 | 待裁决 | 影响 | 未裁决时默认行为 |
|---|---|---|---|
| **D1** | 是否正式批准 `other` 为兜底类 | 27 类 / 1198 实例 | 保持 UNMAPPED |
| **D2** | `Plastic film`(451/310图) 归 `plastic_bag` 还是 `wrapper` | TACO 最大类之一，两侧都受影响 | 保持 UNMAPPED |
| **D3** | Carton 系列是否等同 `box` | 约 201–246 实例 | 保持 UNMAPPED |
| **D4** | `Glass bottle` 是否并入 `bottle` | 104 实例 | 保持 UNMAPPED |
| **D5** | `paper` 是否含杯/袋/吸管形态 | 110 实例 | 保持 UNMAPPED |
| **D6** | `Unlabeled litter` 所在 269 图如何处理 | 占图像 18%，决定可用规模 | 整图排除 |
| **D7** | 多部件多边形合并策略 | 114 条标注 / 81 图 | 不转换 |

**建议的最小可行解锁路径（风险最小）**：
1. 先裁决 D6（`Unlabeled litter` 策略）与 D7（多部件策略）—— 二者是**数据完整性**问题，与语义无关；
2. 再裁决 D1（`other` 是否可用）；
3. 先只启用 `CONFIRMED_CANDIDATE` 层（**2277 实例 / 高置信**）做第一阶段，
   把 `PROVISIONAL`(D2–D5) 作为第二阶段增量，以控制标签噪声；
4. 至此 `TACO_CLASS_MAPPING` 可升为 `CONFIRMED`，转换与训练方可启动。

---

## 7. 交付物与约束核对

### 7.1 最终目录结构

```
datasets/taco/
├── raw/                                       官方源数据（5.1 GiB）
│   ├── TACO.zip                               官方归档（MD5 已校验，可选保留）
│   ├── annotations_from_release.json          官方 COCO 标注（sha256 记录在案）
│   ├── taco_official_repo/                    官方 git 仓库 @29de1a9
│   ├── release_metadata/                      官方 download.py/README/LICENSE/requirements
│   ├── images/batch_1..15/                    官方全量图片 1500 张
│   ├── SOURCE.md                              来源与校验记录
│   ├── flickr_per_image_attempt_FAILED_HTTP429.log
│   └── zenodo_download_VERIFIED.log
├── yolo_seg/
│   └── CONVERSION_STATUS.md                   说明本目录为何刻意为空
├── taco_audit.json                            机器可读审计结果（20 KB）
├── taco_class_mapping_analysis.json           映射场景与划分建议（4 KB）
└── TACO_STAGE1_AUDIT.md                      本报告

tools/taco/
├── download_taco_official.py                  官方 Flickr 逐图下载器（失败但留证）
├── fetch_zenodo_taco.sh                       官方 Zenodo 归档并行下载
├── audit_taco.py                              COCO + 图片文件审计
├── analyze_class_mapping.py                   映射场景与划分分析
└── taco_class_mapping_candidate.json          候选映射（含 6 项待裁决）
```

**Minimal-artifact 核对**：仅保留官方源数据、必要脚本、类别映射候选与一份审计报告。
辅助性的对齐验证图（用于确认 EXIF 结论）在 `/tmp` 中生成，未纳入交付物。

### 7.2 约束核对

| 约束 | 状态 |
|---|---|
| TACO 数据存放于 `datasets/taco/` | ✅ |
| `taco/raw/` 保存官方图片、COCO JSON 及来源信息 | ✅ |
| `taco/yolo_seg/` 保存后续转换数据 | ✅ 目录已建（暂为空，附原因说明） |
| 优先检查已有数据避免重复下载 | ✅ 已确认 `datasets/taco/` 原本不存在 |
| 不下载到 Windows 本地 | ✅ 全部在 Ubuntu 服务器 |
| 不放入其他项目目录 | ✅ 仅 `datasets/taco/` 与 `tools/taco/` |
| 不修改 `garbage_seg_merged_full1444` | ✅ 1444 图 / 1444 标签，全部文件 mtime = 12:30:07（早于 TACO 工作） |
| 类别映射未确认时不生成最终 YOLO 标签 | ✅ `yolo_seg/` 为空，无任何标签文件 |
| 不启动模型训练 | ✅ 未执行任何训练命令 |
| 不使用未经确认的第三方重新标注版本 | ✅ 已识别并排除 `TACO(DST1609).zip`（Roboflow） |
| 下载失败要记录、不静默忽略 | ✅ Flickr 429 失败有独立日志；最终缺失 0 |

### 7.3 下一步（需人工确认后执行）

1. 裁决 **D1–D7**，冻结 `taco_class_mapping_candidate.json` 为 CONFIRMED 版本；
2. 实现 `tools/taco/convert_coco_to_yolo_seg.py`（按 §5 规范，含逐条对账断言）；
3. 生成 `yolo_seg/{images,labels}/{train,val}` 与 `conversion_manifest.json`；
4. 用 Ultralytics `verify_image_label` 全量校验生成的标签；
5. 确认后再进入第二阶段（TACO 与室内 1444 数据集的联合训练策略）。
