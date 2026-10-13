# TACO 官方数据来源记录（provenance）

本文件记录 `datasets/taco/raw/` 下全部官方源数据的获取方式与校验值。
本阶段**未**使用任何第三方重新标注版本。

## 1. 官方工具与标注

| 项 | 值 |
|---|---|
| 官方仓库 | https://github.com/pedropro/TACO |
| 克隆 commit | `29de1a9ba05a647b83a90f18d7772e20bb23d846` (master, 2023-02-13) |
| 克隆位置 | `raw/taco_official_repo/` |
| 使用的标注文件 | `data/annotations.json`（官方已审核标注） |
| 标注 SHA256 | `ac1ec605c9e1fcda767970de3457c69b4a43b62c02aca8c4c80f51d1b5ae8449` |
| 未使用的文件 | `data/annotations_unofficial.json`（官方 README 自述"未经审核，可能不准确"） |
| 未使用的文件 | `data/all_image_urls.csv`（仅供未标注图片参考） |

## 2. 图片获取方式

### 2.1 首选方式（官方 `download.py` 的 Flickr 逐图下载）：失败

按官方 `download.py` 的方式，从 `annotations.json` 的 `flickr_url`
（original 分辨率）逐图下载。

**结果：FAILED。** Flickr 静态 CDN 对本机返回 **HTTP 429 (Too Many Requests)**：

- 12 线程并发：前 125 张中仅 7 张成功，118 张 429。
- 降低到单线程 + 1.5 s 间隔重试：**连续 10 张全部 429**。
- 同一时刻 `flickr_640_url`（`_z`，640px 长边）返回 200 —— 说明是
  original-size（`_o`）接口被限流，不是网络故障。
- 使用 640px 版本不可接受：其分辨率与 `annotations.json` 的 `width/height`
  不一致，会导致分割坐标错位（详见审计报告 §3.3）。

失败日志：`raw/flickr_per_image_attempt_FAILED_HTTP429.log`
（脚本：`tools/taco/download_taco_official.py`，具备失败记录、断点续传、
尺寸校验能力，但在 Flickr 限流下无法完成，故未产出误报成功的 manifest。）

### 2.2 实际采用方式（官方 Zenodo DOI 镜像）：成功

官方 `README.md` 明确给出替代渠道：

> To download the dataset images simply issue `python3 download.py`
> **Alternatively, download from [DOI 10.5281/zenodo.3587843]**

该 DOI 为 TACO 作者本人发布的官方归档（version v1.0），因此属于**官方来源**。

| 项 | 值 |
|---|---|
| DOI | 10.5281/zenodo.3587843 (v1.0) |
| API | https://zenodo.org/api/records/3587843 |
| 归档 | `TACO.zip` |
| 字节数 | 2 719 810 615 |
| 官方 MD5 | `e9149407d883e21a8d224feef8210920` → **校验通过** |
| 本地 SHA256 | `51cd6b009cac1e94decf058e470b1fef64148f9c89923d96dcd54d4db61ceae9` |
| 下载脚本 | `tools/taco/fetch_zenodo_taco.sh`（16 段并行 range 下载） |
| 下载日志 | `raw/zenodo_download_VERIFIED.log` |

**关键交叉验证**：归档内 `TACO/data/annotations.json` 与仓库 `data/annotations.json`
**MD5 完全相同**（`7f54502e321b51332576771beb87d977`），
即归档与 GitHub 仓库携带的是同一份官方标注。

## 3. 本目录文件清单

| 路径 | 说明 |
|---|---|
| `TACO.zip` | 官方归档原件（2.5 GiB）。已完成 MD5 校验；如后续确认无需重解压，可删除以回收空间 |
| `annotations_from_release.json` | 从归档解出的官方 COCO 标注（与仓库版逐字节相同） |
| `taco_official_repo/` | 官方 git 仓库（工具脚本 + 标注 + 文档） |
| `release_metadata/` | 归档内的 `download.py` / `README.md` / `LICENSE` / `requirements.txt` |
| `images/batch_1..15/` | **官方全量图片 1500 张**（另有 15 个官方 per-batch `annotations.json`） |
| `flickr_per_image_attempt_FAILED_HTTP429.log` | Flickr 逐图下载失败记录 |
| `zenodo_download_VERIFIED.log` | Zenodo 下载 + MD5 校验记录 |

## 4. 明确排除的数据源

| 排除对象 | 原因 |
|---|---|
| `~/Desktop/TACO(DST1609).zip` | Roboflow 第三方重新导出（含 YOLO/VOC/COCO 混排、`*.rf.*` 重命名、`data.yaml`），属未经确认的第三方重新标注版本 |
| `data/annotations_unofficial.json` | 官方自述未经审核 |
| Flickr `_z` 640px 版本 | 分辨率与标注坐标系不一致 |
