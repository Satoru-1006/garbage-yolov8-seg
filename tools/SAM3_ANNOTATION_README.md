# SAM 3 室内地面垃圾辅助标注工具包

用于 YOLOv8-Seg 室内地面垃圾数据集的**候选标注生成与人工复核**。

下载本目录的 `sam3_indoor_trash_toolkit.zip` 并解压。压缩包内含：

- `sam3_refine_yolo.py`：以已有 YOLOv8-Seg 多边形标注推导边界框，用 SAM 3 生成精细分割候选；
- `sam3_find_missing.py`：用 SAM 3 文本提示查找疑似漏标垃圾；
- `prompts.single_class.example.json`、`prompts.multi_class.example.json`：类别与提示词模板；
- `README.md`：环境安装、运行命令及结果说明。

建议在具备 NVIDIA GPU 的 Linux 主机上运行。脚本默认只测试少量训练图片，**不会覆盖原始数据集**。先审核候选标签、类目映射和遗漏检出结果，再决定是否用于重新训练。不要把自动候选标注当作验证集或测试集的真值。

当前工具包已进行压缩包完整性与 Python 语法检查，**未在实际 SAM 3 GPU 环境和真实项目数据上验证分割质量**。请先用 20～50 张图片做端到端试运行。

使用说明参见压缩包内的 `sam3_indoor_trash/README.md`。
