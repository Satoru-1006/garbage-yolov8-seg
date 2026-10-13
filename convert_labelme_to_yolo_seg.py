import json
import random
import shutil
from pathlib import Path
from PIL import Image

# =====================================================
# 修改这里
# =====================================================
source_dir = Path(r"half_dataset")
save_dir = Path(r"half_dataset_yolo")

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

random.seed(42)

assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

# =====================================================
# 创建目录
# =====================================================
for p in [
    save_dir / "images/train",
    save_dir / "images/val",
    save_dir / "images/test",
    save_dir / "labels/train",
    save_dir / "labels/val",
    save_dir / "labels/test",
]:
    p.mkdir(parents=True, exist_ok=True)

# =====================================================
# 获取所有图片（而不是JSON）
# =====================================================
image_files = []

for ext in ("*.jpg", "*.jpeg", "*.png"):
    image_files.extend(source_dir.glob(ext))

image_files = sorted(image_files)

print(f"发现 {len(image_files)} 张图片")

# =====================================================
# 自动统计类别（只统计有JSON的）
# =====================================================
classes = set()

for img in image_files:

    json_path = source_dir / (img.stem + ".json")

    if not json_path.exists():
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for shape in data.get("shapes", []):
        classes.add(shape["label"])

classes = sorted(classes)
class2id = {c: i for i, c in enumerate(classes)}

print("\n类别：")
for k, v in class2id.items():
    print(v, "->", k)

# =====================================================
# 数据划分
# =====================================================
random.shuffle(image_files)

n = len(image_files)

train_num = int(n * train_ratio)
val_num = int(n * val_ratio)

train_set = set(image_files[:train_num])
val_set = set(image_files[train_num:train_num + val_num])
test_set = set(image_files[train_num + val_num:])

print("\n数据划分：")
print("Train:", len(train_set))
print("Val:", len(val_set))
print("Test:", len(test_set))

# =====================================================
# 转换
# =====================================================
positive = 0
negative = 0

for img in image_files:

    stem = img.stem

    json_path = source_dir / (stem + ".json")

    txt_lines = []

    if json_path.exists():

        positive += 1

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        w = data["imageWidth"]
        h = data["imageHeight"]

        for shape in data.get("shapes", []):

            label = shape["label"]

            if label not in class2id:
                continue

            cls = class2id[label]

            pts = []

            for x, y in shape["points"]:
                pts.append(f"{x / w:.6f}")
                pts.append(f"{y / h:.6f}")

            txt_lines.append(f"{cls} " + " ".join(pts))

    else:
        # 没有JSON，当作负样本
        negative += 1

        w, h = Image.open(img).size
        # txt_lines保持为空即可

    # 保存位置
    if img in train_set:
        img_dst = save_dir / "images/train" / img.name
        txt_dst = save_dir / "labels/train" / (stem + ".txt")

    elif img in val_set:
        img_dst = save_dir / "images/val" / img.name
        txt_dst = save_dir / "labels/val" / (stem + ".txt")

    else:
        img_dst = save_dir / "images/test" / img.name
        txt_dst = save_dir / "labels/test" / (stem + ".txt")

    shutil.copy2(img, img_dst)

    with open(txt_dst, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

# =====================================================
# 生成data.yaml
# =====================================================
yaml_text = f"""path: {save_dir.as_posix()}

train: images/train
val: images/val
test: images/test

names:
"""

for i, c in enumerate(classes):
    yaml_text += f"  {i}: {c}\n"

with open(save_dir / "data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_text)

print("\n==============================")
print("转换完成！")
print(f"正样本：{positive}")
print(f"负样本：{negative}")
print("类别数量：", len(classes))
print("类别：", classes)
print("==============================")
