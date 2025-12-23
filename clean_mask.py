import os
from PIL import Image

# mask_dir = r'D:\Project\Python\QuickStart\Pytorch-UNet\data\masks'
# files = sorted(os.listdir(mask_dir))
#
# for i, f in enumerate(files):
#     img = Image.open(os.path.join(mask_dir, f))
#     if img.mode != 'L':
#         print(f"找到问题文件！索引: {i}, 文件名: {f}, 模式: {img.mode}")

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

mask_dir = r'D:\Project\Python\QuickStart\Pytorch-UNet\data\masks'
out_dir  = r'D:\Project\Python\QuickStart\Pytorch-UNet\data\masks_clean'
os.makedirs(out_dir, exist_ok=True)

files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

bad = []
for f in tqdm(files, desc="Cleaning masks"):
    p = os.path.join(mask_dir, f)
    img = Image.open(p)

    # 1) 统一转单通道
    img_l = img.convert("L")
    arr = np.array(img_l)

    # 2) 自动选阈值：看这张图“亮的部分”到底有多亮
    p95 = np.percentile(arr, 95)   # 95%分位数，避免被少量噪点影响
    thr = 127 if p95 > 127 else 0  # 白很亮→127；否则→0（适合0/1或偏暗的白）

    bin_arr = (arr > thr).astype(np.uint8) * 255

    # 3) 保存
    Image.fromarray(bin_arr, mode="L").save(os.path.join(out_dir, f))

    # 4) 校验：是否只剩 0/255
    u = np.unique(bin_arr)
    if not (len(u) <= 2 and set(u).issubset({0, 255})):
        bad.append((f, u[:10]))

print("done ✅ 输出目录:", out_dir)
if bad:
    print("⚠️ 仍有异常文件(理论上不该出现)：")
    for item in bad[:20]:
        print(item)
else:
    print("所有mask已规范为 L + {0,255}")
