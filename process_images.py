import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# INPUT (your dataset root)
INPUT_ROOT = "bdata/data0/lsun/bedroom"

# OUTPUT (flat ImageFolder format)
OUTPUT_ROOT = "processed_data"
OUTPUT_CLASS = "bedroom"

os.makedirs(os.path.join(OUTPUT_ROOT, OUTPUT_CLASS), exist_ok=True)

# Transform (exact WGAN preprocessing)
transform = transforms.Compose([
    transforms.Resize(64),        # short side → 64
    transforms.CenterCrop(64),    # make square
])

def process_image(in_path, out_path):
    try:
        img = Image.open(in_path).convert("RGB")
        img = transform(img)

        # Save compressed (reduce size)
        img.save(out_path, "JPEG", quality=85, optimize=True)
    except Exception as e:
        print(f"Skipping {in_path}: {e}")

# Gather all images
image_paths = []
for root, _, files in os.walk(INPUT_ROOT):
    for f in files:
        if f.lower().endswith(".jpg"):
            image_paths.append(os.path.join(root, f))
            print(f"found {f.lower()}")

print(f"Found {len(image_paths)} images")

# Process
for i, path in enumerate(tqdm(image_paths)):
    out_name = f"img_{i:07d}.jpg"
    out_path = os.path.join(OUTPUT_ROOT, OUTPUT_CLASS, out_name)
    process_image(path, out_path)

print("Done.")