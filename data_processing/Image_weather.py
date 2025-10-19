import os
import imageio
import imgaug.augmenters as iaa

INPUT_DIR = "<DATA_DIR>/scenario8/unit1/camera_data"
OUTPUT_DIR = "<DATA_DIR>/scenario8/Image_weather"

rain_aug = iaa.Rain(drop_size=(0.1, 0.2), speed=(0.1, 0.3))
fog_aug = iaa.Fog()
snow_aug = iaa.Snowflakes(flake_size=(0.2, 0.5), speed=(0.01, 0.05))

AUGMENTERS = {
    "rainy": rain_aug,
    "foggy": fog_aug,
    "snowy": snow_aug,
}


for weather in AUGMENTERS.keys():
    os.makedirs(os.path.join(OUTPUT_DIR, weather, "unit1", "camera_data"), exist_ok=True)


count_total = 0
count_success = 0
jpg_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".jpg")]

print(f"Found {len(jpg_files)} images in {INPUT_DIR}")

for filename in jpg_files:
    count_total += 1
    image_path = os.path.join(INPUT_DIR, filename)

    try:
        image = imageio.imread(image_path)
    except Exception as e:
        print(f"[WARN] Failed to read {filename}: {e}")
        continue

    # Apply all weather augmentations
    for weather, augmenter in AUGMENTERS.items():
        try:
            image_aug = augmenter(image=image)
            out_path = os.path.join(OUTPUT_DIR, weather, "unit1", "camera_data", filename)
            imageio.imwrite(out_path, image_aug)
        except Exception as e:
            print(f"[ERROR] Weather={weather}, file={filename}: {e}")
            continue

    count_success += 1
    if count_total % 50 == 0:
        print(f"[INFO] Processed {count_total} images...")

print(f"\n✅ Weather augmentation completed.")
print(f"   Total: {count_total} | Successfully processed: {count_success}")
