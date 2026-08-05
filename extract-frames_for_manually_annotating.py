import os
import random
import shutil

def create_golden_dataset(input_dir, output_dir, sample_size=150):

    os.makedirs(output_dir, exist_ok=True)

    all_images = [f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")]
    golden_images = random.sample(all_images, sample_size)

    for img in golden_images:
        src_path = os.path.join(input_dir, img)
        dst_path = os.path.join(output_dir, img)
        shutil.copy(src_path, dst_path)

    print(f"Successfully sampled {sample_size} images into '{output_dir}'.")

if __name__ == "__main__":
    SIZE = 150
    OUTPUT_DIR = "./ds/golden_dataset"
    INPUT_DIR = "./ds/semantic_dataset"

    create_golden_dataset(INPUT_DIR, OUTPUT_DIR, SIZE)