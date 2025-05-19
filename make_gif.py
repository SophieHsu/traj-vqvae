import os

import glob
from PIL import Image
import argparse

import re
from collections import defaultdict

def make_gif(image_dir):
    all_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    # extract base name and numeric suffix
    pattern = re.compile(r"^(.*?)(?:_(\d+))?\.png$")

    groups = defaultdict(list)
    for filename in all_files:
        match = pattern.match(filename)
        if match:
            base = match.group(1)
            index = int(match.group(2)) if match.group(2) is not None else -1
            groups[base].append((index, filename))

    # Sort each group by index
    grouped_files = {
        base: [fname for _, fname in sorted(file_list)]
        for base, file_list in groups.items()
    }

    for group, files in grouped_files.items():
        if len(files) > 1:
            frames = [Image.open(os.path.join(image_dir, file)) for file in files]
            frame0 = frames[0]
            frame0.save(
                os.path.join(image_dir, f"{group}.gif"),
                format="GIF",
                append_images=frames,
                save_all=True,
                duration=600,
                loop=0,
            )

    return grouped_files
 
def main(args):
    make_gif(args.img_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, help="path to directory containing tsne images")

    args = parser.parse_args()
    main(args)