#!/usr/bin/env python3
import os
import json
import shutil
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ImageNet subset by synset (SR-friendly, flat output)"
    )

    parser.add_argument(
        "--imagenet-root",
        type=str,
        required=True,
        help="Root dir containing train/ val/ and metadata files",
    )

    parser.add_argument(
        "--synset-list",
        type=str,
        required=True,
        help="Text file: allowed synsets, one per line",
    )

    parser.add_argument("--out-root", type=str, required=True, help="Output directory")

    parser.add_argument(
        "--copy", action="store_true", help="Copy files instead of symlink"
    )

    return parser.parse_args()


def load_synsets(path):
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def build_val_index(val_gt_path, class_index_path):
    """
    return: dict { image_name -> synset }
    """
    with open(val_gt_path) as f:
        labels = [int(x.strip()) for x in f]

    with open(class_index_path) as f:
        class_index = json.load(f)

    idx_to_synset = {int(k): v[0] for k, v in class_index.items()}

    mapping = {}
    for i, label in enumerate(labels):
        img = f"ILSVRC2012_val_{i + 1:08d}.JPEG"
        synset = idx_to_synset[label - 1]  # 1-based -> 0-based
        mapping[img] = synset

    return mapping


def link_or_copy(src, dst, copy=False):
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def extract_train(train_dir, allowed_synsets, out_dir, copy):
    out_train = os.path.join(out_dir, "train")
    os.makedirs(out_train, exist_ok=True)

    for synset in tqdm(sorted(allowed_synsets), desc="Train"):
        src_cls = os.path.join(train_dir, synset)
        if not os.path.isdir(src_cls):
            continue

        for img in os.listdir(src_cls):
            src = os.path.join(src_cls, img)
            dst = os.path.join(out_train, img)
            if not os.path.exists(dst):
                link_or_copy(src, dst, copy)


def extract_val(val_dir, val_index, allowed_synsets, out_dir, copy):
    out_val = os.path.join(out_dir, "val")
    os.makedirs(out_val, exist_ok=True)

    for img, synset in tqdm(val_index.items(), desc="Val"):
        if synset not in allowed_synsets:
            continue

        src = os.path.join(val_dir, img)
        dst = os.path.join(out_val, img)
        if os.path.exists(src) and not os.path.exists(dst):
            link_or_copy(src, dst, copy)


def main():
    args = parse_args()

    imagenet_root = args.imagenet_root
    train_dir = os.path.join(imagenet_root, "train")
    val_dir = os.path.join(imagenet_root, "val")

    gt_path = os.path.join(imagenet_root, "ILSVRC2012_validation_ground_truth.txt")
    class_index_path = os.path.join(imagenet_root, "imagenet_class_index.json")

    allowed_synsets = load_synsets(args.synset_list)
    print(f"Using {len(allowed_synsets)} synsets")

    val_index = build_val_index(gt_path, class_index_path)

    extract_train(train_dir, allowed_synsets, args.out_root, args.copy)
    extract_val(val_dir, val_index, allowed_synsets, args.out_root, args.copy)

    print(f"Subset ready at {args.out_root}")


if __name__ == "__main__":
    main()
