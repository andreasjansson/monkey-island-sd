import multiprocessing
import json
import base64
import replicate
import shutil
import os
import random
from PIL import Image
from glob import glob
import numpy as np
from scipy.spatial import KDTree


def filter_dupes():
    vectors = []

    paths = sorted(list(glob("all-images/*.png")))
    for i, path in enumerate(paths):
        if i % 100 == 0:
            print(f"{i + 1}/{len(paths)}")

        img = Image.open(path)
        img = img.resize([32, 16], resample=Image.NEAREST)
        vectors.append(np.array(img).reshape([-1]))

    tree = KDTree(vectors)

    unique = set()
    seen = set()
    indices = np.arange(len(vectors))
    random.shuffle(indices)

    for i in indices:
        if i not in seen:
            dists, idx = tree.query(vectors[i], k=50, distance_upper_bound=900)
            seen |= set(idx.tolist())
            unique.add(i)

    unique = sorted(list(unique))

    os.makedirs("dataset")

    for i in unique:
        in_path = paths[i]
        out_path = os.path.join("dataset", os.path.basename(in_path))
        shutil.copyfile(in_path, out_path)


def get_caption(path):
    model = replicate.models.get("salesforce/blip")
    try:
        with open(path, "rb") as f:
            image_uri = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")

        output = model.predict(image=image_uri)[0]
    except Exception as e:
        print(e)
        return

    with open("dataset/metadata.jsonl", "a") as f:
        line = {"file_name": path, "text": output}
        f.write(json.dumps(line) + "\n")


def get_captions():
    paths = glob("dataset/*.png")
    with multiprocessing.Pool(40) as p:
        p.map(get_caption, paths)


def main():
    pass


if __name__ == "__main__":
    main()
