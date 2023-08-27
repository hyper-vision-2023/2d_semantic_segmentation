import imagehash
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def main():
    image_paths = list(
        Path("2D Semantic Segmentation/training/images").iterdir()
    )
    hashes = dict()

    dhash_z_transformed = with_ztransform_preprocess(
        imagehash.dhash, hash_size=4
    )

    for image_path in tqdm(image_paths, "Detecting similar images"):
        image_hash = dhash_z_transformed(image_path)

        if image_hash in hashes:
            hashes[image_hash].append(image_path)
        else:
            hashes[image_hash] = [image_path]

    for image_paths in tqdm(hashes.values(), "Removing similar images"):
        if len(image_paths) < 2:
            continue

        for image_path in image_paths[1:]:
            image_path.unlink()

            label_path = (
                image_path.parents[1] / "labels" / image_path.stem
            ).with_suffix(".json")
            label_path.unlink()


def alpharemover(image):
    if image.mode != "RGBA":
        return image
    canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
    canvas.paste(image, mask=image)
    return canvas.convert("RGB")


def with_ztransform_preprocess(hashfunc, hash_size=8):
    def function(path):
        image = alpharemover(Image.open(path))
        image = image.convert("L").resize(
            (hash_size, hash_size), Image.Resampling.LANCZOS
        )
        data = image.getdata()
        quantiles = np.arange(100)
        quantiles_values = np.percentile(data, quantiles)
        zdata = (
            np.interp(data, quantiles_values, quantiles) / 100 * 255
        ).astype(np.uint8)
        image.putdata(zdata)
        return hashfunc(image)

    return function


if __name__ == "__main__":
    main()
