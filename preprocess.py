import json
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm


def main():
    label_paths = list(
        Path("2D Semantic Segmentation/training/labels").iterdir()
    )

    masks_path = Path("./data/masks")
    masks_path.mkdir(parents=True, exist_ok=True)

    for path in tqdm(label_paths):
        mask = Image.new("RGB", (1920, 1080), color=(0, 0, 0))
        draw = ImageDraw.Draw(mask)

        with open(path) as f:
            annotations = json.load(f)["Annotation"]

        for annotation in annotations:
            points = annotation["Coordinate"][0]
            try:
                color = COLORS[CLASS_IDS[annotation["Label"]] - 1]
            except KeyError:
                continue

            draw.polygon(points, color)

        mask.save((masks_path / path.stem).with_suffix(".png"), "PNG")


# 순서대로 `COLORS`와 `CLASS_IDS`는 대응됩니다
COLORS = [
    (139, 69, 19),
    (34, 139, 34),
    (128, 128, 0),
    (72, 61, 139),
    (0, 128, 128),
    (70, 130, 180),
    (0, 0, 128),
    (154, 205, 50),
    (143, 188, 143),
    (153, 50, 204),
    (255, 0, 0),
    (255, 165, 0),
    (255, 255, 0),
    (0, 0, 205),
    (0, 255, 0),
    (0, 255, 127),
    (220, 20, 60),
    (0, 255, 255),
    (255, 127, 80),
    (255, 0, 255),
    (30, 144, 255),
    (240, 230, 140),
    (173, 216, 230),
    (255, 20, 147),
    (238, 130, 238),
    (255, 182, 193),
]

CLASS_IDS = {
    "road": 1,
    "full_line": 2,
    "dotted_line": 3,
    "road_mark": 4,
    "crosswalk": 5,
    "speed_bump": 6,
    "curb": 7,
    "static": 8,
    "sidewalk": 9,
    "parking_place": 10,
    "vehicle": 11,
    "motorcycle": 12,
    "bicycle": 13,
    "pedestrian": 14,
    "rider": 15,
    "dynamic": 16,
    "traffic_sign": 17,
    "traffic_light": 18,
    "pole": 19,
    "building": 20,
    "guardrail": 21,
    "sky": 22,
    "water": 23,
    "mountain": 24,
    "vegetation": 25,
    "bridge": 26,
}

if __name__ == "__main__":
    main()
