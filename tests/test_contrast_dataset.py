"""T30: CLAHE copy of the raw photographs."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")

from src.data.create_contrast_dataset import clahe_rgb, main  # noqa: E402
from src.data.create_roi_dataset import find_source_image  # noqa: E402


def low_contrast_photo(width: int = 120, height: int = 80) -> Image.Image:
    # A reddish gradient squeezed into a narrow lightness band, like a flat, dim photograph
    x = np.linspace(0, 1, width)[None, :, None]
    y = np.linspace(0, 1, height)[:, None, None]
    base = 100 + 30 * (0.6 * x + 0.4 * y)
    rgb = np.concatenate([base + 40, base, base - 10], axis=2)
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def lab(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB).astype(float)


def test_clahe_raises_lightness_contrast_and_keeps_colour():
    photo = low_contrast_photo()
    before, after = lab(photo), lab(clahe_rgb(photo))
    # The clip limit caps the gain; on this smooth gradient it is about 1.4x
    assert after[:, :, 0].std() > 1.25 * before[:, :, 0].std()
    # a* and b* are only changed by rounding in the colour conversion
    assert np.abs(after[:, :, 1:] - before[:, :, 1:]).mean() < 3


def test_tree_is_mirrored_upright_without_exif(tmp_path):
    root, out = tmp_path / "raw", tmp_path / "clahe"
    (root / "gsw").mkdir(parents=True)
    photo = low_contrast_photo(120, 80)
    exif = photo.getexif()
    exif[0x0112] = 6  # stored sideways: readers must rotate it to 80 x 120
    photo.save(root / "gsw" / "P0001.JPG", exif=exif)
    low_contrast_photo(60, 60).save(root / "gsw" / "P0002.JPG")
    (root / "gsw" / "notes.txt").write_text("not an image")
    (root / "gsw" / "._P0001.JPG").write_bytes(b"\x00\x05\x16\x07 macOS metadata, not a JPEG")

    assert main(["--image-root", str(root), "--output-dir", str(out), "--workers", "1"]) == 0
    with Image.open(out / "gsw" / "P0001.JPG") as copy:
        assert copy.size == (80, 120)
        assert 0x0112 not in copy.getexif()
    assert not (out / "gsw" / "notes.txt").exists()
    assert not (out / "gsw" / "._P0001.JPG").exists()
    manifest = pd.read_csv(out / "contrast_manifest.csv")
    assert manifest["relative_path"].tolist() == ["gsw/P0001.JPG", "gsw/P0002.JPG"]
    assert json.loads((out / "contrast_report.json").read_text())["images"] == 2
    # The training and crop code find the copy exactly where they found the original
    row = {"filename": "P0001", "wound_label": "gsw", "canonical_wound_label": "gunshot wound"}
    assert find_source_image(out, row) == out / "gsw" / "P0001.JPG"


def test_output_inside_the_image_root_is_refused(tmp_path):
    (tmp_path / "raw").mkdir()
    with pytest.raises(ValueError, match="must not be"):
        main(["--image-root", str(tmp_path / "raw"), "--output-dir", str(tmp_path / "raw" / "clahe")])
