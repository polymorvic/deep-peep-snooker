from pathlib import Path
import json

import numpy as np
import pandas as pd


def _match_from_stem(stem: str) -> str:
    if stem.startswith("skip_"):
        parts = stem.split("_")
        return "_".join(parts[:2]) if len(parts) >= 2 else stem
    if stem[:2].isdigit():
        return "m" + stem[:2]
    if stem.startswith("pic_"):
        return "m_pic"
    return "m_" + stem.split("_")[0] if "_" in stem else "m_other"


def build_shot_labels_df(pic_dir: Path, skip_dir: Path) -> pd.DataFrame:
    skip_img_names = [pic.stem for pic in skip_dir.glob("*.png")]
    zero_labels = np.zeros(len(skip_img_names), dtype=np.uint8)

    img_names = [pic.stem for pic in pic_dir.glob("*.png")]
    one_labels = np.ones(len(img_names), dtype=np.uint8)

    img_df = pd.DataFrame({"img_name": img_names, "label": one_labels})
    skip_df = pd.DataFrame({"img_name": skip_img_names, "label": zero_labels})

    df = pd.concat([img_df, skip_df]).sample(frac=1).reset_index(drop=True)
    df["match"] = df["img_name"].map(_match_from_stem)

    return df


def train_val_test_split(
    df: pd.DataFrame,
    train_matches: list[str],
    val_matches: list[str],
    test_matches: list[str],
    match_col: str = "match",
    img_name_col: str = "img_name",
) -> dict[str, list[str]]:

    def _img_names_for_matches(matches: list[str]) -> list[str]:
        mask = df[match_col].isin(matches)
        return df.loc[mask, img_name_col].astype(str).tolist()

    return {
        "train": _img_names_for_matches(train_matches),
        "val": _img_names_for_matches(val_matches),
        "test": _img_names_for_matches(test_matches),
    }


def save_dataset_split(
    split_data: dict[str, list[str]], out_file_path: Path | str
) -> None:
    out_file_path = Path(out_file_path)
    with open(out_file_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=4, ensure_ascii=False)
