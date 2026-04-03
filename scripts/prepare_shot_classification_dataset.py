from pathlib import Path
import tyro
from deep_peep_snooker.dataset_preparation.classification import (
    build_shot_labels_df,
    train_val_test_split,
    save_dataset_split,
)
from deep_peep_snooker.utils.helpers import load_yaml


def run(
    playfield_shot_pics: Path,
    no_playfield_shot_pics: Path,
    config_path: Path,
    split_dataset_path: Path,
):

    df = build_shot_labels_df(playfield_shot_pics, no_playfield_shot_pics)

    matches_split = load_yaml(config_path)

    split_spec = train_val_test_split(
        df,
        train_matches=matches_split["train_matches"],
        val_matches=matches_split["val_matches"],
        test_matches=matches_split["test_matches"],
    )

    save_dataset_split(split_spec, split_dataset_path)


if __name__ == "__main__":
    tyro.cli(run)
