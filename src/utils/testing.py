from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def prepare_test_results_report(
        dir: str | Path,
        data: list[dict],
        filename: str,
        agg_colname: str,
        not_found_pic_names: list[str],
        sheet_name1: str = 'results', 
        sheet_name2: str = 'stats') -> pd.DataFrame | None:
    
    if not data:
        raise ValueError('Input data is empty!')

    with pd.ExcelWriter(Path(dir) / f'{filename}.xlsx', engine='openpyxl') as writer:
        results_df = pd.DataFrame(data)
        results_df.to_excel(writer, sheet_name=sheet_name1, index=False)

        stats_df = (
            results_df[agg_colname]
            .agg(['median', 'mean', 'std', 'min', 'max'])
            .to_frame()
            .T
        )
        stats_df['not_found'] = len(not_found_pic_names)
        stats_df.to_excel(writer, sheet_name=sheet_name2)


def save_test_histogram(
    dir: str | Path,
    df: pd.DataFrame,
    colname: str,
    filename: str
    ) -> None:

    if colname not in df.columns:
        raise ValueError(f"Column '{colname}' not found in DataFrame.")

    series = df[colname].dropna()

    if series.empty:
        raise ValueError(f"Column '{colname}' is empty.")

    min_val = series.min()
    max_val = series.max()

    bins = np.arange(min_val, max_val + 1, 1)
    output_path = Path(dir) / f"{filename}.png"

    plt.figure()
    plt.hist(series, bins=bins, edgecolor="black")
    plt.xlabel(colname)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {colname}")
    plt.savefig(output_path)
    plt.close()