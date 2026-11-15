#!/usr/bin/env python3
"""
Annual numbers of social-media records and users until 2021
===========================================================

    python Python_A1b_social_media_counts_until_2021.py

Draws Figures S1-S3 (Flickr, Twitter, VK: records and unique users per year,
inside and outside the pre-2022 occupation line) and counts the first rows of
Table S1, using only records dated from START_YEAR to END_YEAR. Records dated
after END_YEAR and records without a date enter no figure and no count; the
script only prints how many of them the archive holds.

Inputs (working files of Python_A1, in the same folder):
    social_merged_cov_3857.gpkg   all records with pseudonymised user_id, the
                                  platform ('layer'), year, month and the ESA
                                  WorldCover class ('esa')
    ordlo.gpkg                    area occupied before 24 February 2022

Outputs (folder Supplementary_figures):
    FigureS1_Flickr_until_2021.pdf / .png
    FigureS2_Twitter_until_2021.pdf / .png
    FigureS3_VK_until_2021.pdf / .png
    social_media_counts_until_2021.csv   platform x year x side: records, users
    TableS1_first_rows_until_2021.csv    counts for the first rows of Table S1

Other folders:
    python Python_A1b_social_media_counts_until_2021.py --data-dir <folder> --out-dir <folder>
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

START_YEAR = 2006
END_YEAR = 2021                                  # last year shown and counted
TARGET_ESA_CLASSES = [10, 20, 30, 80, 90, 100]   # as in Python_A1
SUMMER_MONTHS = [5, 6, 7, 8, 9]                  # as in Python_A1
PRESENCE_START_YEAR = 2015                       # as in Python_A1
PLATFORMS = [("flic", "Flickr", "FigureS1"), ("twitt", "Twitter", "FigureS2"), ("vk", "VK", "FigureS3")]
COLOURS = {"yes": "#d62728", "no": "#1f77b4"}    # the same colours in both panels

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "support"))
try:
    import folders
    DEFAULT_DATA = Path(getattr(folders, "DOWNLOADS", HERE))
    DEFAULT_OUT = Path(getattr(folders, "FIGURES", HERE / "Supplementary_figures"))
except Exception:                                # folders.py not found
    DEFAULT_DATA, DEFAULT_OUT = HERE, HERE / "Supplementary_figures"


def read_records(path):
    gdf = gpd.read_file(path)
    if "year" not in gdf.columns or "month" not in gdf.columns:
        date = pd.to_datetime(gdf["datetaken"], errors="coerce")
        gdf["year"], gdf["month"] = date.dt.year, date.dt.month
    return gdf[["user_id", "layer", "year", "month", "esa", "geometry"]]


def tag_occupation_line(gdf, ordlo_path):
    ordlo = gpd.read_file(ordlo_path)[["geometry"]].to_crs(gdf.crs)
    joined = gpd.sjoin(gdf, ordlo, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    gdf = gdf.copy()
    gdf["ordlo"] = joined["index_right"].notna().map({True: "yes", False: "no"}).values
    return gdf


def annual_counts(gdf):
    stats = (gdf.groupby(["layer", "year", "ordlo"])
             .agg(total_rows=("user_id", "size"), unique_users=("user_id", "nunique"))
             .reset_index())
    stats["year"] = stats["year"].astype(int)
    return stats.sort_values(["layer", "year", "ordlo"])


def draw(stats, name, out_stem):
    years = list(range(int(stats["year"].min()), END_YEAR + 1))
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    panels = (("total_rows", "Total {0} content per year", "Number of posts", ""),
              ("unique_users", "Unique {0} users per year", "Number of unique users", "Year"))
    for ax, (column, title, ylabel, xlabel) in zip(axes, panels):
        sns.barplot(data=stats, x="year", y=column, hue="ordlo", order=years,
                    hue_order=["no", "yes"], palette=COLOURS, ax=ax)
        ax.set_title(title.format(name), fontsize=14)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.legend(title="Inside ORDLO")
    plt.tight_layout()
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="folder with the working files of Python_A1")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="folder for the figures and tables")
    a = ap.parse_args()
    records_path = a.data_dir / "social_merged_cov_3857.gpkg"
    ordlo_path = a.data_dir / "ordlo.gpkg"
    if not ordlo_path.exists() and hasattr(folders, "DATA"):
        ordlo_path = Path(folders.DATA) / "ordlo.gpkg"     # where Python_A1 reads it
    for p in (records_path, ordlo_path):
        if not p.exists():
            raise SystemExit("Missing input: {0}".format(p))
    a.out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading {0} ...".format(records_path))
    gdf = read_records(records_path)
    n_all = len(gdf)
    n_no_date = int(gdf["year"].isna().sum())
    n_after = int((gdf["year"] > END_YEAR).sum())
    n_before = int((gdf["year"] < START_YEAR).sum())
    gdf = gdf[(gdf["year"] >= START_YEAR) & (gdf["year"] <= END_YEAR)].copy()
    gdf["year"] = gdf["year"].astype(int)

    in_classes = gdf["esa"].isin(TARGET_ESA_CLASSES)
    in_window = gdf["month"].isin(SUMMER_MONTHS) & (gdf["year"] >= PRESENCE_START_YEAR)
    rows = [
        ("All georeferenced public records, {0}-{1}".format(START_YEAR, END_YEAR), len(gdf)),
        ("In one of the six natural and water ESA WorldCover classes", int(in_classes.sum())),
        ("Taken in May-September of {0}-{1} (check: 165,417 in Table S1)".format(PRESENCE_START_YEAR, END_YEAR),
         int((in_classes & in_window).sum())),
    ]
    pd.DataFrame(rows, columns=["step", "records"]).to_csv(
        a.out_dir / "TableS1_first_rows_until_{0}.csv".format(END_YEAR), index=False)

    gdf = tag_occupation_line(gdf, ordlo_path)
    stats = annual_counts(gdf)
    stats.to_csv(a.out_dir / "social_media_counts_until_{0}.csv".format(END_YEAR), index=False)
    for code, name, figure in PLATFORMS:
        part = stats[stats["layer"] == code]
        if part.empty:
            print("No records for {0}.".format(name))
            continue
        draw(part, name, a.out_dir / "{0}_{1}_until_{2}".format(figure, name, END_YEAR))

    print("\nRecords in the archive:            {0:>10,}".format(n_all))
    print("  dated after {0} (not used):      {1:>10,}".format(END_YEAR, n_after))
    print("  without a date (not used):       {0:>10,}".format(n_no_date))
    print("  dated before {0} (not used):     {1:>10,}".format(START_YEAR, n_before))
    print("\nTable S1, first rows:")
    for step, n in rows:
        print("  {0:<72}{1:>10,}".format(step, n))
    print("\nRecords and users by platform, {0}-{1}:".format(START_YEAR, END_YEAR))
    by_platform = gdf.groupby("layer").agg(records=("user_id", "size"), users=("user_id", "nunique"),
                                           first_year=("year", "min"), last_year=("year", "max"))
    print(by_platform.to_string())
    print("\nFigures and tables written to {0}".format(a.out_dir))


if __name__ == "__main__":
    main()
