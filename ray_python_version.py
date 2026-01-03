#!/usr/bin/env python

import pandas as pd
from packaging.version import Version

os_map = {"linux": 1, "macos": 2, "windows": 3}
versions = None
for os, table_num in os_map.items():
    df = pd.read_html("https://docs.ray.io/en/latest/ray-overview/installation.html")[
        table_num
    ]
    mask = df.apply(lambda col: col.astype(str).str.contains("beta")).any(axis=1)
    df = df.loc[~mask]
    for col in df.columns:
        if versions is None:
            versions = set(
                df[col].astype(str).str.extract(r"(\d+\.\d+)").values.flatten().tolist()
            )
        else:
            versions.intersection(
                (
                    set(
                        df[col]
                        .astype(str)
                        .str.extract(r"(\d+\.\d+)")
                        .values.flatten()
                        .tolist()
                    )
                )
            )
versions = list(versions)

versions.sort(key=Version)
print(versions[-1])
