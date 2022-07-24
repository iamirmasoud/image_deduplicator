import logging
import os

import pandas as pd
from sklearn.cluster import DBSCAN

from configs import delete_files, epsilon

logger = logging.getLogger(__name__)
logging.basicConfig(
    level="INFO",
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

features = pd.read_pickle("features.pkl")

model = DBSCAN(min_samples=1, eps=epsilon)

clusters = pd.DataFrame(
    {"path": features.index, "label": model.fit_predict(features)}
).sort_values(["label", "path"], ascending=False)

# select one of elements from each cluster to keep
list_to_keep = set(clusters.groupby("label")["path"].first())
all_files = set(clusters["path"])

files_to_remove = all_files - list_to_keep
logger.info(f"Found {len(files_to_remove)} duplicate files:\n {files_to_remove}")

for group, paths in clusters[clusters["label"].duplicated(keep=False)].groupby("label")[
    "path"
]:
    print(f'Duplicate items for "{paths.iloc[0]}" are:\n {paths.iloc[1:].values}\n')


if delete_files:
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
        except OSError as e:
            logger.exception("Cannot delete file '%s': %s" % (file_path, e.strerror))
    logger.info(f"Successfully deleted {len(files_to_remove)} duplicate files.")
