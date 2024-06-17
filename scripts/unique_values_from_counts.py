import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
COUNT_DIR = DATA_DIR / "counts"


def iterdir(dir):
    return [Path(file_) for file_ in dir.glob("*")]


if __name__ == "__main__":
    key_values = dict()
    for filename in iterdir(COUNT_DIR):
        df = pd.read_csv(
            filename,
            sep="\t",
            header=None,
            skiprows=1,
            names=["unique_values", "count"],
        )
        key_values[filename.stem] = df["unique_values"].to_numpy()

    mat = list()
    for key, value in key_values.items():
        key_list = [key for i in range(len(value))]
        for (
            k,
            v,
        ) in zip(key_list, value):
            mat.append([k, v])

    unique_values_df = pd.DataFrame(mat, columns=["key", "value"])

    outfile = DATA_DIR / "metadata_key_value_pairs.tsv"
    unique_values_df.to_csv(outfile, sep="\t", index=False)
