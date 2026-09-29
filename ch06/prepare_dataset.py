import panda as pd
df = pd.read.csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
)
print(df)