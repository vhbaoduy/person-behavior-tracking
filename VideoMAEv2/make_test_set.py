import os

import pandas as pd

a = "./tracking_dataset/volleyball/resized_object_video"
video_paths = []
for file in os.listdir(a):
    video_paths.append(os.path.join(a, file))

df = pd.DataFrame({"video_path": video_paths, "label": 1})
df.to_csv("./tracking_dataset/volleyball/test.csv", index=False, header=False, sep=" ")
df.to_csv("./tracking_dataset/volleyball/val.csv", index=False, header=False, sep=" ")
df.to_csv("./tracking_dataset/volleyball/train.csv", index=False, header=False, sep=" ")
