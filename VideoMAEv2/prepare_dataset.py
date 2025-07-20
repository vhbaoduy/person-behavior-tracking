import pandas as pd
import os 
mapping = {
    84: 0,
    34: 1,
    283: 2,
    342: 3,
    399: 4,
    218: 5,
    220: 6,
    235: 7,
    246: 8,
    251: 9,
}

label_names = {
    84: "dancing ballet",
    34: "breakdancing",
    283: "salsa dancing",
    342: "swing dancing",
    399: "zumba",
    218: "playing badminton",
    220: "playing basketball",
    235: "playing ice hockey",
    246: "playing tennis",
    251: "playing volleyball",
}

df = pd.read_csv('k400_list/train.csv', header=None, sep=' ')
df.columns = ['video_path', 'label']
df['video_path'] = df['video_path'].apply(lambda x: x.split('/')[-1])
df = df[df['label'].isin(mapping.keys())]
new_video_paths = []
new_labels = []
for i, row in df.iterrows():
    path = f'/data/clc_hcmus/TNhan/data/k400_resized/{label_names.get(row["label"]).replace(" ", "_")}/{row["video_path"]}'
    if os.path.exists(path):
        new_video_paths.append(path)
        new_labels.append(row["label"])
    else:
        print(f"File does not exist: {path}")
        new_video_paths.append(path)  # or append None, depending on your needs
        new_labels.append(row["label"])

df = pd.DataFrame({'video_path': new_video_paths, 'label': new_labels})
df['label'] = df['label'].map(mapping)
df.to_csv('data/all_train.csv', index=False, header=False, sep=' ')
mapping_df = pd.DataFrame(list(mapping.items()), columns=['kinetics400_label', 'selected_label'])
mapping_df['name'] = mapping_df['kinetics400_label'].map(label_names)
mapping_df.to_csv('data/maps.csv', index=False)