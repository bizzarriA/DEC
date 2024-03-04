import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# Read the dataset in chunks
file_path = '../../dataset/CICIDS2017/CICIDS2017_payload_matched.csv'
classes = ['BENIGN',
           'Bot',
           'DDoS',  # DT train
           'DoS GoldenEye',  # DT train
           'DoS Hulk',
           'DoS Slowhttptest',
           'DoS slowloris',  # DT train
           'FTP-Patator',  # ood
           'Heartbleed',
           'Infiltration',  # ood
           'PortScan',
           'SSH-Patator',  # ood
           'Web Attack – Brute Force',
           'Web Attack – Sql Injection',
           'Web Attack – XSS']

chunk_size = 100000  # You can adjust the chunk size based on your system's capacity

# Load dataset in chunks
df_chunks = pd.read_csv(file_path, chunksize=chunk_size)
dict = {
    "BENIGN": 134165,
    "DoS Hulk": 41283,
    "DDoS": 618544,
    "SSH-Patator": 181147,
    "FTP-Patator": 110636,
    "Infiltration": 41725,
    "Heartbleed": 41283,
    "DoS GoldenEye": 34293,
    "Web Attack – Brute Force": 28920,
    "DoS slowloris": 20877,
    "DoS Slowhttptest": 9778,
    "Web Attack – XSS": 6767,
    "Bot": 5143,
    "PortScan": 946,
    "Web Attack – Sql Injection": 45,
}
ood_classes = [11, 9, 7, 6, 3, 2]
print('Random Sampler:')

# Initialize an empty DataFrame to store the sampled chunks
df_reduced = pd.DataFrame()
n_chunks = 6647756//chunk_size + 1
for i, chunk in enumerate(df_chunks):
    print(f"{i + 1}/{n_chunks}", end="\r")

    # Get unique labels and their counts
    labels, counts = np.unique(chunk['label'], return_counts=True)

    # Subsample "BENIGN" class
    if 'BENIGN' in labels:
        oversample_ratio_b = 1-0.4
        b_index = np.where(labels == 'BENIGN')[0][0]
        subset_indices_b = np.random.choice(chunk.index[chunk['label'] == 'BENIGN'],
                                              size=int(oversample_ratio_b * counts[b_index]), replace=False)
        chunk = chunk.drop(subset_indices_b)
    # Subsample "DoS Hulk" class
    if 'DoS Hulk' in labels:
        oversample_ratio_dos = 1-0.19
        dos_index = np.where(labels == 'DoS Hulk')[0][0]
        subset_indices_dos = np.random.choice(chunk.index[chunk['label'] == 'DoS Hulk'],
                                              size=int(oversample_ratio_dos * counts[dos_index]), replace=False)
        chunk = chunk.drop(subset_indices_dos)
    # Concatenate the current chunk to df_reduced
    df_reduced = pd.concat([df_reduced, chunk])

print(np.unique(df_reduced['label'], return_counts=True))
print("Split OOD vs ID")
df_reduced['label'] = df_reduced['label'].apply(lambda x: classes.index(x))
ood_data = df_reduced[df_reduced['label'].isin(ood_classes)]
id_df = df_reduced[~df_reduced['label'].isin(ood_classes)]
del df_reduced

print("Split Train-Val-Test")
# Split into training, validation, test
train_data, test_data = train_test_split(id_df, test_size=0.3, random_state=42, stratify=id_df.label)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data.label)

print("Save DataFrames")
train_data.to_csv('../../dataset/CICIDS2017/payload_train.csv')
test_data.to_csv('../../dataset/CICIDS2017/payload_test.csv')
val_data.to_csv('../../dataset/CICIDS2017/payload_val.csv')
ood_data.to_csv('../../dataset/CICIDS2017/zero_day.csv')
