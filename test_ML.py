import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from eval_ML import evalXG, eval_LoOP
from xgboost import XGBClassifier
import argparse
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

def get_arg():
    # Function to parse command-line arguments using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, type=str,
                        help='Name of experiments, must be the same as experiments sub dir')
    parser.add_argument('--save_dir', default='results/', help='Location where model and assets will be saved')
    parser.add_argument('--rules', type=bool, default=False, help="True if you want to extract rules from DT")
    args = parser.parse_args()
    return args


def preprocess_data(df, remove, UNSW=None):
    # Function to preprocess the data, including label conversion, splitting, and combining datasets
    if UNSW is not None:
        df_c = UNSW
        df_train = df
    else:
        df_c = df.loc[df['real_label'].isin(remove)]
        df_train = df.loc[~df['real_label'].isin(remove)]
    # Convert labels to binary in df_train and df_c
    # Convert labels to binary in df_train and df_c using loc
    df_train.loc[:, 'real_label'] = df_train['real_label'].apply(lambda label: 0 if label == 0 else 1)
    df_c.loc[:, 'real_label'] = df_c['real_label'].apply(lambda label: 0 if label == 0 else 1)
    df_c = df_c.sample(frac=0.01)
    X = df_train[feature_name]
    y = df_train[target_name]

    # Split data into train and test and add zero days to the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    X_test = pd.concat([X_test, df_c[feature_name]])
    y_test = pd.concat([y_test, df_c[target_name]])

    X_c = pd.concat([df_c[feature_name], X_test[y_test['OOD'] == 0].sample(n=len(df_c))])
    y_c = pd.concat([df_c[target_name], y_test[y_test['OOD'] == 0].sample(n=len(df_c))])

    return X_train, y_train, X_c, y_c, X_test, y_test


def train_decision_tree(X_train, y_train, X_c, y_c, X_test, y_test):
    # Function to train a Decision Tree classifier and evaluate on test sets

    # Define multi-output DT classifier
    tree_classifier = DecisionTreeClassifier()
    tree_classifier.fit(X_train, y_train)

    # Print results
    print('*' * 50)
    print('OOD all cicids2017 test set:')
    evalXG(tree_classifier, X_test, y_test, name=exp_dir + "/DT_zero_days_")
    print('*' * 50)
    print('OOD zero days test set:')
    evalXG(tree_classifier, X_c, y_c, name=exp_dir + "/DT_test_set_")


def train_xgboost(X_train, y_train, X_c, y_c, X_test, y_test):
    # Function to train an XGBoost classifier and evaluate on test sets

    bst_ood = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
    bst_ood.fit(X_train, y_train)

    print('*' * 50)
    print('OOD all cicids2017 test set:')
    evalXG(bst_ood, X_test, y_test, threshold=0.5, name=exp_dir + "/XG_test_set_")
    print('*' * 50)
    print('OOD zero days test set:')
    evalXG(bst_ood, X_c, y_c, threshold=0.5, name=exp_dir + "/XG_zero_days_")


def calculate_intra_cluster_distances(point, X, labels, cluster_label):
    # Helper function to calculate distances within a cluster for LoOP

    # find all index of same cluster points
    cluster_indices = np.where(np.array(labels) == cluster_label)[0]

    # Compute distance between point and other points from same cluster
    distances = np.linalg.norm(np.array(X)[cluster_indices] - np.array(point), axis=1)

    return distances


def calculate_lof(X, labels, k, threshold):
    ### WORK IN PROGRESS FUNZION!
    # Function to calculate Local Outlier Factor (LOF) using DBSCAN for clustering

    clustering = DBSCAN(eps=0.5, min_samples=k)
    labels = clustering.fit_predict(X)

    # neigh = NearestNeighbors(n_neighbors=k)
    # neigh.fit(X)
    # distances, _ = neigh.kneighbors(X)

    lof_values = []

    for i, (_, x) in enumerate(X.iterrows()):
        distances = calculate_intra_cluster_distances(x, X, labels, labels[i])
        avg_distance = np.mean(distances)
        lof = 1.0 / (avg_distance ** k)
        lof_values.append(lof)

    lof_values = np.array(lof_values)
    lof_values = (lof_values - np.min(lof_values)) / (np.max(lof_values) - np.min(lof_values))

    outliers = X[lof_values > threshold]

    return lof_values, outliers


if __name__ == "__main__":
    # Main section of the script
    arg = get_arg()
    exp_dir = os.path.join(arg.save_dir, arg.name)

    # Various flags to control which parts of the code to execute
    DT = False
    xg = False
    do_UNSW = False
    add_Netflow = False
    LoOP = True

    # Read data and define feature and target names
    df_payload = pd.read_csv(os.path.join(exp_dir, f'{arg.name}_result_test.csv'))
    feature_name = ['f_0', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_7', 'f_8', 'f_9', 'f_10', 'f_11']  # , 'pred']
    target_name = ['real_label', 'OOD', 'pred']
    label_classification = ['OOD']
    if add_Netflow:
        # if you want add NetFlow information
        df_netflow = pd.read_csv('../../dataset/CICIDS2017/CICIDS2017_netflow_matched.csv')
        netflow_column = ['sport', 'dsport', 'protocol_m',
                      'stime_flow', 'duration', 'Total Fwd Packets', 'Total Backward Packets',
                      'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                      'Fwd Packet Length Max', 'Fwd Packet Length Min',
                      'Fwd Packet Length Mean', 'Fwd Packet Length Std',
                      'Bwd Packet Length Max', 'Bwd Packet Length Min',
                      'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
                      'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
                      'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
                      'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
                      'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
                      'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
                      'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
                      'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
                      'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
                      'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
                      'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
                      'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
                      'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
                      'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
                      'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
                      'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
                      'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
                      'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
                      'Idle Std', 'Idle Max', 'Idle Min', 'unique_id']
        # feature_name += netflow_column
        # merge two dataframe using 'unique_id'
        df = pd.merge(df_payload, df_netflow, on='unique_id', how='left')
        # Encode categorical column
        label_encoder = LabelEncoder()
        column_type = df_netflow.dtypes
        str_column = column_type[column_type == 'object'].index
        for col in str_column:
            df[col] = label_encoder.fit_transform(df[col])
    else:
        df = df_payload

    classes = ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris',
               'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator', 'Web Attack – Brute Force',
               'Web Attack – Sql Injection', 'Web Attack – XSS']

    # Classes to remove
    remove = [11, 9, 7]
    # remove = [14, 13, 12, 10, 8, 1]

    for attack in remove:
        # Loop over attacks to evaluate DT, XGBoost and LoOP
        print('*' * 50)
        print(f'Prediction attack {classes[attack]}:')

        X_train, y_train, X_c, y_c, X_test, y_test = preprocess_data(df, remove)

        if DT:
            print('*' * 50)
            print('Decision Tree:')
            train_decision_tree(X_train, y_train[label_classification], X_c, y_c[label_classification], X_test, y_test[label_classification])

        if xg:
            print('*' * 50)
            print('XGboost:')
            train_xgboost(X_train, y_train[label_classification], X_c, y_c[label_classification], X_test, y_test[label_classification])

        if LoOP:
            # Set parameters
            k_neighbors = 5
            outlier_threshold = 0.9

            # Run LoOP algorithm
            lof_values, outliers = calculate_lof(X_c, y_c['pred'], k_neighbors, outlier_threshold)
            eval_LoOP(y_c['OOD'], lof_values, threshold=outlier_threshold, name=f'LoOP_{classes[attack]}')

    if do_UNSW:
        # Optionally, preprocess and evaluate on additional UNSW dataset
        df_UNSW = pd.read_csv(os.path.join(exp_dir, 'UNSW_result_test.csv'))

        X_train, y_train, X_c, y_c, _, _ = preprocess_data(df, remove, df_UNSW)
        if DT:
            print('*' * 50)
            print('Decision Tree:')
            train_decision_tree(X_train, y_train[label_classification], X_c, y_c[label_classification])

        if xg:
            print('*' * 50)
            print('XGboost:')
            train_xgboost(X_train, y_train[label_classification], X_c, y_c[label_classification])
