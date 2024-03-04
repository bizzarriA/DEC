import torch
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.tree import _tree


import matplotlib

matplotlib.rcParams.update({'font.size': 16})

from scipy.optimize import linear_sum_assignment

def map_clusters_to_labels(true_labels, predicted_clusters):
    # use linear assignment to map cluster in label
    unique_true_labels = np.unique(true_labels)
    unique_predicted_clusters = np.unique(true_labels)

    cost_matrix = confusion_matrix(true_labels, predicted_clusters)
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    mapping = {unique_predicted_clusters[j]: unique_true_labels[i] for i, j in zip(row_ind, col_ind)}

    mapped_labels = np.vectorize(mapping.get)(predicted_clusters)
    return mapped_labels



def eval_DEC(name, OOD_ds, out_dir, test_ds=None):
    # evaluate DEC and print {name}_test_result.csv - it is input file for next step (ML step)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"{name} test OOD ad ID")
    print()
    print('*' * 50)
    print('load the best DEC ...')
    dec = torch.load(f'{out_dir}/DEC.pt', device)
    print('Evaluate ...')
    print('*' * 50)
    dec.eval()
    feature, y, ood_id, pred, ids = [], [], [], [], []
    with torch.no_grad():
        for batch in OOD_ds:
            data, labels, id = batch
            data = data.to(device)
            x = dec.encoder(data)
            feature.append(x)
            labels = labels.to(device)
            ids.append(id)
            ood_id.append(torch.full(labels.shape, 1))
            y.append(labels)
        if test_ds is not None:
            for batch in test_ds:
                data, labels, id = batch
                data = data.to(device)
                x = dec.encoder(data)
                labels = labels.to(device)
                feature.append(x)
                ood_id.append(torch.full(labels.shape, 0))
                y.append(labels)
                ids.append(id)
        feature = torch.cat(feature)
        y = torch.cat(y).cpu()
        ids = torch.cat(ids).cpu()
        ood_id = torch.cat(ood_id).cpu()
        pred = dec.cluster(feature)[:, 1].cpu().numpy()
        feature = feature.cpu()

        print('save csv ...')
        feature_names = [f"f_{i}" for i in range(feature.shape[1])]
        df = pd.DataFrame(feature, columns=feature_names)
        df['real_label'] = y
        df['pred'] = pred
        df['OOD'] = ood_id
        df['unique_id'] = ids
        print(np.unique(df.real_label, return_counts=True))
        df.to_csv(out_dir + f'/{name}_result_test.csv', index=False)


def get_rule(tree, feature_names, class_names, n_sample=1000):
    # funtion to get rule from DT
    tree_ = tree  # .tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        if path[-1][1] > n_sample:
            rule = "if "
            for p in path[:-1]:
                if rule != "if ":
                    rule += " and "
                rule += str(p)
            rule += " then "
            if class_names is None:
                rule += "response: " + str(np.round(path[-1][0][0][0], 3))
            else:
                classes = path[-1][0][0]
                l = np.argmax(classes)
                rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
            rule += f" | based on {path[-1][1]:,} samples"
            rules += [rule]

    print("*" * 50)
    print(f"RULE for {class_names}:\n")
    for r in rules:
        print(r)
    print("*" * 50)

    return rules
