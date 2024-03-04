import pandas as pd
import numpy as np

chunk_iter = pd.read_csv('../../dataset/CICIDS2017/CICIDS2017_payload_matched.csv', chunksize=100000)

# Initialize a counter for classes
class_counter = {}

# Iterate through the dataset in chunks
for chunk in chunk_iter:
    # Count occurrences of each class in the current chunk
    chunk_count = chunk['label'].value_counts().to_dict()

    # Update the overall counter
    for class_value, count in chunk_count.items():
        if class_value not in class_counter:
            class_counter[class_value] = count
        else:
            class_counter[class_value] += count

# Print the results
for class_value, count in class_counter.items():
    print(f'Class: {class_value}, Number of elements: {count}')