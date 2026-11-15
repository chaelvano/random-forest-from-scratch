"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels, criterion='gini'):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.
    _, counts = np.unique(labels, return_counts=True) 
    total_count = np.sum(counts)
    probs = counts / total_count

    if criterion == 'gini':
        return 1 - np.sum(np.square(probs))
    elif criterion == 'entropy':
        return -np.sum(probs * np.log2(probs))

# Step 2 - split_dataset
def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    mask = features[:, feature_index] > threshold

    return (features[~mask], labels[~mask], features[mask], labels[mask])

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels, criterion='entropy'):
    # TODO: return a score where higher means the children are purer than the parent.    
    return impurity(parent_labels, criterion) - (len(left_labels) * impurity(left_labels, criterion) + len(right_labels) * impurity(right_labels, criterion)) / len(parent_labels)

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices, criterion='entropy'):
    # TODO: search feature_indices for the (feature, threshold) that best improves purity.
    best = {
        'feature_index': None,
        'threshold': None,
        'score': 0.0
    }
    
    for feature_index in feature_indices:
        column = features[:, feature_index]
        values = np.unique(column)

        if len(values) < 2:
            continue

        thresholds = (values[:-1] + values[1:]) / 2

        for threshold in thresholds:
            _, left_labels, _, right_labels = split_dataset(features, labels, feature_index, threshold)
            score = split_score(labels, left_labels, right_labels, criterion)

            if score > best['score']:
                best = {
                    'feature_index': feature_index,
                    'threshold': threshold,
                    'score': score
                }

    return best

# Step 5 - should_stop (not yet solved)
# TODO: implement

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

