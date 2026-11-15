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
            left_features, left_labels, right_features, right_labels = split_dataset(features, labels, feature_index, threshold)
            score = split_score(labels, left_labels, right_labels, criterion)

            if score > best['score']:
                best = {
                    'feature_index': feature_index,
                    'threshold': threshold,
                    'score': score,
                    'left_features': left_features,
                    'left_labels': left_labels,
                    'right_features': right_features,
                    'right_labels': right_labels
                }

    return best

# Step 5 - should_stop
def should_stop(labels, depth, max_depth=10, min_samples_split=2):
    """Return True if this node should become a leaf instead of splitting further."""
    # TODO: decide whether to stop growing based on purity, depth, and size...
    return np.unique(labels).shape[0] == 1 or depth >= max_depth or labels.shape[0] < min_samples_split

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    # TODO: choose a single class label to output for a leaf given the labels that reached it
    values, counts = np.unique(labels, return_counts=True)

    return int(values[np.argmax(counts)])

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0, criterion='entropy'):
    # TODO: recursively grow a decision tree, returning a nested dict of leaf/internal nodes.
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {
            'leaf': True,
            'prediction': leaf_prediction(labels)
        }
    else:
        feature_indices = np.arange(len(features[0])) if feature_subset is None else feature_subset
        
        best = best_split(features, labels, feature_indices, criterion)

        if best['feature_index'] is None:
            return {
                'leaf': True,
                'prediction': leaf_prediction(labels)
            }

        feature_index, threshold, left_features, left_labels, right_features, right_labels = best['feature_index'], best['threshold'], best['left_features'], best['left_labels'], best['right_features'], best['right_labels']

        return {
            'leaf': False,
            'feature_index': feature_index,
            'threshold': threshold,
            'left': build_tree(left_features, left_labels, max_depth, min_samples_split, feature_subset, depth + 1, criterion),
            'right': build_tree(right_features, right_labels, max_depth, min_samples_split, feature_subset, depth + 1, criterion)
        }

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

