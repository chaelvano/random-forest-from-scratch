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
    return impurity(parent_labels, criterion) - (left_labels.shape[0] * impurity(left_labels, criterion) + right_labels.shape[0] * impurity(right_labels, criterion)) / parent_labels.shape[0]

# Step 4 - best_split
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

        if values.shape[0] < 2:
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
        feature_indices = np.arange(features.shape[1]) if feature_subset is None else feature_subset
        
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

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    # TODO: walk the example down the fitted tree until you reach a leaf, then return its prediction.
    while not tree['leaf']:
        if example[tree['feature_index']] > tree['threshold']:
            tree = tree['right']
        else:
            tree = tree['left']

    return tree['prediction']

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    # TODO: return predicted class for each row of features using the fitted tree.
    return np.array([predict_example_tree(tree, example) for example in features])

# Step 10 - bootstrap_sample
def bootstrap_sample(features, labels, rng):
    # TODO: draw a bootstrap sample of rows (with replacement) using rng.
    n = features.shape[0]
    indices = rng.integers(0, n, size=n)

    return (features[indices], labels[indices])

# Step 11 - feature_subset
def feature_subset(num_features, num_to_pick, rng):
    # TODO: return num_to_pick distinct random feature indices from range(num_features) using rng.
    return rng.choice(num_features, size=num_to_pick, replace=False)

# Step 12 - train_forest
def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, criterion='entropy', random_state=0):
    # TODO: grow num_trees decision trees on bootstrap samples with random feature subsets.
    forest = []
    rng = np.random.default_rng(random_state)
    d = len(features[0])
    
    if num_features_per_split is None:
        num_features_per_split = np.maximum(1, int(np.round(np.sqrt(d))))

    for _ in range(num_trees):
        bootstrap_features, bootstrap_labels = bootstrap_sample(features, labels, rng)
        feature_indices = feature_subset(d, num_features_per_split, rng)

        forest.append({
            'tree': build_tree(bootstrap_features, bootstrap_labels, max_depth=max_depth, min_samples_split=min_samples_split, feature_subset=feature_indices, criterion=criterion),
            'feature_indices': feature_indices
        })

    return forest

# Step 13 - combine_predictions
def combine_predictions(tree_predictions):
    # TODO: aggregate the per-tree predictions of an ensemble into one prediction per example.
    return np.array([np.bincount(tree_predictions[:, col]).argmax() for col in range(len(tree_predictions[0]))])

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

