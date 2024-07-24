import numpy as np
from sklearn import metrics
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import accuracy_score


def get_purity_score(y_true, y_pred):
    """Purity score

    To compute purity, each cluster is assigned to the class which is most frequent 
    in the cluster [1], and then the accuracy of this assignment is measured by counting 
    the number of correctly assigned documents and dividing by the number of documents.
    We suppose here that the ground truth labels are integers, the same with the predicted clusters i.e
    the clusters index.

    Args:
        y_true(np.ndarray): n*1 matrix Ground truth labels
        y_pred(np.ndarray): n*1 matrix Predicted clusters

    Returns:
        float: Purity score

    References:
        [1] https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    # Labels might be missing e.g with set like 0,2 where 1 is missing
    # First find the unique labels, then map the labels to an ordered set
    # 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bin
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def get_metrics(data, preds, true_labels, model):
    silhouette_coef = silhouette_score(
        data,
        preds
    )
    ari = adjusted_rand_score(
        true_labels,
        preds
    )

    sse = model.inertia_
    try:
        confusion_matrix = metrics.cluster.contingency_matrix(
            true_labels, preds)

        FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)

        FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)

        TP = np.diag(confusion_matrix)

        TN = confusion_matrix.sum() - (FP + FN + TP)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)

        # Specificity or true negative rate
        TNR = TN/(TN+FP)

        # Precision or positive predictive value
        PPV = TP/(TP+FP)

        # Negative predictive value
        NPV = TN/(TN+FN)

        # Fall out or false positive rate
        FPR = FP/(FP+TN)

        F1 = (2*TP) / (2*TP + FP + FN)
    except:

        FP = np.zeros(2)
        FN = np.zeros(2)
        TP = np.zeros(2)
        TN = np.zeros(2)
        TPR = np.zeros(2)
        TNR = np.zeros(2)
        PPV = np.zeros(2)
        NPV = np.zeros(2)
        FPR = np.zeros(2)
        F1 = np.zeros(2)

    return {'purity_score': purity_score(true_labels,  preds),
            'purity_score2': get_purity_score(true_labels,  preds),
            'silhouette_coef': silhouette_coef,
            'ari': ari,
            'sse': sse,
            'AMI': metrics.adjusted_mutual_info_score(true_labels,  preds),
            'NMI': metrics.normalized_mutual_info_score(true_labels,  preds),
            'FP': np.mean(FP),
            'FN': np.mean(FN),
            'TP': np.mean(TP),
            'TN': np.mean(TN),
            'NPV': np.mean(NPV),
            'FPR': np.mean(FPR),
            'recall': np.mean(TPR),
            'specificity': np.mean(TNR),
            'precision': np.mean(PPV),
            'F1': np.mean(F1)}
