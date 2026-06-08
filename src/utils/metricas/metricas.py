
def precision_recall_f1(pred, true):
    pred = set(pred)
    true = set(true)
    comunes = pred & true
    solo_pred = pred - true
    solo_true = true - pred

    tp = len(comunes)
    fp = len(solo_pred)
    fn = len(solo_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, list(comunes), list(solo_pred), list(solo_true)
