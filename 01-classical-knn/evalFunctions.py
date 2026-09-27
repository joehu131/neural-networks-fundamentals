import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    
    # Check where predictions match the truth and take the mean
    # (True = 1, False = 0, so mean gives the percentage)
    acc = np.mean(LPred == LTrue)
    
    # ============================================
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    
    # Determine matrix size based on the largest label found + 1 (assuming 0-indexed integer labels)
    n_classes = max(np.max(LPred), np.max(LTrue)) + 1
    
    # Initialize square matrix of zeros
    cM = np.zeros((n_classes, n_classes), dtype=int)
    
    # Use unbuffered add to populate matrix without a python loop
    # LPred provides row indices, LTrue provides column indices
    np.add.at(cM, (LPred, LTrue), 1)
    
    # ============================================

    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    
    # The diagonal contains the correct predictions (where Row Index == Column Index)
    # np.trace sums the diagonal elements of the confusion matrix
    correct_predictions = np.trace(cM)
    
    # The sum of the entire matrix is the total number of samples
    total_samples = np.sum(cM)
    
    acc = correct_predictions / total_samples
    
    # ============================================
    
    return acc