

def calculate_iou_1d(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """
    Calculate the IoU (Intersection over Union) between two moments.
    Args:
        pred_start (float): Start time of the predicted moment.
        pred_end (float): End time of the predicted moment.
        gt_start (float): Start time of the ground truth moment.
        gt_end (float): End time of the ground truth moment.
    Returns:
        IoU (float): IoU between the two moments.
    """
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    return intersection / float(union) if union > 0 else 0

def calculate_recall_rate(ground_truth_list: list, recall_list: list) -> float:
    """
    Calculate the recall rate between the ground truth list and the recall list.
    This function is used for multiple ground truth scenarios.
    Args:
        ground_truth_list (list): List of ground truth items (for one sample).
        recall_list (list): List of items recalled by the model (for one sample).
        Here we allow any list containing the same elements.
    Returns:
        recall_rate (float): Recall rate between the ground truth list and the recall
    """
    # Convert lists to sets to perform set operations
    ground_truth_set = set(ground_truth_list)
    recall_set = set(recall_list)
    assert len(ground_truth_set) > 0, "Ground truth list cannot be empty"
    
    # Calculate the number of true positives (items in recall list that are in ground truth)
    true_positives = len(ground_truth_set.intersection(recall_set))
    
    # Calculate recall rate
    recall_rate = true_positives / len(ground_truth_set)
    
    return recall_rate
