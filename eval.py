from src import config
import sys
import os
import torch
import cv2
import numpy
from loguru import logger

# load our object detector, set it evaluation mode, and label

if len(sys.argv) < 2:
    logger.debug("Please enter the path to the model to be evaluated")
    sys.exit(1)

model_path = sys.argv[1]

logger.debug(f"**** loading object detector at {model_path}...")
model = torch.load(model_path, weights_only=False).to(config.DEVICE)
model.eval()
logger.debug(f"**** object detector loaded")

results_labels = dict()

def dist(bbox_1, bbox_2, h, w) -> float:
    start_x_1, start_y_1, end_x_1, end_y_1 = bbox_1
    start_x_2, start_y_2, end_x_2, end_y_2 = bbox_2 
    return numpy.sqrt(((start_x_1 - start_x_2)/w) ** 2 + ((start_y_1 - start_y_2)/h) ** 2 + ((end_x_1 - end_x_2)/w) ** 2 + ((end_y_1 - end_y_2)/h) ** 2)

def calculate_iou(bbox_pred, bbox_gt):
    """Calculate Intersection over Union (IoU) between predicted and ground truth bounding boxes"""
    x1_pred, y1_pred, x2_pred, y2_pred = bbox_pred
    x1_gt, y1_gt, x2_gt, y2_gt = bbox_gt
    
    # Calculate intersection coordinates
    x1_int = max(x1_pred, x1_gt)
    y1_int = max(y1_pred, y1_gt)
    x2_int = min(x2_pred, x2_gt)
    y2_int = min(y2_pred, y2_gt)
    
    # Calculate intersection area
    if x2_int <= x1_int or y2_int <= y1_int:
        intersection = 0.0
    else:
        intersection = (x2_int - x1_int) * (y2_int - y1_int)
    
    # Calculate areas of both bounding boxes
    area_pred = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    area_gt = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    
    # Calculate union
    union = area_pred + area_gt - intersection
    
    # Calculate IoU
    if union == 0:
        return 0.0
    return intersection / union


for mode, csv_file in [['train', config.TRAIN_PATH],
                       ['validation', config.VAL_PATH],
                       ['test', config.TEST_PATH],]:
    data = []
    assert(csv_file.endswith('.csv'))

    logger.debug(f"Evaluating {mode} set...")
    # loop over CSV file rows (filename, startX, startY, endX, endY, label)
    for row in open(csv_file).read().strip().split("\n"):
        filename, startX, startY, endX, endY, label = row.split(',')
        filename = os.path.join(config.IMAGES_PATH, label, filename)
        data.append((filename, startX, startY, endX, endY, label))

    logger.debug(f"Evaluating {len(data)} samples...")

    # Store all results as well as per class results
    results_labels[mode] = dict()
    results_labels[mode]['all'] = []
    results_labels[mode]['bbox'] = {}
    results_labels[mode]['bbox']['all'] = []
    results_labels[mode]['iou'] = {}
    results_labels[mode]['iou']['all'] = []
    for label_str in config.LABELS:
        results_labels[mode][label_str] = []
        results_labels[mode]['bbox'][label_str] = []
        results_labels[mode]['iou'][label_str] = []

    # loop over the images that we'll be testing using our bounding box
    # regression model
    for filename, gt_start_x, gt_start_y, gt_end_x, gt_end_y, gt_label in data:
        # load the image, copy it, swap its colors channels, resize it, and
        # bring its channel dimension forward
        image = cv2.imread(filename)
        
        # Check if image was loaded successfully
        if image is None:
            logger.warning(f"Could not load image: {filename}. Skipping...")
            continue
            
        display = image.copy()
        h, w = display.shape[:2]

        # convert image to PyTorch tensor, normalize it, upload it to the
        # current device, and add a batch dimension
        image = config.TRANSFORMS(image).to(config.DEVICE)
        image = image.unsqueeze(0)

        # predict the bounding box of the object along with the class label
        label_predictions, bbox_pred_norm = model(image)
        # Handle tuple output (classification, bounding_box)

        # determine the class label with the largest predicted probability
        label_predictions = torch.nn.Softmax(dim=-1)(label_predictions)
        most_likely_label = label_predictions.argmax(dim=-1).cpu()
        label = config.LABELS[most_likely_label]

        startX_norm, startY_norm, endX_norm, endY_norm = bbox_pred_norm[0].detach().cpu().numpy()
        bbox_pred = (int(startX_norm * w), int(startY_norm * h), int(endX_norm * w), int(endY_norm * h))
        bbox_gt = (int(gt_start_x), int(gt_start_y), int(gt_end_x), int(gt_end_y))

        # Compare to gt data
        results_labels[mode]['all'].append(label == gt_label)
        results_labels[mode][gt_label].append(label == gt_label)

        # Compute bounding box metrics
        dist_bbox = dist(bbox_pred, bbox_gt, h, w)
        iou_score = calculate_iou(bbox_pred, bbox_gt)
        
        results_labels[mode]['bbox']['all'].append(dist_bbox)
        results_labels[mode]['bbox'][gt_label].append(dist_bbox)
        results_labels[mode]['iou']['all'].append(iou_score)
        results_labels[mode]['iou'][gt_label].append(iou_score)

        if label != gt_label:
            logger.debug(f"\tFailure at {filename}")


# Compute per dataset accuracy
for mode in ['train', 'validation', 'test']:
    logger.info(f'*** {mode} set accuracy')
    logger.info(f"Mean accuracy for all labels: "
          f"{numpy.mean(numpy.array(results_labels[mode]['all'])):.4f}")
    
    # Display bounding box metrics
    bbox_distances = numpy.array(results_labels[mode]['bbox']['all'])
    iou_scores = numpy.array(results_labels[mode]['iou']['all'])
    
    logger.info(f"Bounding Box Metrics:")
    logger.info(f"\t - Mean normalized distance: {numpy.mean(bbox_distances):.4f}")
    logger.info(f"\t - Mean IoU: {numpy.mean(iou_scores):.4f}")
    logger.info(f"\t - Median IoU: {numpy.median(iou_scores):.4f}")
    logger.info(f"\t - IoU > 0.5 (good detection): {numpy.sum(iou_scores > 0.5)} / {len(iou_scores)} ({100*numpy.mean(iou_scores > 0.5):.1f}%)")
    logger.info(f"\t - IoU > 0.7 (very good detection): {numpy.sum(iou_scores > 0.7)} / {len(iou_scores)} ({100*numpy.mean(iou_scores > 0.7):.1f}%)")

    for label_str in config.LABELS:
        logger.debug(f'\tMean accuracy for label {label_str}: '
              f'{numpy.mean(numpy.array(results_labels[mode][label_str])):.4f}')
        logger.debug(f'\t\t {numpy.sum(results_labels[mode][label_str])} over '
              f'{len(results_labels[mode][label_str])} samples')
        
        # Per-label bounding box metrics
        label_bbox_dist = numpy.array(results_labels[mode]['bbox'][label_str])
        label_iou = numpy.array(results_labels[mode]['iou'][label_str])
        
        logger.debug(f'\t\t BBox:')
        logger.debug(f'\t\t      - Mean distance: {numpy.mean(label_bbox_dist):.4f}')
        logger.debug(f'\t\t      - Mean IoU: {numpy.mean(label_iou):.4f}')
        logger.debug(f'\t\t BBox:')
        logger.debug(f'\t\t      - IoU > 0.5: {100*numpy.mean(label_iou > 0.5):.1f}%')
        logger.debug(f'\t\t      - IoU > 0.7: {100*numpy.mean(label_iou > 0.7):.1f}%')

