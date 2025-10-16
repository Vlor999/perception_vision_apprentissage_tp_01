from src import config
import sys
import os
import torch
import cv2
from loguru import logger
import torch.nn as nn
from src.arguments.handle_args import setup_args_prediction

# load our object detector, set it evaluation mode, and label
# encoder from disk
def load_model(filename:str, device) -> nn.Module:
    logger.debug("**** loading object detector...")
    model = torch.load(filename, weights_only=False).to(device)
    logger.debug(f"Using model: {filename}")
    model.eval()
    return model


def load_data(filename: str) -> list[tuple[str, int, int, int, int, str]]:
    data = []
    path = filename
    if path.endswith('.csv'):
        # loop over CSV file rows (filename, startX, startY, endX, endY, label)
        for row in open(path).read().strip().split("\n"):
            filename, startX, startY, endX, endY, label = row.split(',')
            filename = os.path.join(config.IMAGES_PATH, label, filename)
            startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
            data.append((filename, startX, startY, endX, endY, label))
    else:
        data.append((path, None, None, None, None, None))
    return data


def test_model(model: nn.Module, data: list[tuple[str, int,int,int,int,str]], show_all: bool = True):
    # loop over images to be tested with our model, with ground truth if available
    for filename, gt_start_x, gt_start_y, gt_end_x, gt_end_y, gt_label in data:
        # load the image, copy it, swap its colors channels, resize it, and
        # bring its channel dimension forward
        image = cv2.imread(filename)
        if image is None:
            continue
        display = image.copy()
        h, w = display.shape[:2]

        # convert image to PyTorch tensor, normalize it, upload it to the
        # current device, and add a batch dimension
        image = config.TRANSFORMS(image).to(config.DEVICE)
        image = image.unsqueeze(0)

        # predict the bounding box of the object along with the class label
        label_predictions, bbox_pred_norm = model(image)

        # determine the class label with the largest predicted probability
        label_predictions = torch.nn.Softmax(dim=-1)(label_predictions)
        most_likely_label = label_predictions.argmax(dim=-1).cpu()
        label = config.LABELS[most_likely_label]

        startX_norm, startY_norm, endX_norm, endY_norm = bbox_pred_norm[0].detach().cpu().numpy()
        startX, startY, endX, endY = int(startX_norm * w), int(startY_norm * h), int(endX_norm * w), int(endY_norm * h)

        # draw the ground truth box and class label on the image, if any
        if gt_label is not None:
            cv2.putText(display, 'gt ' + gt_label, (0, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0,  0), 2)
            # BGR format
            cv2.rectangle(display, (gt_start_x, gt_start_y), (gt_end_x, gt_end_y), (255, 0, 0),3)

        # draw the predicted bounding box and class label on the image
        cv2.putText(display, label, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.rectangle(display, (startX, startY), (endX, endY), (0, 0, 255), 3)


        # show the output image
        if show_all or label != gt_label:
            cv2.imshow("Output", display)

            # exit on escape key or window close 
            key = -1
            while key == -1:
                key = cv2.waitKey(100)
                closed = cv2.getWindowProperty('Output', cv2.WND_PROP_VISIBLE) < 1
                if key == 27 or closed:
                    logger.info("Closing the window")
                    sys.exit(0)

def test_model_file(model: nn.Module, test_file: str, save_output:bool = False, output_file:str | None= None) -> None:
    image = cv2.imread(test_file)
    if image is None:
        return 
    
    display = image.copy()
    h, w = display.shape[:2]

    # convert image to PyTorch tensor, normalize it, upload it to the
    # current device, and add a batch dimension
    image = config.TRANSFORMS(image).to(config.DEVICE)
    image = image.unsqueeze(0)

    # predict the bounding box of the object along with the class label
    label_predictions, bbox_pred_norm = model(image)

    label_predictions = torch.nn.Softmax(dim=-1)(label_predictions)
    most_likely_label = label_predictions.argmax(dim=-1).cpu()
    label = config.LABELS[most_likely_label]

    startX_norm, startY_norm, endX_norm, endY_norm = bbox_pred_norm[0].detach().cpu().numpy()
    startX, startY, endX, endY = int(startX_norm * w), int(startY_norm * h), int(endX_norm * w), int(endY_norm * h)


    # draw the predicted bounding box and class label on the image
    cv2.putText(display, label, (0, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(display, (startX, startY), (endX, endY), (0, 0, 255), 3)

    cv2.imshow("Output", display)

    if save_output:
        if output_file is None:
            logger.error("No output file")
            return
        logger.info(f"File save at: {output_file}")
        cv2.imwrite(output_file, display)
    else:
        logger.info(f"File not saved")

    # exit on escape key or window close 
    key = -1
    while key == -1:
        key = cv2.waitKey(100)
        closed = cv2.getWindowProperty('Output', cv2.WND_PROP_VISIBLE) < 1
        if key == 27 or closed:
            logger.info("Closing the window")
            sys.exit(0)


def main():
    arg = setup_args_prediction()
    model = load_model(arg.model, config.DEVICE)
    if arg.filename:
        test_file = arg.filename
        save_file = arg.save_file
        test_model_file(model=model, test_file=test_file, save_output=save_file, output_file=arg.output_file)
    else:
        data = load_data(arg.directory)
        test_model(model=model, data=data, show_all=arg.show_all_images)


if __name__ == "__main__":
    main()