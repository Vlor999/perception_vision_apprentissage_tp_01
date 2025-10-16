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
    model.eval()
    return model


def load_data(filename: str) -> list[tuple[str, int, int, int, int, str]]:
    data = []
    path = filename
    if path.endswith('.csv'):
        # loop over CSV file rows (filename, startX, startY, endX, endY, label)
        for row in open(path).read().strip().split("\n"):
            # TODO: read bounding box annotations
            filename, startX, startY, endX, endY, label = row.split(',')
            filename = os.path.join(config.IMAGES_PATH, label, filename)
            # TODO: add bounding box annotations here
            startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
            data.append((filename, startX, startY, endX, endY, label))
    else:
        data.append((path, None, None, None, None, None))
    return data


def test_model(model: nn.Module, data: list[tuple[str, int,int,int,int,str]], show_all: bool = True):
    # loop over images to be tested with our model, with ground truth if available
    # TODO: must read bounding box annotations once added
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
        # TODO: need to retrieve label AND bbox predictions once added in network
        label_predictions = model(image)

        # determine the class label with the largest predicted probability
        label_predictions = torch.nn.Softmax(dim=-1)(label_predictions)
        most_likely_label = label_predictions.argmax(dim=-1).cpu()
        label = config.LABELS[most_likely_label]

        # TODO:denormalize bounding box from (0,1)x(0,1) to (0,w)x(0,h)

        # draw the ground truth box and class label on the image, if any
        if gt_label is not None:
            cv2.putText(display, 'gt ' + gt_label, (0, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0,  0), 2)
            # TODO: display ground truth bounding box in blue
            # BGR format
            cv2.rectangle(display, (gt_start_x, gt_start_y), (gt_end_x, gt_end_y), (255, 0, 0),3)

        # draw the predicted bounding box and class label on the image
        cv2.putText(display, label, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        # TODO: display predicted bounding box, don't forget tp denormalize it!

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

def main():
    arg = setup_args_prediction()
    model = load_model(config.LAST_MODEL_PATH, config.DEVICE)
    data = load_data(arg.filename)
    test_model(model=model, data=data, show_all=arg.show_all_images)


if __name__ == "__main__":
    main()