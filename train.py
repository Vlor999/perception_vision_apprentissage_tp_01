from src.dataset import ImageDataset
from src.network import SimpleDetector as ObjectDetector
from src import config
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as fun
from torch.optim import Adam
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time
import os
from loguru import logger
from src.arguments.handle_args import setup_args

from PyQt5.QtCore import QLibraryInfo

# Optimisations pour macOS
def optimize_for_device():
    """Optimise l'environnement selon le device utilisé"""
    if config.DEVICE == "mps":
        # Variables d'environnement pour MPS
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        logger.debug("**** Optimisations MPS activées")
        
        # Vider le cache MPS si disponible
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    
    elif config.DEVICE == "cpu":
        # Optimiser pour CPU multi-thread
        num_threads = os.cpu_count() or 4  # fallback à 4 si None
        torch.set_num_threads(num_threads)
        logger.debug(f"**** CPU optimisé avec {num_threads} threads")

def display_graphs(plots):

    # loop over the plots and add each one to the figure
    for key, values in plots.items():
        plt.plot(values, label=key)

    # add a legend and show the plot
    plt.legend()
    plt.show()

def display_graphs_and_save(plots):
    # loop over the plots and add each one to the figure
    for key, values in plots.items():
        plt.plot(values, label=key)

    # add a legend
    plt.legend()
    
    # IMPORTANT: save the figure BEFORE showing it
    # plt.show() clears the figure, so we must save first
    plt.savefig(config.PLOT_PATH)
    logger.info(f"**** plot saved to {config.PLOT_PATH}")
    
    # then show the plot
    plt.show()

def load_data() -> list[str]:
    logger.debug("**** loading dataset...")
    data = []

    # loop over all CSV files in the annotations directory
    for csv_file in os.listdir(config.ANNOTS_PATH):
        csv_file = os.path.join(config.ANNOTS_PATH, csv_file)
        # loop over CSV file rows (filename, startX, startY, endX, endY, label)
        for row in open(csv_file).read().strip().split("\n"):
            data.append(row.split(','))
    return data

def get_loaders(data: list[str]) -> tuple[DataLoader, DataLoader, DataLoader]:
    random.seed(0)
    random.shuffle(data)

    cut_val = int(0.8 * len(data))   # 0.8
    cut_test = int(0.9 * len(data))  # 0.9
    train_data = data[:cut_val]
    val_data = data[cut_val:cut_test]
    test_data = data[cut_test:]

    # create Torch datasets for our training, validation and test data
    train_dataset = ImageDataset(train_data, transforms=config.TRANSFORMS)
    val_dataset = ImageDataset(val_data, transforms=config.TRANSFORMS)
    test_dataset = ImageDataset(test_data, transforms=config.TRANSFORMS)
    logger.debug(f"**** {len(train_data)} training, {len(val_data)} validation and "
          f"{len(test_data)} test samples")

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NB_WORKERS,
                              pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NB_WORKERS,
                              pin_memory=config.PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             num_workers=config.NB_WORKERS,
                             pin_memory=config.PIN_MEMORY)

    # save testing image paths to use for evaluating/testing our object detector
    logger.debug("**** saving training, validation and testing split data as CSV...")
    with open(config.TEST_PATH, "w") as f:
        f.write("\n".join([','.join(row) for row in test_data]))
    with open(config.VAL_PATH, "w") as f:
        f.write("\n".join([','.join(row) for row in val_data]))
    with open(config.TRAIN_PATH, "w") as f:
        f.write("\n".join([','.join(row) for row in train_data]))
    
    return train_loader, val_loader, test_loader

# function to compute loss over a batch
def compute_loss(loader, object_detector, optimizer, back_prop=False):
    # initialize the total loss and number of correct predictions
    total_loss, correct = 0, 0

    # loop over batches of the training set
    for batch in loader:
        # send the inputs and training annotations to the device
        # TODO: modify line below to get bbox data
        images, labels = [datum.to(config.DEVICE) for datum in batch]

        # perform a forward pass and calculate the training loss
        predict = object_detector(images)

        # TODO: add loss term for bounding boxes
        bbox_loss = 0
        class_loss = fun.cross_entropy(predict, labels, reduction="sum")
        batch_loss = config.BBOXW * bbox_loss + config.LABELW * class_loss

        # zero out the gradients, perform backprop & update the weights
        if back_prop:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        total_loss += batch_loss
        correct_labels = predict.argmax(1) == labels
        correct += correct_labels.type(torch.float).sum().item()

    # return sample-level averages of the loss and accuracy
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def train(object_detector, optimizer, train_loader, val_loader, plots: dict, store_model: bool = False):
    # loop over epochs
    logger.debug("**** training the network...")
    prev_val_acc = None
    prev_val_loss = None
    for e in range(config.NUM_EPOCHS):
        # set model in training mode & backpropagate train loss for all batches
        object_detector.train()

        # Do not use the returned loss
        # The loss of each batch is computed with a "different network"
        # as the weights are updated per batch
        _, _ = compute_loss(loader=train_loader, object_detector=object_detector, optimizer=optimizer, back_prop=True)

        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode and compute validation loss
            object_detector.eval()
            train_loss, train_acc = compute_loss(loader=train_loader, object_detector=object_detector, optimizer=optimizer, back_prop=False)
            val_loss, val_acc = compute_loss(loader=val_loader, object_detector=object_detector, optimizer=optimizer, back_prop=False)

        # update our training history
        plots['Training loss'].append(train_loss.cpu())
        plots['Training class accuracy'].append(train_acc)

        plots['Validation loss'].append(val_loss.cpu())
        plots['Validation class accuracy'].append(val_acc)

        # logger.debug the model training and validation information
        logger.debug(f"**** EPOCH: {e + 1}/{config.NUM_EPOCHS}")
        logger.debug(f"Train loss: {train_loss:.8f}, Train accuracy: {train_acc:.8f}")
        logger.debug(f"Val loss: {val_loss:.8f}, Val accuracy: {val_acc:.8f}")

        # TODO: write code to store model with highest accuracy, lowest loss
        if prev_val_loss is None and val_loss is not None:
            prev_val_loss = val_loss
        if prev_val_acc is None and val_acc is not None:
            prev_val_acc = val_acc

        if store_model and val_acc > prev_val_acc and val_loss < prev_val_loss:
            prev_val_acc = val_acc
            prev_val_loss = val_loss

            # serialize the model to disk
            logger.info("**** saving BEST object detector model...")
            # When a network has dropout and / or batchnorm layers
            # one needs to explicitly set the eval mode before saving
            object_detector.eval()
            torch.save(object_detector, config.BEST_MODEL_PATH)
        elif not store_model:
            logger.debug(f"Val acc: {val_acc} - prev: {prev_val_acc}")
            logger.debug(f"Val loss: {val_loss} - prev: {prev_val_loss}")

def main():
    args = setup_args()
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

    # Optimiser l'environnement selon le device
    optimize_for_device()

    # initialize the list of data (images), class labels, target bounding
    # box coordinates, and image paths

    data = load_data()

    # randomly partition the data: 80% training, 10% validation, 10% testing
    train_loader, val_loader, test_loader = get_loaders(data)

    # create our custom object detector model and upload to the current device
    logger.debug("**** initializing network...")
    object_detector = ObjectDetector(len(config.LABELS)).to(config.DEVICE)

    # initialize the optimizer, compile the model, and show the model summary
    optimizer = Adam(object_detector.parameters(), lr=config.INIT_LR)
    logger.debug(object_detector)

    # initialize history variables for future plot
    plots = defaultdict(list)

    logger.debug("**** saving LAST object detector model...")
    object_detector.eval()
    torch.save(object_detector, config.LAST_MODEL_PATH)

    start_time = time.time()
    train(object_detector=object_detector, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, plots=plots, store_model=True)
    end_time = time.time()
    logger.debug(f"**** total time to train the model: {end_time - start_time:.2f}s")

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()

    # TODO: build and save matplotlib plot
    display_graphs_and_save(plots)


if __name__ == '__main__':
    main()