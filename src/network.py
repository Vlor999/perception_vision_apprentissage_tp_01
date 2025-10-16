from torch import nn
from torchvision.models import resnet18


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class SimpleDetector(nn.Module):
    """VGG11 inspired feature extraction layers"""

    def __init__(self, nb_classes):
        """initialize the network"""
        super().__init__()
        # TODO: play with simplifications of this network
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
        )
        self.features.apply(init_weights)

        # create classifier path for class label prediction
        # TODO: play with dimensions of this network and compare
        self.classifier = nn.Sequential(
            # dimension = 64 [nb features per map pixel] x 3x3 [nb_map_pixels]
            # 3 = ImageNet_image_res/(maxpool_stride^#maxpool_layers) = 224/4^3
            nn.Linear(64 * 3 * 3, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, nb_classes),
        )
        self.classifier.apply(init_weights)

        # create regressor path for bounding box coordinates prediction
        # TODO: take inspiration from above without dropouts

    def forward(self, x):
        # get features from input then run them through the classifier
        x = self.features(x)
        # TODO: compute and add the bounding box regressor term
        return self.classifier(x)


# TODO: create a new class based on SimpleDetector to create a deeper model
class DeeperDetector(nn.Module):
    def __init__(self, nb_classes: int) -> None:
        """Deeper network based on SimpleDetector with more layers

        Args:
            - nb_classes: int represent the number of final classes
        """
        super().__init__()
        self.number_classes = nb_classes

        # Deeper feature extraction with more convolutional layers
        # Starting from 224x224 input images
        self.features = nn.Sequential(
            # First block: 224x224 -> 56x56
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),  # 224/4 = 56
            # Second block: 56x56 -> 14x14
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),  # 56/4 = 14
            # Third block: 14x14 -> 7x7 (keeping reasonable size for linear layer)
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14/2 = 7
            # Fourth block: 7x7 -> 7x7 (no pooling to maintain reasonable size)
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Fifth block: Add more depth with 1x1 convolutions
            nn.Conv2d(256, 512, kernel_size=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Flatten for fully connected layers
            # Final feature map size: 7x7x512 = 25088 features
            nn.Flatten(),
        )
        self.features.apply(init_weights)

        # Deeper classifier with more layers and gradual reduction
        self.classifier = nn.Sequential(
            # 7x7x512 = 25088 input features
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, nb_classes),
        )
        self.classifier.apply(init_weights)

        # create regressor path for bounding box coordinates prediction
        # TODO: take inspiration from above without dropouts

    def forward(self, x):
        # get features from input then run them through the classifier
        selected_class = self.classifier(self.features(x))
        # TODO: compute and add the bounding box regressor term
        return selected_class


class VGGInspired(nn.Module):
    """VGG11-inspired network with classic architecture and 512 features"""

    def __init__(self, nb_classes: int) -> None:
        """VGG11-inspired architecture following the original design

        Classic VGG11 structure:
        - Block 1: Conv 64 -> MaxPool
        - Block 2: Conv 128 -> MaxPool
        - Block 3: Conv 256, Conv 256 -> MaxPool
        - Block 4: Conv 512, Conv 512 -> MaxPool
        - Block 5: Conv 512, Conv 512 -> MaxPool
        - FC: 512 features (instead of original 4096)

        Args:
            nb_classes: Number of output classes
        """
        super().__init__()
        self.nb_classes = nb_classes

        # VGG11 feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 3 -> 64 channels, 224x224 -> 112x112
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 64 -> 128 channels, 112x112 -> 56x56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: 128 -> 256 channels (2 conv layers), 56x56 -> 28x28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4: 256 -> 512 channels (2 conv layers), 28x28 -> 14x14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5: 512 -> 512 channels (2 conv layers), 14x14 -> 7x7
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Adaptive pooling to ensure 7x7 output regardless of input size
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.features.apply(init_weights)

        # VGG-style classifier with 512 features instead of 4096
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 512),  # First FC: 25088 -> 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),  # Second FC: 512 -> 512
            nn.ReLU(inplace=True),
            nn.Linear(512, nb_classes),  # Output layer
        )

        # Initialize weights
        self.classifier.apply(init_weights)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# TODO: once played with VGG, play with this
class ResnetObjectDetector(nn.Module):
    """Resnet18 based feature extraction layers"""

    def __init__(self, nb_classes):
        super().__init__()
        # copy resnet up to the last conv layer prior to fc layers, and flatten
        # TODO: add pretrained=True to get pretrained coefficients: what effect?
        features = list(resnet18(pretrained=True).children())[:9]
        self.features = nn.Sequential(*features, nn.Flatten())

        # TODO: first freeze these layers, then comment this loop to
        #  include them in the training
        # freeze all ResNet18 layers during the training process
        for param in self.features.parameters():
            param.requires_grad = False

        # create classifier path for class label prediction
        # TODO: play with dimensions below and see how it compares
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, nb_classes),
        )

        # create regressor path for bounding box coordinates prediction
        # TODO: take inspiration from above without dropouts

    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from two different branches of the network
        x = self.features(x)
        # TODO: compute and add the bounding box regressor term
        return self.classifier(x)
