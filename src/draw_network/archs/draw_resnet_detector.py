import sys
import os

# Get the absolute path to PlotNeuralNet
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
plotnn_path = os.path.join(project_root, "PlotNeuralNet")
sys.path.insert(0, plotnn_path)

from PlotNeuralNet.pycore.tikzeng import *

# ResnetObjectDetector Architecture
# ResNet18-based feature extraction with frozen pretrained weights
arch = [
    to_head(plotnn_path),
    to_cor(),
    to_begin(),
    # ResNet18 Backbone (Frozen, Pretrained)
    # Represented as a large block to show it's a complete backbone
    to_Conv(
        "resnet18",
        224,
        512,
        offset="(0,0,0)",
        to="(0,0,0)",
        height=56,
        depth=56,
        width=15,
        caption="ResNet18 Backbone\\\\(Pretrained \\& Frozen)\\\\3x224x224 to 512 features",
    ),
    # Flatten layer
    to_Conv(
        "flatten",
        1,
        512,
        offset="(1,0,0)",
        to="(resnet18-east)",
        height=40,
        depth=40,
        width=12,
        caption="Flatten\\\\512 features",
    ),
    to_connection("resnet18", "flatten"),
    # Trainable Classifier Layers
    to_Conv(
        "fc1",
        1,
        512,
        offset="(2,0,0)",
        to="(flatten-east)",
        height=40,
        depth=40,
        width=10,
        caption="FC1\\\\512 to 512\\\\(trainable)",
    ),
    to_connection("flatten", "fc1"),
    to_Conv(
        "fc2",
        1,
        512,
        offset="(2,0,0)",
        to="(fc1-east)",
        height=40,
        depth=40,
        width=10,
        caption="FC2\\\\512 to 512\\\\(trainable)",
    ),
    to_connection("fc1", "fc2"),
    # Output layer
    to_SoftMax(
        "output", 3, "(2,0,0)", "(fc2-east)", caption="Output\\\\512 to 3 classes"
    ),
    to_connection("fc2", "output"),
    to_end(),
]


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")


if __name__ == "__main__":
    main()
