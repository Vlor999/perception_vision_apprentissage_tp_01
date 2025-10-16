import sys
import os

# Get the absolute path to PlotNeuralNet
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
plotnn_path = os.path.join(project_root, "PlotNeuralNet")

sys.path.insert(0, plotnn_path)
sys.path.insert(0, project_root)

from PlotNeuralNet.pycore.tikzeng import *

# SimpleDetector Architecture
# VGG11-inspired feature extraction layers
arch = [
    to_head(plotnn_path),
    to_cor(),
    to_begin(),
    # Feature extraction block 1: 224x224  to  56x56
    to_Conv(
        "conv1",
        224,
        32,
        offset="(0,0,0)",
        to="(0,0,0)",
        height=56,
        depth=56,
        width=4,
        caption="Conv 3 to 32",
    ),
    to_Pool(
        "pool1",
        offset="(0,0,0)",
        to="(conv1-east)",
        height=14,
        depth=14,
        width=1,
        opacity=0.5,
    ),
    # Feature extraction block 2: 56x56  to  14x14
    to_Conv(
        "conv2",
        56,
        64,
        offset="(2,0,0)",
        to="(pool1-east)",
        height=14,
        depth=14,
        width=6,
        caption="Conv 32 to 64",
    ),
    to_connection("pool1", "conv2"),
    to_Pool(
        "pool2",
        offset="(0,0,0)",
        to="(conv2-east)",
        height=7,
        depth=7,
        width=1,
        opacity=0.5,
    ),
    # Feature extraction block 3: 14x14  to  3x3
    to_Conv(
        "conv3",
        14,
        64,
        offset="(2,0,0)",
        to="(pool2-east)",
        height=7,
        depth=7,
        width=6,
        caption="Conv 64 to 64",
    ),
    to_connection("pool2", "conv3"),
    to_Pool(
        "pool3",
        offset="(0,0,0)",
        to="(conv3-east)",
        height=3,
        depth=3,
        width=1,
        opacity=0.5,
    ),
    # Flatten
    to_Conv(
        "flatten",
        3,
        64,
        offset="(2,0,0)",
        to="(pool3-east)",
        height=3,
        depth=3,
        width=6,
        caption="Flatten\\\\576",
    ),
    to_connection("pool3", "flatten"),
    # Classifier - FC layers
    to_Conv(
        "fc1",
        1,
        32,
        offset="(3,0,0)",
        to="(flatten-east)",
        height=32,
        depth=32,
        width=3,
        caption="FC 576 to 32",
    ),
    to_connection("flatten", "fc1"),
    to_Conv(
        "fc2",
        1,
        16,
        offset="(2,0,0)",
        to="(fc1-east)",
        height=16,
        depth=16,
        width=2,
        caption="FC 32 to 16",
    ),
    to_connection("fc1", "fc2"),
    # Output
    to_SoftMax("output", 3, "(2,0,0)", "(fc2-east)", caption="Output\\\\3 classes"),
    to_connection("fc2", "output"),
    to_end(),
]


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")


if __name__ == "__main__":
    main()
