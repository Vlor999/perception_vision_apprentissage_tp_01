
import sys
import os

# Get the absolute path to PlotNeuralNet
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
plotnn_path = os.path.join(project_root, "PlotNeuralNet")

sys.path.insert(0, plotnn_path)
sys.path.insert(0, project_root)

from PlotNeuralNet.pycore.tikzeng import *

arch = [
    to_head(plotnn_path),
    to_cor(),
    to_begin(),
    # Conv layer 1: 3 -> 32 channels
    to_Conv(
        "conv1",
        224,
        32,
        offset="(0,0,0)",
        to="(0,0,0)",
        height=224,
        depth=224,
        width=4,
        caption="Conv 3 to 32",
    ),
    # MaxPool 4x4
    to_Pool(
        "pool1",
        offset="(0,0,0)",
        to="(conv1-east)",
        height=56,
        depth=56,
        width=1,
        opacity=0.5,
    ),
    # Conv layer 2: 32 -> 64 channels
    to_Conv(
        "conv2",
        56,
        64,
        offset="(2,0,0)",
        to="(pool1-east)",
        height=56,
        depth=56,
        width=8,
        caption="Conv 32 to 64",
    ),
    to_connection("pool1", "conv2"),
    # MaxPool 4x4
    to_Pool(
        "pool2",
        offset="(0,0,0)",
        to="(conv2-east)",
        height=14,
        depth=14,
        width=1,
        opacity=0.5,
    ),
    # Conv layer 3: 64 -> 64 channels
    to_Conv(
        "conv3",
        14,
        64,
        offset="(2,0,0)",
        to="(pool2-east)",
        height=14,
        depth=14,
        width=8,
        caption="Conv 64 to 64",
    ),
    to_connection("pool2", "conv3"),
    # MaxPool 4x4
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
    # Fully connected layer 1: 576 -> 32
    to_Conv(
        "fc1",
        1,
        32,
        offset="(3,0,0)",
        to="(flatten-east)",
        height=32,
        depth=32,
        width=4,
        caption="FC 576 to 32",
    ),
    to_connection("flatten", "fc1"),
    # Fully connected layer 2: 32 -> 16
    to_Conv(
        "fc2",
        1,
        16,
        offset="(3,0,0)",
        to="(fc1-east)",
        height=16,
        depth=16,
        width=2,
        caption="FC 32 to 16",
    ),
    to_connection("fc1", "fc2"),
    # Fully connected layer 3: 16 -> 3
    to_Conv(
        "fc3",
        1,
        3,
        offset="(3,0,0)",
        to="(fc2-east)",
        height=8,
        depth=8,
        width=2,
        caption="FC 16 to 3",
    ),
    to_connection("fc2", "fc3"),
    # Output
    to_SoftMax("output", 3, "(2,0,0)", "(fc3-east)", caption="Output\\\\3 classes"),
    to_connection("fc3", "output"),
    to_end(),
]

def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")


if __name__ == "__main__":
    main()
