import sys
import os

# Get the absolute path to PlotNeuralNet
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
plotnn_path = os.path.join(project_root, "PlotNeuralNet")
sys.path.insert(0, plotnn_path)

from PlotNeuralNet.pycore.tikzeng import *

# DeeperDetector Architecture
# Deeper VGG-inspired network with more convolutional layers
arch = [
    to_head(plotnn_path),
    to_cor(),
    to_begin(),
    # Block 1: 224x224  to  56x56 (2 conv layers + pool)
    to_Conv(
        "conv1_1",
        224,
        32,
        offset="(0,0,0)",
        to="(0,0,0)",
        height=56,
        depth=56,
        width=3,
        caption="Conv1\\_1\\\\3 to 32",
    ),
    to_Conv(
        "conv1_2",
        224,
        32,
        offset="(0,0,0)",
        to="(conv1_1-east)",
        height=56,
        depth=56,
        width=3,
        caption="Conv1\\_2\\\\32 to 32",
    ),
    to_connection("conv1_1", "conv1_2"),
    to_Pool(
        "pool1",
        offset="(0,0,0)",
        to="(conv1_2-east)",
        height=14,
        depth=14,
        width=1,
        opacity=0.5,
        caption="Pool1\\\\4x4",
    ),
    # Block 2: 56x56  to  14x14 (2 conv layers + pool)
    to_Conv(
        "conv2_1",
        56,
        64,
        offset="(1,0,0)",
        to="(pool1-east)",
        height=14,
        depth=14,
        width=5,
        caption="Conv2\\_1\\\\32 to 64",
    ),
    to_connection("pool1", "conv2_1"),
    to_Conv(
        "conv2_2",
        56,
        64,
        offset="(0,0,0)",
        to="(conv2_1-east)",
        height=14,
        depth=14,
        width=5,
        caption="Conv2\\_2\\\\64 to 64",
    ),
    to_connection("conv2_1", "conv2_2"),
    to_Pool(
        "pool2",
        offset="(0,0,0)",
        to="(conv2_2-east)",
        height=7,
        depth=7,
        width=1,
        opacity=0.5,
        caption="Pool2\\\\4x4",
    ),
    # Block 3: 14x14  to  7x7 (2 conv layers + pool)
    to_Conv(
        "conv3_1",
        14,
        128,
        offset="(1,0,0)",
        to="(pool2-east)",
        height=7,
        depth=7,
        width=8,
        caption="Conv3\\_1\\\\64 to 128",
    ),
    to_connection("pool2", "conv3_1"),
    to_Conv(
        "conv3_2",
        14,
        128,
        offset="(0,0,0)",
        to="(conv3_1-east)",
        height=7,
        depth=7,
        width=8,
        caption="Conv3\\_2\\\\128 to 128",
    ),
    to_connection("conv3_1", "conv3_2"),
    to_Pool(
        "pool3",
        offset="(0,0,0)",
        to="(conv3_2-east)",
        height=7,
        depth=7,
        width=1,
        opacity=0.5,
        caption="Pool3\\\\2x2",
    ),
    # Block 4: 7x7  to  7x7 (2 conv layers, no pool)
    to_Conv(
        "conv4_1",
        7,
        256,
        offset="(1,0,0)",
        to="(pool3-east)",
        height=7,
        depth=7,
        width=10,
        caption="Conv4\\_1\\\\128 to 256",
    ),
    to_connection("pool3", "conv4_1"),
    to_Conv(
        "conv4_2",
        7,
        256,
        offset="(0,0,0)",
        to="(conv4_1-east)",
        height=7,
        depth=7,
        width=10,
        caption="Conv4\\_2\\\\256 to 256",
    ),
    to_connection("conv4_1", "conv4_2"),
    # Block 5: 7x7  to  7x7 (1x1 conv + 3x3 conv)
    to_Conv(
        "conv5_1",
        7,
        512,
        offset="(1,0,0)",
        to="(conv4_2-east)",
        height=7,
        depth=7,
        width=12,
        caption="Conv5\\_1\\\\256 to 512\\\\(1x1)",
    ),
    to_connection("conv4_2", "conv5_1"),
    to_Conv(
        "conv5_2",
        7,
        512,
        offset="(0,0,0)",
        to="(conv5_1-east)",
        height=7,
        depth=7,
        width=12,
        caption="Conv5\\_2\\\\512 to 512",
    ),
    to_connection("conv5_1", "conv5_2"),
    # Flatten
    to_Conv(
        "flatten",
        1,
        512,
        offset="(1,0,0)",
        to="(conv5_2-east)",
        height=7,
        depth=7,
        width=12,
        caption="Flatten\\\\25088",
    ),
    to_connection("conv5_2", "flatten"),
    # Deep Classifier - 6 FC layers
    to_Conv(
        "fc1",
        1,
        1024,
        offset="(2,0,0)",
        to="(flatten-east)",
        height=40,
        depth=40,
        width=8,
        caption="FC\\\\25088 to 1024",
    ),
    to_connection("flatten", "fc1"),
    to_Conv(
        "fc2",
        1,
        512,
        offset="(1.5,0,0)",
        to="(fc1-east)",
        height=30,
        depth=30,
        width=6,
        caption="FC\\\\1024 to 512",
    ),
    to_connection("fc1", "fc2"),
    to_Conv(
        "fc3",
        1,
        256,
        offset="(1.5,0,0)",
        to="(fc2-east)",
        height=24,
        depth=24,
        width=5,
        caption="FC\\\\512 to 256",
    ),
    to_connection("fc2", "fc3"),
    to_Conv(
        "fc4",
        1,
        128,
        offset="(1.5,0,0)",
        to="(fc3-east)",
        height=18,
        depth=18,
        width=4,
        caption="FC\\\\256 to 128",
    ),
    to_connection("fc3", "fc4"),
    to_Conv(
        "fc5",
        1,
        64,
        offset="(1.5,0,0)",
        to="(fc4-east)",
        height=12,
        depth=12,
        width=3,
        caption="FC\\\\128 to 64",
    ),
    to_connection("fc4", "fc5"),
    # Output
    to_SoftMax("output", 3, "(1.5,0,0)", "(fc5-east)", caption="Output\\\\64 to 3"),
    to_connection("fc5", "output"),
    to_end(),
]


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")


if __name__ == "__main__":
    main()
