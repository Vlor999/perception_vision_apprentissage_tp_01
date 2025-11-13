
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
    # Conv layer 2: 32 -> 32 channels
    to_Conv(
        "conv2",
        224,
        32,
        offset="(2,0,0)",
        to="(conv1-east)",
        height=224,
        depth=224,
        width=4,
        caption="Conv 32 to 32",
    ),
    to_connection("conv1", "conv2"),
    # MaxPool 4x4
    to_Pool(
        "pool1",
        offset="(0,0,0)",
        to="(conv2-east)",
        height=56,
        depth=56,
        width=1,
        opacity=0.5,
    ),
    # Conv layer 3: 32 -> 64 channels
    to_Conv(
        "conv3",
        56,
        64,
        offset="(2,0,0)",
        to="(pool1-east)",
        height=56,
        depth=56,
        width=8,
        caption="Conv 32 to 64",
    ),
    to_connection("pool1", "conv3"),
    # Conv layer 4: 64 -> 64 channels
    to_Conv(
        "conv4",
        56,
        64,
        offset="(2,0,0)",
        to="(conv3-east)",
        height=56,
        depth=56,
        width=8,
        caption="Conv 64 to 64",
    ),
    to_connection("conv3", "conv4"),
    # MaxPool 4x4
    to_Pool(
        "pool2",
        offset="(0,0,0)",
        to="(conv4-east)",
        height=14,
        depth=14,
        width=1,
        opacity=0.5,
    ),
    # Conv layer 5: 64 -> 128 channels
    to_Conv(
        "conv5",
        14,
        128,
        offset="(2,0,0)",
        to="(pool2-east)",
        height=14,
        depth=14,
        width=8,
        caption="Conv 64 to 128",
    ),
    to_connection("pool2", "conv5"),
    # Conv layer 6: 128 -> 128 channels
    to_Conv(
        "conv6",
        14,
        128,
        offset="(2,0,0)",
        to="(conv5-east)",
        height=14,
        depth=14,
        width=8,
        caption="Conv 128 to 128",
    ),
    to_connection("conv5", "conv6"),
    # MaxPool 2x2
    to_Pool(
        "pool3",
        offset="(0,0,0)",
        to="(conv6-east)",
        height=7,
        depth=7,
        width=1,
        opacity=0.5,
    ),
    # Conv layer 7: 128 -> 256 channels
    to_Conv(
        "conv7",
        7,
        256,
        offset="(2,0,0)",
        to="(pool3-east)",
        height=7,
        depth=7,
        width=8,
        caption="Conv 128 to 256",
    ),
    to_connection("pool3", "conv7"),
    # Conv layer 8: 256 -> 256 channels
    to_Conv(
        "conv8",
        7,
        256,
        offset="(2,0,0)",
        to="(conv7-east)",
        height=7,
        depth=7,
        width=8,
        caption="Conv 256 to 256",
    ),
    to_connection("conv7", "conv8"),
    # Conv layer 9: 256 -> 512 channels
    to_Conv(
        "conv9",
        7,
        512,
        offset="(2,0,0)",
        to="(conv8-east)",
        height=7,
        depth=7,
        width=8,
        caption="Conv 256 to 512",
    ),
    to_connection("conv8", "conv9"),
    # Conv layer 10: 512 -> 512 channels
    to_Conv(
        "conv10",
        7,
        512,
        offset="(2,0,0)",
        to="(conv9-east)",
        height=7,
        depth=7,
        width=8,
        caption="Conv 512 to 512",
    ),
    to_connection("conv9", "conv10"),
    # Flatten
    to_Conv(
        "flatten",
        7,
        512,
        offset="(2,0,0)",
        to="(conv10-east)",
        height=7,
        depth=7,
        width=6,
        caption="Flatten\\\\25088",
    ),
    to_connection("conv10", "flatten"),
    # Fully connected layer 1: 25088 -> 1024
    to_Conv(
        "fc1",
        1,
        1024,
        offset="(3,0,0)",
        to="(flatten-east)",
        height=64,
        depth=64,
        width=4,
        caption="FC 25088 to 1024",
    ),
    to_connection("flatten", "fc1"),
    # Fully connected layer 2: 1024 -> 512
    to_Conv(
        "fc2",
        1,
        512,
        offset="(3,0,0)",
        to="(fc1-east)",
        height=64,
        depth=64,
        width=4,
        caption="FC 1024 to 512",
    ),
    to_connection("fc1", "fc2"),
    # Fully connected layer 3: 512 -> 256
    to_Conv(
        "fc3",
        1,
        256,
        offset="(3,0,0)",
        to="(fc2-east)",
        height=64,
        depth=64,
        width=4,
        caption="FC 512 to 256",
    ),
    to_connection("fc2", "fc3"),
    # Fully connected layer 4: 256 -> 128
    to_Conv(
        "fc4",
        1,
        128,
        offset="(3,0,0)",
        to="(fc3-east)",
        height=64,
        depth=64,
        width=4,
        caption="FC 256 to 128",
    ),
    to_connection("fc3", "fc4"),
    # Fully connected layer 5: 128 -> 64
    to_Conv(
        "fc5",
        1,
        64,
        offset="(3,0,0)",
        to="(fc4-east)",
        height=64,
        depth=64,
        width=4,
        caption="FC 128 to 64",
    ),
    to_connection("fc4", "fc5"),
    # Fully connected layer 6: 64 -> 3
    to_Conv(
        "fc6",
        1,
        3,
        offset="(3,0,0)",
        to="(fc5-east)",
        height=8,
        depth=8,
        width=2,
        caption="FC 64 to 3",
    ),
    to_connection("fc5", "fc6"),
    # Output
    to_SoftMax("output", 3, "(2,0,0)", "(fc6-east)", caption="Output\\\\3 classes"),
    to_connection("fc6", "output"),
    to_end(),
]

def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")


if __name__ == "__main__":
    main()
