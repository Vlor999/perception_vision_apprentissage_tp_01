
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
    # Conv layer 1: 3 -> 64 channels
    to_Conv(
        "conv1",
        224,
        64,
        offset="(0,0,0)",
        to="(0,0,0)",
        height=224,
        depth=224,
        width=8,
        caption="Conv 3 to 64",
    ),
    # MaxPool 2x2
    to_Pool(
        "pool1",
        offset="(0,0,0)",
        to="(conv1-east)",
        height=112,
        depth=112,
        width=1,
        opacity=0.5,
    ),
    # Conv layer 2: 64 -> 128 channels
    to_Conv(
        "conv2",
        112,
        128,
        offset="(2,0,0)",
        to="(pool1-east)",
        height=112,
        depth=112,
        width=8,
        caption="Conv 64 to 128",
    ),
    to_connection("pool1", "conv2"),
    # MaxPool 2x2
    to_Pool(
        "pool2",
        offset="(0,0,0)",
        to="(conv2-east)",
        height=56,
        depth=56,
        width=1,
        opacity=0.5,
    ),
    # Conv layer 3: 128 -> 256 channels
    to_Conv(
        "conv3",
        56,
        256,
        offset="(2,0,0)",
        to="(pool2-east)",
        height=56,
        depth=56,
        width=8,
        caption="Conv 128 to 256",
    ),
    to_connection("pool2", "conv3"),
    # Conv layer 4: 256 -> 256 channels
    to_Conv(
        "conv4",
        56,
        256,
        offset="(2,0,0)",
        to="(conv3-east)",
        height=56,
        depth=56,
        width=8,
        caption="Conv 256 to 256",
    ),
    to_connection("conv3", "conv4"),
    # MaxPool 2x2
    to_Pool(
        "pool3",
        offset="(0,0,0)",
        to="(conv4-east)",
        height=28,
        depth=28,
        width=1,
        opacity=0.5,
    ),
    # Conv layer 5: 256 -> 512 channels
    to_Conv(
        "conv5",
        28,
        512,
        offset="(2,0,0)",
        to="(pool3-east)",
        height=28,
        depth=28,
        width=8,
        caption="Conv 256 to 512",
    ),
    to_connection("pool3", "conv5"),
    # Conv layer 6: 512 -> 512 channels
    to_Conv(
        "conv6",
        28,
        512,
        offset="(2,0,0)",
        to="(conv5-east)",
        height=28,
        depth=28,
        width=8,
        caption="Conv 512 to 512",
    ),
    to_connection("conv5", "conv6"),
    # MaxPool 2x2
    to_Pool(
        "pool4",
        offset="(0,0,0)",
        to="(conv6-east)",
        height=14,
        depth=14,
        width=1,
        opacity=0.5,
    ),
    # Conv layer 7: 512 -> 512 channels
    to_Conv(
        "conv7",
        14,
        512,
        offset="(2,0,0)",
        to="(pool4-east)",
        height=14,
        depth=14,
        width=8,
        caption="Conv 512 to 512",
    ),
    to_connection("pool4", "conv7"),
    # Conv layer 8: 512 -> 512 channels
    to_Conv(
        "conv8",
        14,
        512,
        offset="(2,0,0)",
        to="(conv7-east)",
        height=14,
        depth=14,
        width=8,
        caption="Conv 512 to 512",
    ),
    to_connection("conv7", "conv8"),
    # MaxPool 2x2
    to_Pool(
        "pool5",
        offset="(0,0,0)",
        to="(conv8-east)",
        height=7,
        depth=7,
        width=1,
        opacity=0.5,
    ),
    # Fully connected layer 1: 25088 -> 512
    to_Conv(
        "fc1",
        1,
        512,
        offset="(3,0,0)",
        to="(pool5-east)",
        height=64,
        depth=64,
        width=4,
        caption="FC 25088 to 512",
    ),
    to_connection("pool5", "fc1"),
    # Fully connected layer 2: 512 -> 512
    to_Conv(
        "fc2",
        1,
        512,
        offset="(3,0,0)",
        to="(fc1-east)",
        height=64,
        depth=64,
        width=4,
        caption="FC 512 to 512",
    ),
    to_connection("fc1", "fc2"),
    # Fully connected layer 3: 512 -> 3
    to_Conv(
        "fc3",
        1,
        3,
        offset="(3,0,0)",
        to="(fc2-east)",
        height=8,
        depth=8,
        width=2,
        caption="FC 512 to 3",
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
