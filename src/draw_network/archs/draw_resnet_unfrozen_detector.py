
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
    # MaxPool 3x3
    to_Pool(
        "pool1",
        offset="(0,0,0)",
        to="(conv1-east)",
        height=74,
        depth=74,
        width=1,
        opacity=0.5,
    ),
    # Conv layer 2: 64 -> 64 channels
    to_Conv(
        "conv2",
        74,
        64,
        offset="(2,0,0)",
        to="(pool1-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 64 to 64",
    ),
    to_connection("pool1", "conv2"),
    # Conv layer 3: 64 -> 64 channels
    to_Conv(
        "conv3",
        74,
        64,
        offset="(2,0,0)",
        to="(conv2-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 64 to 64",
    ),
    to_connection("conv2", "conv3"),
    # Conv layer 4: 64 -> 64 channels
    to_Conv(
        "conv4",
        74,
        64,
        offset="(2,0,0)",
        to="(conv3-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 64 to 64",
    ),
    to_connection("conv3", "conv4"),
    # Conv layer 5: 64 -> 64 channels
    to_Conv(
        "conv5",
        74,
        64,
        offset="(2,0,0)",
        to="(conv4-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 64 to 64",
    ),
    to_connection("conv4", "conv5"),
    # Conv layer 6: 64 -> 128 channels
    to_Conv(
        "conv6",
        74,
        128,
        offset="(2,0,0)",
        to="(conv5-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 64 to 128",
    ),
    to_connection("conv5", "conv6"),
    # Conv layer 7: 128 -> 128 channels
    to_Conv(
        "conv7",
        74,
        128,
        offset="(2,0,0)",
        to="(conv6-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 128 to 128",
    ),
    to_connection("conv6", "conv7"),
    # Conv layer 8: 64 -> 128 channels
    to_Conv(
        "conv8",
        74,
        128,
        offset="(2,0,0)",
        to="(conv7-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 64 to 128",
    ),
    to_connection("conv7", "conv8"),
    # Conv layer 9: 128 -> 128 channels
    to_Conv(
        "conv9",
        74,
        128,
        offset="(2,0,0)",
        to="(conv8-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 128 to 128",
    ),
    to_connection("conv8", "conv9"),
    # Conv layer 10: 128 -> 128 channels
    to_Conv(
        "conv10",
        74,
        128,
        offset="(2,0,0)",
        to="(conv9-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 128 to 128",
    ),
    to_connection("conv9", "conv10"),
    # Conv layer 11: 128 -> 256 channels
    to_Conv(
        "conv11",
        74,
        256,
        offset="(2,0,0)",
        to="(conv10-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 128 to 256",
    ),
    to_connection("conv10", "conv11"),
    # Conv layer 12: 256 -> 256 channels
    to_Conv(
        "conv12",
        74,
        256,
        offset="(2,0,0)",
        to="(conv11-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 256 to 256",
    ),
    to_connection("conv11", "conv12"),
    # Conv layer 13: 128 -> 256 channels
    to_Conv(
        "conv13",
        74,
        256,
        offset="(2,0,0)",
        to="(conv12-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 128 to 256",
    ),
    to_connection("conv12", "conv13"),
    # Conv layer 14: 256 -> 256 channels
    to_Conv(
        "conv14",
        74,
        256,
        offset="(2,0,0)",
        to="(conv13-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 256 to 256",
    ),
    to_connection("conv13", "conv14"),
    # Conv layer 15: 256 -> 256 channels
    to_Conv(
        "conv15",
        74,
        256,
        offset="(2,0,0)",
        to="(conv14-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 256 to 256",
    ),
    to_connection("conv14", "conv15"),
    # Conv layer 16: 256 -> 512 channels
    to_Conv(
        "conv16",
        74,
        512,
        offset="(2,0,0)",
        to="(conv15-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 256 to 512",
    ),
    to_connection("conv15", "conv16"),
    # Conv layer 17: 512 -> 512 channels
    to_Conv(
        "conv17",
        74,
        512,
        offset="(2,0,0)",
        to="(conv16-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 512 to 512",
    ),
    to_connection("conv16", "conv17"),
    # Conv layer 18: 256 -> 512 channels
    to_Conv(
        "conv18",
        74,
        512,
        offset="(2,0,0)",
        to="(conv17-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 256 to 512",
    ),
    to_connection("conv17", "conv18"),
    # Conv layer 19: 512 -> 512 channels
    to_Conv(
        "conv19",
        74,
        512,
        offset="(2,0,0)",
        to="(conv18-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 512 to 512",
    ),
    to_connection("conv18", "conv19"),
    # Conv layer 20: 512 -> 512 channels
    to_Conv(
        "conv20",
        74,
        512,
        offset="(2,0,0)",
        to="(conv19-east)",
        height=74,
        depth=74,
        width=8,
        caption="Conv 512 to 512",
    ),
    to_connection("conv19", "conv20"),
    # Flatten
    to_Conv(
        "flatten",
        74,
        512,
        offset="(2,0,0)",
        to="(conv20-east)",
        height=74,
        depth=74,
        width=6,
        caption="Flatten\\\\2803712",
    ),
    to_connection("conv20", "flatten"),
    # Fully connected layer 1: 512 -> 512
    to_Conv(
        "fc1",
        1,
        512,
        offset="(3,0,0)",
        to="(flatten-east)",
        height=64,
        depth=64,
        width=4,
        caption="FC 512 to 512",
    ),
    to_connection("flatten", "fc1"),
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
