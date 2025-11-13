from src.network import DeeperDetector, ResnetObjectDetector, SimpleDetector, VGGInspired
from typing import Union, Optional
from src import config
from loguru import logger
ObjectDetector = Union[SimpleDetector, DeeperDetector, VGGInspired, ResnetObjectDetector]


def header() -> str:
    return """
import sys
import os

# Get the absolute path to PlotNeuralNet
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
plotnn_path = os.path.join(project_root, "PlotNeuralNet")

sys.path.insert(0, plotnn_path)
sys.path.insert(0, project_root)

from PlotNeuralNet.pycore.tikzeng import *
"""

def parse_conv2d(line: str) -> Optional[dict]:
    """Parse Conv2d layer information"""
    import re
    # Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    match = re.search(r'Conv2d\((\d+),\s*(\d+)', line)
    if match:
        in_channels = int(match.group(1))
        out_channels = int(match.group(2))
        return {'type': 'conv', 'in': in_channels, 'out': out_channels}
    return None

def parse_linear(line: str) -> Optional[dict]:
    """Parse Linear layer information"""
    import re
    # Linear(in_features=576, out_features=32, bias=True)
    match = re.search(r'Linear\(in_features=(\d+),\s*out_features=(\d+)', line)
    if match:
        in_features = int(match.group(1))
        out_features = int(match.group(2))
        return {'type': 'linear', 'in': in_features, 'out': out_features}
    return None

def parse_pool(line: str) -> Optional[dict]:
    """Parse MaxPool2d layer information"""
    import re
    # MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
    match = re.search(r'MaxPool2d\(kernel_size=(\d+)', line)
    if match:
        kernel_size = int(match.group(1))
        return {'type': 'pool', 'kernel_size': kernel_size}
    return None

def calculate_output_size(input_size: int, kernel_size: int, stride: int, padding: int = 0) -> int:
    """Calculate output size after conv/pool operation"""
    return ((input_size - kernel_size + 2 * padding) // stride) + 1

def get_content(model: str) -> str:
    """Parse model string and generate PlotNeuralNet architecture code"""
    lines = model.split("\n")
    
    # Track layers
    layers = []
    current_section = None
    
    for line in lines:
        line_lower = line.lower()
        
        # Track which section we're in
        if "(features):" in line_lower:
            current_section = "features"
            continue
        elif "(classifier):" in line_lower:
            current_section = "classifier"
            continue
        elif "(regression):" in line_lower:
            current_section = "regression"
            break  # We'll focus on features and classifier
        
        # Parse layers
        if "conv2d" in line_lower:
            layer_info = parse_conv2d(line)
            if layer_info:
                layer_info['section'] = current_section
                layers.append(layer_info)
        elif "maxpool2d" in line_lower:
            layer_info = parse_pool(line)
            if layer_info:
                layer_info['section'] = current_section
                layers.append(layer_info)
        elif "flatten" in line_lower:
            layers.append({'type': 'flatten', 'section': current_section})
        elif "linear" in line_lower:
            layer_info = parse_linear(line)
            if layer_info:
                layer_info['section'] = current_section
                layers.append(layer_info)
    
    # Generate PlotNeuralNet code
    arch_code = []
    arch_code.append("arch = [")
    arch_code.append("    to_head(plotnn_path),")
    arch_code.append("    to_cor(),")
    arch_code.append("    to_begin(),")
    
    # Track spatial dimensions (starting with 224x224 input)
    current_size = 224
    conv_count = 0
    pool_count = 0
    fc_count = 0
    last_layer_name = None
    
    for i, layer in enumerate(layers):
        if layer['type'] == 'conv':
            conv_count += 1
            layer_name = f"conv{conv_count}"
            in_channels = layer['in']
            out_channels = layer['out']
            
            # Determine offset and connection
            if conv_count == 1:
                offset = "(0,0,0)"
                to_position = "(0,0,0)"
            else:
                offset = "(2,0,0)"
                to_position = f"({last_layer_name}-east)"
            
            # Scale width based on channels
            width = max(2, min(8, out_channels // 8))
            
            arch_code.append(f"    # Conv layer {conv_count}: {in_channels} -> {out_channels} channels")
            arch_code.append(f"    to_Conv(")
            arch_code.append(f'        "{layer_name}",')
            arch_code.append(f'        {current_size},')
            arch_code.append(f'        {out_channels},')
            arch_code.append(f'        offset="{offset}",')
            arch_code.append(f'        to="{to_position}",')
            arch_code.append(f'        height={current_size},')
            arch_code.append(f'        depth={current_size},')
            arch_code.append(f'        width={width},')
            arch_code.append(f'        caption="Conv {in_channels} to {out_channels}",')
            arch_code.append(f"    ),")
            
            if last_layer_name and conv_count > 1:
                arch_code.append(f'    to_connection("{last_layer_name}", "{layer_name}"),')
            
            last_layer_name = layer_name
            
        elif layer['type'] == 'pool':
            pool_count += 1
            layer_name = f"pool{pool_count}"
            kernel_size = layer['kernel_size']
            
            # Calculate new size after pooling
            new_size = current_size // kernel_size
            
            arch_code.append(f"    # MaxPool {kernel_size}x{kernel_size}")
            arch_code.append(f"    to_Pool(")
            arch_code.append(f'        "{layer_name}",')
            arch_code.append(f'        offset="(0,0,0)",')
            arch_code.append(f'        to="({last_layer_name}-east)",')
            arch_code.append(f'        height={new_size},')
            arch_code.append(f'        depth={new_size},')
            arch_code.append(f'        width=1,')
            arch_code.append(f'        opacity=0.5,')
            arch_code.append(f"    ),")
            
            current_size = new_size
            last_layer_name = layer_name
            
        elif layer['type'] == 'flatten':
            layer_name = "flatten"
            # Get the last conv layer's output channels
            last_conv = None
            for l in reversed(layers[:i]):
                if l['type'] == 'conv':
                    last_conv = l
                    break
            
            if last_conv:
                flat_size = last_conv['out'] * current_size * current_size
                
                arch_code.append(f"    # Flatten")
                arch_code.append(f"    to_Conv(")
                arch_code.append(f'        "{layer_name}",')
                arch_code.append(f'        {current_size},')
                arch_code.append(f'        {last_conv["out"]},')
                arch_code.append(f'        offset="(2,0,0)",')
                arch_code.append(f'        to="({last_layer_name}-east)",')
                arch_code.append(f'        height={current_size},')
                arch_code.append(f'        depth={current_size},')
                arch_code.append(f'        width=6,')
                arch_code.append(f'        caption="Flatten\\\\\\\\{flat_size}",')
                arch_code.append(f"    ),")
                arch_code.append(f'    to_connection("{last_layer_name}", "{layer_name}"),')
                
                last_layer_name = layer_name
                
        elif layer['type'] == 'linear':
            fc_count += 1
            layer_name = f"fc{fc_count}"
            in_features = layer['in']
            out_features = layer['out']
            
            # Scale height based on features
            height = max(8, min(64, out_features))
            width = max(2, min(4, out_features // 8))
            
            arch_code.append(f"    # Fully connected layer {fc_count}: {in_features} -> {out_features}")
            arch_code.append(f"    to_Conv(")
            arch_code.append(f'        "{layer_name}",')
            arch_code.append(f'        1,')
            arch_code.append(f'        {out_features},')
            arch_code.append(f'        offset="(3,0,0)",')
            arch_code.append(f'        to="({last_layer_name}-east)",')
            arch_code.append(f'        height={height},')
            arch_code.append(f'        depth={height},')
            arch_code.append(f'        width={width},')
            arch_code.append(f'        caption="FC {in_features} to {out_features}",')
            arch_code.append(f"    ),")
            arch_code.append(f'    to_connection("{last_layer_name}", "{layer_name}"),')
            
            last_layer_name = layer_name
    
    # Add output layer
    # Find the last linear layer to get number of output classes
    last_linear = None
    for layer in reversed(layers):
        if layer['type'] == 'linear' and layer['section'] == 'classifier':
            last_linear = layer
            break
    
    if last_linear:
        num_classes = last_linear['out']
        arch_code.append(f"    # Output")
        arch_code.append(f'    to_SoftMax("output", {num_classes}, "(2,0,0)", "({last_layer_name}-east)", caption="Output\\\\\\\\{num_classes} classes"),')
        arch_code.append(f'    to_connection("{last_layer_name}", "output"),')
    
    arch_code.append("    to_end(),")
    arch_code.append("]")
    
    return "\n".join(arch_code)

def get_footer() -> str:
    return """
def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")


if __name__ == "__main__":
    main()
"""

def get_model(model_name: str, nb_classes: int) -> ObjectDetector:
    """Factory function to create model based on command line argument"""
    if model_name == "simple":
        return SimpleDetector(nb_classes)
    elif model_name == "deeper":
        return DeeperDetector(nb_classes)
    elif model_name == "vgg_inspired":
        return VGGInspired(nb_classes)
    elif model_name == "resnet":
        return ResnetObjectDetector(nb_classes, freeze_features=True)
    elif model_name == "resnet_unfrozen":
        return ResnetObjectDetector(nb_classes, freeze_features=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Generate PlotNeuralNet drawing files')
    parser.add_argument('--model', type=str, default=None, 
                        choices=['simple', 'deeper', 'vgg_inspired', 'resnet', 'resnet_unfrozen'],
                        help='Model to generate drawing for (if not specified, generates for all models)')
    args = parser.parse_args()
    
    # If no model specified, generate for all models
    if args.model is None:
        models_to_generate = ['simple', 'deeper', 'vgg_inspired', 'resnet', 'resnet_unfrozen']
        logger.debug("No model specified, generating files for all models...\n")
    else:
        models_to_generate = [args.model]
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "archs")
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name in models_to_generate:
        logger.debug("="*80)
        logger.debug(f"Generating drawing file for {model_name}...")
        logger.debug("="*80)
        
        # Get model
        model = get_model(model_name, len(config.LABELS))
        model_str = model.__str__()
        
        logger.debug(model_str)
        logger.debug("\n")
        
        # Generate content
        header_txt = header()
        content_txt = get_content(model_str)
        footer_txt = get_footer()
        
        full_file = header_txt + "\n" + content_txt + "\n" + footer_txt
        
        # Save to file
        output_file = os.path.join(output_dir, f"draw_{model_name}_detector.py")
        
        with open(output_file, 'w') as f:
            f.write(full_file)
        
        logger.debug(f"✓ Generated file: {output_file}")
    
    logger.debug("="*80)
    logger.debug(f"All done! Generated {len(models_to_generate)} file(s).")
    logger.debug("="*80)
    logger.info("To generate all the PDF, PNG and LATEX files you must run : ./draw_networks.sh")


if __name__ == "__main__":
    main()