from loguru import logger
from src.network import SimpleDetector, DeeperDetector, VGGInspired, ResnetObjectDetector
from train import optimize_for_device, load_data, get_loaders, get_model, train
from src.timer.timer import Timer
from src import config
from torch.optim import Adam
from collections import defaultdict
import os
from PyQt5.QtCore import QLibraryInfo
import torch
import matplotlib.pyplot as plt
import numpy as np

def train_model_by_name(model_name: str, save_model: bool = False):
    """Train a model by its string name using the factory function"""
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

    # Optimiser l'environnement selon le device
    optimize_for_device()

    # initialize the list of data (images), class labels, target bounding
    # box coordinates, and image paths
    data = load_data()

    # randomly partition the data: 80% training, 10% validation, 10% testing
    train_loader, val_loader, test_loader = get_loaders(data)

    # create our custom object detector model using the factory function
    object_detector = get_model(model_name, len(config.LABELS)).to(config.DEVICE)

    # initialize the optimizer, compile the model, and show the model summary
    optimizer = Adam(object_detector.parameters(), lr=config.INIT_LR)
    logger.debug(object_detector)

    # initialize history variables for future plot
    plots = defaultdict(list)

    with Timer():
        train(object_detector=object_detector, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, plots=plots, store_model=save_model)

    return plots

def compare(end_of_line: str):
    all_plots = []
    model_names = ['SimpleDetector', 'DeeperDetector', 'VGGInspired', 'ResnetObjectDetector (Frozen)', 'ResnetObjectDetector (Unfrozen)']
    model_args = ['simple', 'deeper', 'vgg_inspired', 'resnet', 'resnet_unfrozen']
    
    # Train all models and collect their plots
    for i, model_arg in enumerate(model_args):
        logger.info(f"Training {model_names[i]}...")
        plots = train_model_by_name(model_arg)
        all_plots.append((model_names[i], plots))
    
    # Create histograms for each metric
    create_comparison_histograms(all_plots, end_of_line=end_of_line)
    
    # Print summary statistics
    print_summary_statistics(all_plots)
    
    return all_plots

def create_comparison_histograms(all_plots, end_of_line):
    """Create histograms comparing all models for each metric"""
    
    # Define the metrics we want to compare
    metrics = ['Training loss', 'Training class accuracy', 'Validation loss', 'Validation class accuracy']
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        # Collect data for current metric from all models
        model_data = []
        model_names = []
        
        for model_name, plots in all_plots:
            if metric in plots and len(plots[metric]) > 0:
                # Convert tensor values to numpy if needed
                values = plots[metric]
                if hasattr(values[0], 'numpy'):
                    values = [v.numpy() if hasattr(v, 'numpy') else float(v) for v in values]
                else:
                    values = [float(v) for v in values]
                
                model_data.extend(values)
                model_names.extend([model_name] * len(values))
        
        if model_data:
            # Create histogram
            unique_models = list(set(model_names))
            colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_models)))
            
            for i, model in enumerate(unique_models):
                model_values = [val for val, name in zip(model_data, model_names) if name == model]
                ax.hist(model_values, alpha=0.7, label=model, color=colors[i], bins=20, density=True)
            
            ax.set_title(f'{metric}', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No data for\n{metric}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{metric}', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'doc/model_comparison_histograms{end_of_line}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also create individual line plots for training progression
    create_training_progression_plots(all_plots, end_of_line=end_of_line)

def create_training_progression_plots(all_plots, end_of_line:str):
    """Create line plots showing training progression over epochs"""
    
    metrics = ['Training loss', 'Training class accuracy', 'Validation loss', 'Validation class accuracy']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Progression Comparison', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(all_plots)))
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        for model_idx, (model_name, plots) in enumerate(all_plots):
            if metric in plots and len(plots[metric]) > 0:
                # Convert tensor values to numpy if needed
                values = plots[metric]
                if hasattr(values[0], 'numpy'):
                    values = [v.numpy() if hasattr(v, 'numpy') else float(v) for v in values]
                else:
                    values = [float(v) for v in values]
                
                epochs = range(1, len(values) + 1)
                ax.plot(epochs, values, label=model_name, color=colors[model_idx], 
                       linewidth=2, marker='o', markersize=4)
        
        ax.set_title(f'{metric}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'doc/training_progression_comparison{end_of_line}.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_statistics(all_plots):
    """Print summary statistics for each model and metric"""
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    metrics = ['Training loss', 'Training class accuracy', 'Validation loss', 'Validation class accuracy']
    
    for model_name, plots in all_plots:
        print(f"\n* {model_name}:")
        print("-" * 40)
        
        for metric in metrics:
            if metric in plots and len(plots[metric]) > 0:
                values = plots[metric]
                if hasattr(values[0], 'numpy'):
                    values = [v.numpy() if hasattr(v, 'numpy') else float(v) for v in values]
                else:
                    values = [float(v) for v in values]
                
                final_value = values[-1]
                mean_value = np.mean(values)
                std_value = np.std(values)
                min_value = np.min(values)
                max_value = np.max(values)
                
                print(f"  {metric}:")
                print(f"\t* Final: {final_value:.4f}")
                print(f"\t* Mean:  {mean_value:.4f} ± {std_value:.4f}")
                print(f"\t* Range: [{min_value:.4f}, {max_value:.4f}]")
    


if __name__ == "__main__":
    all_plots = compare("bbox")
    logger.info("Model comparison completed!")
    logger.info("Generated files:")
    logger.info("- model_comparison_histograms.png")
    logger.info("- training_progression_comparison.png")