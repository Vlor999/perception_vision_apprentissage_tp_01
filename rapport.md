Computer Vision and Pattern Recognition
=========================================

Practical 1: Introduction to Neural Networks
=========================================

### Evaluation
The practical will be evaluated in groups of two.
It is expected that you write a report with the plots / results of the 
different experiments. The code must be submitted with the report.

A good report :
* briefly details the experiment setting 
* illustrates the results (with screenshots, tables)
* briefly discusses the outcome of the experiment

```
Reports are due on Dec 2nd 2025
```

### Getting started

The practical will takes place on the Ensimag / Grenoble INP new educational
GPU cluster with manual [here](https://gitlab.ensimag.fr/ensicompute/docs/-/wikis/Utilisation-de-la-plateforme-ensicompute).
In a nutshell, first you log on the slurm server:

```
$ ssh -YK nash.ensimag.fr
```

Then, you can execute pytorch scripts by prefixing them with srun command as follows: 

```
$ srun --gres=shard:1 --cpus-per-task=8 --mem=12GB --x11=all python3 train.py
```

## Part 1: Discover PyTorch  with a simple first classification network

1. Discover the project layout, dataset and code. 
   - **dataset** : Our toy dataset contains images for 3 categories: 
     `motorcycles`, `airplanes` and `faces`. The dataset in on
     Ensimag machines in folder `/matieres/5MMVORF/01-dataset`, referred
     throughout as `dataset` folder below for simplicity. 
     
   - [`train.py`](train.py) is the file containing the main training code. It first
     retrieves the image paths in `dataset/images` and annotations in 
     `dataset/annotations/`, which contain a folder per label type.
     It then constructs a `ImageDataset` object which abstracts the dataset
     for Pytorch. This `Dataset` object is then accessed through a Pytorch 
     `DataLoader` object which allows to retrieve a data group, or batch
     for the optimization. Typically we will use a batch size of 32 in this
     exercise. We will split the main image dataset into three Datasets: the
     training, validation and test sets which were described in the lectures.
     Then comes the training loop where the magic happens and network
     parameters are optimized.
   - the `model` folder contains the model description:
       - [`network.py`](model/network.py) contains the Pytorch Neural Network descriptions
       - [`config.py`](model/config.py) defines the configuration constants and paths
       - [`dataset.py`](model/dataset.py) is where we define our custom PyTorch `Dataset`,
         which basically tells PyTorch where it can retrieve the data an how
         to format it for training.
   - [`predict.py`](predict.py) is a python script to visualize the data and annotations
     in a CSV file. The CSV file can either be one of the dataset files or one
     of the train, validation or test sets generated in the `output/` folder at
     training. It basically loops over all images or CSV files given as
     arguments, runs them through the network, then retrieves and displays
     the predictions.
   - [`eval.py`](eval.py) is a more elaborate evaluation script to run after training.
     It evaluates the performance of a trained model provided as argument
     (stored in `output/{model}.pth`). 
     
     It provides statistics about the overall performance of the training,
     and also per-label statistics which may help you identify imbalances or
     disparities between the labels.
    
   
2. Run the training code (SimpleDetector) `python train.py` in the main
   directory. It loops for 20 epochs, then stores the trained model and 
   parameters after the last epoch in the Pytroch dump file 
   `output/last_model.pth`.

3. Run `python eval.py output/last_model.pth` and observe the numerical 
   results.

*  Answer:
```bash
*** train set accuracy
Mean accuracy for all labels: 0.4833948339483395
    Mean accuracy for label face: 0.0
             0 over 333 samples
    Mean accuracy for label motorcycle: 0.9443585780525502
             611 over 647 samples
    Mean accuracy for label airplane: 0.2708978328173375
             175 over 646 samples
*** validation set accuracy
Mean accuracy for all labels: 0.46798029556650245
    Mean accuracy for label face: 0.0
             0 over 45 samples
    Mean accuracy for label motorcycle: 0.922077922077922
             71 over 77 samples
    Mean accuracy for label airplane: 0.2962962962962963
             24 over 81 samples
*** test set accuracy
Mean accuracy for all labels: 0.47058823529411764
    Mean accuracy for label face: 0.0
             0 over 57 samples
    Mean accuracy for label motorcycle: 0.9324324324324325
             69 over 74 samples
    Mean accuracy for label airplane: 0.3698630136986301
             27 over 73 samples
```
We can see that with the last model the performnance are very low.
For motorcycle the accuracy is even at 0 but for planes it is quite high.

4. Run `predict.py output/{dataset_name}.csv` and observe the predictions.
   Add a flag to the application, such that when this flag is provided 
   on the command line, the script only shows the failure cases 
   in the display loop instead of all images.

* We can see that the model barely always predict motorcycle wich does not make sense for faces and planes.

Example of the use : 
```bash
uv run python predict.py --directory output/test_data.csv --show-all-images true
# If you don't know what to write : 
uv run python predict.py --help
```

5. Write some code in [`train.py`](train.py) to create training loss and accuracy plots
   with matplotlib.

* ![text](output/convergence_plot.png)
Here is the plot for 20 epochs

6. Write some code in [`train.py`](train.py) to store the model with best validation
   accuracy. In case two models have the same accuracy, select the one with lowest loss.

* Now the best model is saved based on its accuracy and on a second time for its loss. for validation and test sets faces have a bit of error.
   
7. Retrain the network and eval again. What can you say about the plot, 
   the selected epoch and the obtained results?

* The best network have way better results for 20 epochs. We are now at 95% of accuracy on the testing dataset   

8. Try Increasing the number of epochs (up to 200)

* With 50 epochs the model is already at is best performance.
* Before running the 200 epochs the expectation is to have overfitting
* After running and evaluating we can see that the accuracy for all labels are at 100% wich is quite estonishing since it should not be this high.

9. For each of the next items, adapt the network or configuation, train it, evaluate it by running [`eval.py`](eval.py)
and comment on the obtained results and failure cases:

    9.1 Train on a small train set (1% - 5% - 10%) and small validation set (20%).
      
      * Train set size:
         * 1% -> 20-50 epochs:
            * mean accuracy on test: 69%
            * mean accuracy on validation: 73%
         * 5% -> 50 epochs
            * mean accuracy on test: 85%
            * mean accuracy on validation: 89%
         * 20% 
            * 20 epochs 
               * mean accuracy on test: 97% 
               * mean accuracy on validation: 96% 
            * 40 epochs 
               * mean accuracy on test: 99% 
               * mean accuracy on validation: 98% 
      
      * We can clearly see that a larger train set improve the performance and this without any huge difference counting on the number of epochs.

    9.2 Try removing the dropout layers and the batch normalizations.

    * ![text](doc/removing_dropout_norm.png)
    We can see that it is less stable. The validation loss and validation accuracy are less continuous.

    9.3 Train the configuration with no dropout, no batch normalization, 
    200 epochs,  0.005 for cut_val and 0.10 for cut_test. 
    Compare the results of the model with best validation accuracy and the model form the last epoch.
    What can you see?

    * ![text](doc/no_dropout_batch_200.png)


10. Download internet images and run them through the network. Show a few
    success and failure cases and comment.


* Here are the results :
```bash
*** test set accuracy
Mean accuracy for all labels: 0.9
    Mean accuracy for label face: 0.75
             3 over 4 samples
    Mean accuracy for label motorcycle: 1.0
             3 over 3 samples
    Mean accuracy for label airplane: 1.0
             3 over 3 samples
```
for all the images it works but for the next image it does not work :
![text](web_images/images/face/web_test_face_4.jpg)
this error makes sens since there is multiple face on the ams image wich is not a case trained by the model


## Part 2: Creating a deeper model + using a pretrained architecture

1. Create another detector class `DeeperDetector` in `model/network.py`. 
   Add layers to it to make it deeper.
   Be careful: MaxPool divides the size of the image by a stride factor.
   In general the kernel size is chosen with the same size as the stride. 
   The Linear layers (fully connected) need to get as input a reasonable 
   number of features: typically 7x7 size of the feature map, multiplied by the
   number of features per pixel, for example 512 to avoid too many combinations. 
   For the output feature map after convolutions and maxpool, the feature map
   needs to be flattened (Flatten).  Train, evaluate and comment.

   **Note**: *the number of input features of a layer needs to match the
   number of output features of the previous layer
   (including after a Flatten operation)*.

   * The new model has been trained on 100 epochs and then evaluated by saving the best model and not the last one. The results are the next ones :
      * Train set accuracy: 100%
      * Test set accuracy: 100%
      * Validation set accuracy: 100%
   
2. Draw inspiration from the VGG11  architecture (look up on internet or in
   course slides) and further deepen your network. Use 512 features in 
   the deeper layer (instead of the 4096 in the vanilla VGG11).
   Train, evaluate and comment.
 
   * The model does not render perfect graphs but gives good results:
      * Train set accuracy: 100%
      * Test set accuracy: 95%
      * Validation set accuracy: 100%
   * The computation takes somehting like 17 minutes for 50 epochs. This represent the fact that there is lots of neuron to train and lots to do.

3. Using pretrained models with Pytorch. It is possible to load pre-trained,
   standard architectures with Pytorch. Look at the given class
   `ResnetObjectDetector` which uses ResNet18. 
   
   The feature extraction layers are simply copied, and we just replace
   the Fully Connected (Linear) layers with our own blocs. Train, evaluate and comment.

   * Even with a small number of epoch the model is really efficient : 
      * Comand line :
         ```bash
         uv run python train.py --model resnet --save-model true --epoch-size 10 && uv run python eval.py output/best_model.pth
         ```
      * For 10-20 epochs:
         * Train set accuracy: 100%
         * Test set accuracy: 100%
         * Validation set accuracy: 100%  

### Here are soome representations of the models:
Simple model:
![Simple Model](src/draw_network/network_diagrams/draw_simple_detector.png)

Deeper model:
![Deeper Model](src/draw_network/network_diagrams/draw_deeper_detector.png)

Resnet model:
![Resnet Model](src/draw_network/network_diagrams/draw_resnet_detector.png)

to create those models you have to run : 
```bash
./draw_networks.sh
```
Those draw are related to the current models but are not automatically extracted from [`network.py](src/network.py). So if a change is made we must change the python files

4. Command to launch: 
```bash
uv run python compare_models.py
```
### Model Comparison: Training Loss

| Model                | Training Loss (Mean) | Training Loss (Average) | Training Loss (Final) |
|----------------------|---------------------|-------------------------|-----------------------|
| SimpleDetector       | 0.1066              | 0.1322                  | 0.0107                |
| DeeperDetector       | 0.0431              | 0.1122                  | 0.0001                |
| VGGInspired          | 0.0278              | 0.0470                  | 0.0000                |
| ResnetObjectDetector | 0.0055              | 0.0160                  | 0.0001                |

### Model Comparison: Training Accuracy

| Model                | Training Accuracy (Mean) | Training Accuracy (Average) | Training Accuracy (Final) |
|----------------------|-------------------------|-----------------------------|---------------------------|
| SimpleDetector       | 0.9910                  | 0.0165                      | 0.9988                    |
| DeeperDetector       | 0.9838                  | 0.0515                      | 1.0000                    |
| VGGInspired          | 0.9895                  | 0.0206                      | 1.0000                    |
| ResnetObjectDetector | 0.9999                  | 0.0002                      | 1.0000                    |

### Model Comparison: Validation Loss

| Model                | Validation Loss (Mean) | Validation Loss (Average) | Validation Loss (Final) |
|----------------------|-----------------------|---------------------------|-------------------------|
| SimpleDetector       | 0.1355                | 0.1365                    | 0.0329                  |
| DeeperDetector       | 0.0821                | 0.1095                    | 0.0414                  |
| VGGInspired          | 0.0893                | 0.0600                    | 0.0617                  |
| ResnetObjectDetector | 0.0053                | 0.0155                    | 0.0001                  |

### Model Comparison: Validation Accuracy

| Model                | Validation Accuracy (Mean) | Validation Accuracy (Average) | Validation Accuracy (Final) |
|----------------------|---------------------------|-------------------------------|-----------------------------|
| SimpleDetector       | 0.9759                    | 0.0243                        | 0.9901                      |
| DeeperDetector       | 0.9768                    | 0.0543                        | 0.9951                      |
| VGGInspired          | 0.9788                    | 0.0219                        | 0.9901                      |
| ResnetObjectDetector | 1.0000                    | 0.0000                        | 1.0000                      |

### Model Comparison: Epoch Time (20 epochs)

| Model                | Epoch Time (20 epochs) |
|----------------------|-----------------------|
| SimpleDetector       | 1m38s                 |
| DeeperDetector       | 3m02s                 |
| VGGInspired          | 7m02s                 |
| ResnetObjectDetector | 1m58s                 |

   **Comments:**
   - ResnetObjectDetector achieves near-perfect accuracy and lowest loss, with fast convergence due to pretrained weights.
   - VGGInspired and DeeperDetector require longer training and more time per epoch, but reach high accuracy.
   - SimpleDetector is fastest but a bit less robust.
   - Loss curves show Resnet starts strong, while other models improve gradually.
   - Per-label results and internet image tests confirm Resnet's generalization; other models may fail on more complex or unseen cases.

   **Images:**
   ![text](doc/Comparaison_models.png)
   ![text](doc/COmparaison_models_progression.png)

   

## Part 3: Bounding Box Regression

So far we have dealt with a classification problem. Now we are going to
estimate the bounding box of the object. This estimation problem is cast as
a *neural network regression problem*, for which the trained neural network will
provide predictions. The bounding box annotations used for training are given
in the same CSV files as we have used up to now. They include the top left 
and bottom right coordinates of the box. 

For best mutual performance on the classification and regression tasks,
we will make a co-learning network. In such a network, the feature layers will be shared between the two tasks of 
classification and bounding box prediction. For this, we are going to add a 
new second branch to our networks, dedicated to the bounding box regression. 
It will be composed by (Linear, Relu, Linear, Relu, Linear, **Sigmoid**, 
*and no Dropout*). From the same extracted features, each branch
(classification and regression) are going to have their own Fully Connected
layers to perform their specialized task.

You shall start with SimpleNetwork, and once you have better understanding,
propagate similar additions to your other networks.

1. Create the fully connected network for the bounding box regression task in
   your pre-existing networks.  

2. The forward method now needs to be modified, instead of only outputting the
   label, it needs to output the regressed bounding box (return a 4-tuple).
   The 4-tuple needs to be retrieved in [`train.py`](train.py), [`eval.py`](eval.py)
   and [`predict.py`](predict.py):
   `predict = object_detector(images)` will now have two components 
   `predict[0]` and `predict[1]`.
   
3. Add code to read the annotations in [`train.py`](train.py) and in
   [`dataset.py`](model/dataset.py). Be careful, the bounding box coordinates need to be
   normalized to be in [0,1] x [0,1], a ratio with respect the size
   of the image, even though the images are themselves normalized at 224x224 
   (ImageNet Size) before they are passed to the network. 

4. We need to compute the new training loss component for the bounding boxes: 
   modify in [`train.py`](train.py) the `compute_loss` function to get the
   annotations of the bounding box and compute the loss of the prediction. 

5. You can now execute `python train.py` and display bounding box losses to see if 
   things work. 
   
6. Now let's see what happens visually! Modify [`predict.py`](predict.py) to display
   the bounding boxes in images for predicted bounding boxes and the ground
   truth bounding boxes (using `cv2.rectangle`). 

* We can launch the current training and evaluation with : 
```bash
uv run python train.py --model simple --save-model true --epoch-size 20 && uv run python predict.py --filename output/test_data.csv --show-all-images true
```

We can see that on most of the cases the bouding boxe is center on the object (face, plane or moto). 
If we do not need to have very precise information for most of the case it is efficient and enough with the small model.

* Images:
   * Simple model:
![text](doc/bbox_data/convergence_plot_simple.png)

   * Deeper model:
![text](doc/bbox_data/convergence_plot_deeper.png)

   * VGG model:
![text](doc/bbox_data/convergence_plot_vgg.png)

   * Resnet model:
![text](doc/bbox_data/convergence_plot_resnet.png)


   
7. Resnet can be loaded with pretrained weights (constructor argument 
   `resnet18(pretrained = True`, in [`network.py`](model/network.py)).
   They can be frozen or included in the training to be refined towards the task 
   (transfer learning). For the latter, simply comment the code calling the 
   `param.requires_grad = False` in [`network.py`](model/network.py).
   Test the possibilities and compare.

* One the classification domain there is no big difference between the frozen and the unfrozen model.
But on the bbox detection there is a huge difference.
The files : 
[`frozen_resnet.log`](doc/evaluation_models_resnet/frozen_resnet.log)
[`unfrozen_resnet.log`](doc/evaluation_models_resnet/unfrozen_resnet)
show that the bbox values like IoU and distance are a way better with unfrozen parameters.

### Bounding Box Regression: Frozen vs Unfrozen ResNet

#### Bounding Box Regression Results: Frozen vs Unfrozen ResNet

| Category   | Model      | Mean Distance | Mean IoU | IoU ≥ 0.5 (%) | IoU ≥ 0.7 (%) |
|------------|------------|---------------|----------|---------------|---------------|
| Face       | Frozen     | 0.2799        | 0.4242   | 40.0%         | 5.0%          |
| Face       | Unfrozen   | 0.1196        | 0.6879   | 91.2%         | 52.6%         |
| Motorcycle | Frozen     | 0.2442        | 0.4242   | 40.0%         | 5.0%          |
| Motorcycle | Unfrozen   | 0.1039        | 0.5718   | 70.3%         | 17.6%         |
| Airplane   | Frozen     | 0.2520        | 0.5339   | 64.4%         | 12.3%         |
| Airplane   | Unfrozen   | 0.0926        | 0.7895   | 100.0%        | 91.8%         |

**Summary:**  
Unfreezing the ResNet backbone significantly improves bounding box localization across all categories, with higher IoU scores and lower mean distances. The effect is most pronounced for faces and airplanes.

8. (optional) Do the same for your other Neural Network architectures. 
   Test and compare your different architectures `SimpleDetector`, 
   `DeeperDetector`
   and `ResnetDetector` and plot the loss evolutions on a common error plot
   now adjusted for bounding box prediction. 
   Note your observations on training losses, results, failure cases in a 
   error analysis table.

### Comprehensive Model Comparison with Bounding Box Regression

We conducted a comprehensive comparison of all neural network architectures with bounding box regression capabilities. All models were trained for 20 epochs with identical hyperparameters.

**Command to reproduce:**
```bash
uv run python compare_models.py
```

#### Training Performance Analysis

**Training Loss Comparison:**

| Model | Final Loss | Mean Loss | Std Dev | Convergence |
|-------|------------|-----------|---------|-------------|
| SimpleDetector | 0.0200 | 0.1789 | ±0.3517 | Moderate |
| DeeperDetector | 0.0043 | 0.0453 | ±0.0896 | Good |
| VGGInspired | 0.0021 | 0.0509 | ±0.0935 | Good |
| ResNet (Frozen) | 0.0154 | 0.0479 | ±0.0535 | Very Good |
| ResNet (Unfrozen) | 0.0068 | 0.0119 | ±0.0144 | Excellent |

**Training Classification Accuracy:**

| Model | Final Accuracy | Mean Accuracy | Performance |
|-------|----------------|---------------|-------------|
| SimpleDetector | 99.7% | 98.6% ±2.2% | Good |
| DeeperDetector | 100.0% | 99.2% ±2.2% | Very Good |
| VGGInspired | 100.0% | 98.7% ±2.8% | Very Good |
| ResNet (Frozen) | 100.0% | 99.97% ±0.06% | Excellent |
| ResNet (Unfrozen) | 100.0% | 100.0% ±0.0% | Perfect |

#### Validation Performance Analysis

**Validation Loss:**

| Model | Final Loss | Mean Loss | Generalization |
|-------|------------|-----------|----------------|
| SimpleDetector | 0.0434 | 0.2048 ±0.3545 | Moderate |
| DeeperDetector | 0.0080 | 0.0799 ±0.1034 | Good |
| VGGInspired | 0.0476 | 0.1064 ±0.1050 | Good |
| ResNet (Frozen) | 0.0356 | 0.0685 ±0.0531 | Very Good |
| ResNet (Unfrozen) | 0.0079 | 0.0156 ±0.0150 | Excellent |

**Validation Classification Accuracy:**

| Model | Final Accuracy | Mean Accuracy | Stability |
|-------|----------------|---------------|-----------|
| SimpleDetector | 99.0% | 97.5% ±2.4% | Good |
| DeeperDetector | 100.0% | 98.3% ±2.6% | Very Good |
| VGGInspired | 99.0% | 97.0% ±3.6% | Good |
| ResNet (Frozen) | 100.0% | 100.0% ±0.0% | Perfect |
| ResNet (Unfrozen) | 100.0% | 100.0% ±0.0% | Perfect |

#### Key Observations and Analysis

**Training Convergence:**
- **ResNet (Unfrozen)**: Best overall performance with lowest loss variance (±0.0144) and perfect accuracy
- **ResNet (Frozen)**: Excellent performance with minimal training, demonstrating transfer learning effectiveness
- **VGGInspired & DeeperDetector**: Good convergence but require more epochs than ResNet variants
- **SimpleDetector**: Moderate performance with highest loss variance, indicating training instability

**Generalization Capability:**
- **ResNet models**: Both variants show perfect validation accuracy with excellent generalization
- **DeeperDetector**: Best among non-pretrained models for generalization
- **VGGInspired**: Good performance but slightly more prone to overfitting (higher validation loss variance)
- **SimpleDetector**: Shows largest gap between training and validation performance

**Transfer Learning Impact:**
- **Frozen ResNet**: Achieves excellent results with minimal parameter updates, proving transfer learning efficiency
- **Unfrozen ResNet**: Superior performance justifies the computational cost of fine-tuning
- **From-scratch models**: Require significantly more training for comparable results

**Failure Case Analysis:**
- **SimpleDetector**: Occasional instability during training (high loss variance)
- **VGGInspired**: Slower initial convergence requiring careful learning rate tuning
- **All models**: Perfect classification but varying bounding box precision (see previous analysis)

**Computational Efficiency:**
- **ResNet (Frozen)**: Best performance/computation ratio
- **SimpleDetector**: Fastest training but lowest final performance
- **VGGInspired**: Highest computational cost for marginal gains over DeeperDetector

**Visualization:**
The complete loss evolution plots and training progression are available in:
![Model Comparison Histograms](doc/compare_all_bbox/model_comparaison_bbox.png)
![Training Progression](doc/compare_all_bbox/training_comparaison_bbox.png)

#### Bounding Box Performance Analysis

We evaluated all models for bounding box localization performance. Here are the detailed results:

**SimpleDetector Test Set Performance:**
- **Mean IoU**: 0.803 (excellent localization)
- **Good Detection Rate** (IoU > 0.5): 97.1%
- **Very Good Detection Rate** (IoU > 0.7): 77.5%
- **Mean Distance**: 0.083 (normalized)

**DeeperDetector Test Set Performance:**
- **Mean IoU**: 0.833 (excellent localization)
- **Good Detection Rate** (IoU > 0.5): 100.0%
- **Very Good Detection Rate** (IoU > 0.7): 91.2%
- **Mean Distance**: 0.071 (normalized)

**VGGInspired Test Set Performance:**
- **Mean IoU**: 0.852 (excellent localization)
- **Good Detection Rate** (IoU > 0.5): 100.0%
- **Very Good Detection Rate** (IoU > 0.7): 97.1%
- **Mean Distance**: 0.062 (normalized)

**ResNet Frozen Test Set Performance:**
- **Mean IoU**: 0.660 (good localization)
- **Good Detection Rate** (IoU > 0.5): 85.3%
- **Very Good Detection Rate** (IoU > 0.7): 47.1%
- **Mean Distance**: 0.156 (normalized)

**ResNet Unfrozen Test Set Performance:**
- **Mean IoU**: 0.758 (excellent localization)
- **Good Detection Rate** (IoU > 0.5): 97.5%
- **Very Good Detection Rate** (IoU > 0.7): 78.9%
- **Mean Distance**: 0.104 (normalized)

**Per-Category Bounding Box Performance Comparison (Test Set):**

| Model | Category | Mean IoU | Mean Distance | Good Detection (%) | Very Good Detection (%) |
|-------|----------|----------|---------------|-------------------|------------------------|
| SimpleDetector | Face | 0.646 | 0.141 | 89.5% | 31.6% |
| SimpleDetector | Motorcycle | 0.848 | 0.071 | 100.0% | 93.2% |
| SimpleDetector | Airplane | 0.881 | 0.051 | 100.0% | 97.3% |
| DeeperDetector | Face | 0.754 | 0.092 | 100.0% | 77.2% |
| DeeperDetector | Motorcycle | 0.872 | 0.059 | 100.0% | 97.3% |
| DeeperDetector | Airplane | 0.855 | 0.065 | 100.0% | 95.9% |
| VGGInspired | Face | 0.809 | 0.071 | 100.0% | 91.2% |
| VGGInspired | Motorcycle | 0.851 | 0.068 | 100.0% | 98.6% |
| VGGInspired | Airplane | 0.886 | 0.048 | 100.0% | 100.0% |
| ResNet Frozen | Face | 0.499 | 0.209 | 50.9% | 15.8% |
| ResNet Frozen | Motorcycle | 0.726 | 0.139 | 100.0% | 60.8% |
| ResNet Frozen | Airplane | 0.718 | 0.132 | 97.3% | 57.5% |
| ResNet Unfrozen | Face | 0.688 | 0.120 | 91.2% | 52.6% |
| ResNet Unfrozen | Motorcycle | 0.781 | 0.104 | 100.0% | 86.5% |
| ResNet Unfrozen | Airplane | 0.790 | 0.093 | 100.0% | 91.8% |

**Key Observations:**
- **VGGInspired** achieves the best overall bounding box performance with 100% good detection rate and 97.1% very good detection rate, surpassing all other models
- **DeeperDetector** shows excellent performance with 100% good detection rate and 91.2% very good detection rate
- **ResNet Unfrozen** demonstrates the significant impact of transfer learning fine-tuning compared to frozen features
- **ResNet Frozen** shows notably poor performance (47.1% very good detection rate), highlighting the importance of unfreezing backbone features for bbox regression
- **SimpleDetector** shows good performance but struggles significantly with face localization (31.6% very good detection rate for faces)
- **VGGInspired** achieves perfect airplane localization (100% very good detection rate) and excellent face detection (91.2% very good detection rate)
- **Transfer Learning Impact**: Unfrozen ResNet (78.9% very good detection) vs Frozen ResNet (47.1% very good detection) shows a 31.8% improvement
- **Face Localization Challenge**: ResNet Frozen performs particularly poorly on faces (15.8% very good detection) compared to custom architectures
- Custom deeper architectures (VGGInspired, DeeperDetector) consistently outperform transfer learning approaches for bounding box regression tasks

### Note:
The deeper model use lot of energy and have lots of parameters wich lead to lobg training compare to other models trained here.

9. (optional) Download internet images and run them through the network. 
   Show a few success and failure cases and comment.

For the bounding box detection with the model: Resnet (unfrozen) this image does not properly work:
![error plane](doc/error_plane.jpg)
In fact the proper type has been found but the bouding box does not make sense.
All the other picture are good (good bounding boxe and proper class)
To test a specific image you can run :
```bash
uv run python predict.py --filename web_images/images/airplane/web_test_plane_2.jpg --model best --save-file true --output-file "doc/error_plane.jpg"
```
and for classic prediction
```bash
uv run python predict.py --model best --directory output/test_data.csv
```


## Part 4: Comparison with classic approaches (optional)

- Document yourself with OpenCv classic object and face detection approaches.
Execute them on the provided dataset. Compare with ground truth and compute the same loss
that you used for your Networks to have comparable error analysis.
Compare the different approaches with your learning approaches
by putting extra table entries in your error analysis table.

- Check out [Tina Face](https://paperswithcode.com/paper/tinaface-strong-but-simple-baseline-for-face)
  and [Yolo](https://paperswithcode.com/paper/you-only-look-once-unified-real-time-object) architectures and note the specificities.


