# LwF-Project

Final project exam of Machine Learning, University of Trento, 2023-24.

## Table of Contents

- [General Info](#general-info)
- [Project Description](#project-description)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Setup Instructions](#setup-instructions)
  - [Running Locally](#running-locally)
  - [Running on Google Colab](#running-on-google-colab)
- [References](#references)

## General Info

This project is based on the paper [Learning Without Forgetting](https://arxiv.org/abs/1606.09282) by Zhizhong Li and Derek Hoiem (2016).

## Project Description

The main objective of this project was to implement a continual learning paradigm using the Learning Without Forgetting (LwF) approach. The goal was to demonstrate the effectiveness of LwF in handling both single and multiple new tasks, leveraging a pre-trained AlexNet model.

### Key Assignments:

1. **Continual Learning Implementation**: 
   - Implement the Learning Without Forgetting (LwF) approach to perform continual learning.

2. **Model and Pretraining**:
   - Use an AlexNet model pretrained on the Places365 dataset as the base model.

3. **New Task Scenarios**:
   - **Single New Task**: Demonstrate the model's capability on a single new task using the CUB-200-2011 dataset. This dataset contains 200 categories of birds with a total of 11,788 images, focusing on fine-grained visual categorization.
   - **Multiple New Tasks**: Demonstrate the model's capability on multiple new tasks using the PASCAL VOC 2012 dataset. This dataset consists of 20 object classes, used for object detection, classification, and segmentation tasks.

4. **Evaluation**:
   - Evaluate the model's performance in both single and multiple task scenarios, ensuring that the model retains its ability to perform on the original tasks while adapting to new ones.

## Implementation Details

In this project, the AlexNet model, pre-trained on the Places365 dataset, was employed to perform continual learning using the Learning Without Forgetting (LwF) strategy. The model was tested on two different scenarios:

- **Single Task Scenario**: The CUB-200-2011 dataset was used as the new task. This dataset presents a challenging scenario with fine-grained categories, requiring the model to adapt to a single, highly specific task.

- **Multiple Task Scenario**: The PASCAL VOC 2012 dataset was used to introduce multiple new tasks to the model. This scenario tested the model's ability to handle several tasks simultaneously, such as object detection and classification across various classes.

### Challenges

The implementation faced challenges related to the computational demands of training deep learning models in a continual learning context. Given the limited computational resources, the training process was conducted using Google Colab, which provides a free GPU with limited continuous usage.

## Setup Instructions

There are two ways to run the code:

### Running Locally

- Detailed instructions for setting up and running the project on a local machine.
  
#### 1. Clone the Repository

First, clone the repository to your local machine using Git:

```bash
git clone https://github.com/federicolombo/LwF_exam_ML.git
cd your-repository-name
```

### 2. Set Up the Conda Environment

Make sure you have Conda installed on your machine. If you don’t have Conda installed, you can download it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual).

Once Conda is installed, you can create the environment using the provided `.yml` file. This file contains all the dependencies needed for the project.

### 3. Prepare the Datasets

Make sure that the datasets (VOC and CUB-200-2011) are downloaded and properly set up. You may need to modify the paths in your code to point to the location of the datasets on your local machine.

- **PASCAL VOC 2012**: You can download it from the [official site](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
- **CUB-200-2011**: You can download it from the [official dataset site](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).

Once downloaded, extract the datasets to your desired location, and update the dataset paths in your code if needed.

### 4. Running the Training Script

With the environment set up and the datasets ready, you can now run the training script. Here’s an example command:

```bash
python train.py --epochs 100 --batch_size 64 --learning_rate 0.001 --model_to_train voc
```
5. Additional Arguments
   
You can modify other parameters as needed by including additional arguments or changing the existing ones. To see all available options, you might want to check if the script provides a help command:

```bash
python train.py --help
```

### Running on Google Colab

- We recommend using Google Colab due to its ease of setup and faster execution.
[![🚀 Open in Colab](https://img.shields.io/badge/Open%20in-Google%20Colab-orange?logo=google-colab&style=for-the-badge)](https://colab.research.google.com/drive/1IpjkPncag0BgxxcclQEeoXXXiL59-mfK#scrollTo=ukqGY89_c3X9)

## References

[1] Li, Zhizhong, and Derek Hoiem. "Learning Without Forgetting." arXiv preprint arXiv:1606.09282 (2016).
