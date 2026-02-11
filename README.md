# coreML

A collection of Machine Learning algorithms implemented from scratch and using PyTorch, designed for learning and understanding the core concepts of ML and Deep Learning.

## Table of Contents
- [Foundations](#foundations)
- [Deep Learning (Torchy)](#deep-learning-torchy)
- [Resources](#resources)
- [dependencies](#dependencies)
- [Usage](#usage)

## Foundations
The root directory contains Jupyter notebooks demonstrating fundamental ML algorithms. Many of these are implemented from scratch using NumPy to verify understanding of the underlying mathematics.

- **Linear Regression**: `LinearRegression.ipynb`
- **K-Nearest Neighbors (KNN)**: `KNN.ipynb`
- **Support Vector Machine (SVM)**: `SupportVectorMachine.ipynb`
- **Decision Trees**: `ClassificationTree_1.ipynb`, `ClassificationTree_2.ipynb`
- **Random Forests**: `RandomForests.ipynb`
- **Gradient Boosting**: `GradientBoost.ipynb`
- **XGBoost**: `XGBoost.ipynb`
- **Gradient Descent**: `GradientDescent.ipynb`
- **Neural Networks**: `NeuralNetwork_1.ipynb`
- **LSTM**: `LSTM.ipynb`

## Deep Learning (Torchy)
The `torchy/` directory focuses on Deep Learning concepts using PyTorch.

- **PyTorch Basics**: `torchy/00.ipynb`
- **Workflow**: `torchy/01.ipynb`
- **Computer Vision**: `torchy/03.ipynb`
- **CNNs**: `torchy/CNN.ipynb`
- **Logistic Regression**: `torchy/LogisticRegression.ipynb`

## Resources
- **Papers**: `paper/` contains reference papers like `AlexNet.pdf`.
- **Data**: Datasets used in the notebooks are located in `data/` and `dataset/`.

## Dependencies
To run these notebooks, you will need the following Python libraries:
- numpy
- pandas
- matplotlib
- scikit-learn
- torch (for `torchy/` notebooks)
- jupyter

## Usage
1. Clone the repository.
2. Navigate to the directory.
3. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open any `.ipynb` file to explore the implementation.
