# Fashion-Classification
A ML and DS project on fashion products classification based on MNIST fashion dataset.


## Table of Contents

- [Fashion-Classification](#fashion-classification)
  - [Table of Contents](#table-of-contents)
  - [About the Project](#about-the-project)
  - [Dataset](#dataset)
  - [Project Structure](#project-structure)
  - [Dependencies](#dependencies)
  - [Model](#model)
  - [How to run (Windows)](#how-to-run-windows)


## About the Project

The goal of this project is to classify fashion products based on the MNIST fashion dataset. The dataset contains 10 classes of fashion products.


## Dataset

The dataset used for this project is the [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)


## Project Structure

The models folder contains the trained model. The notebooks folder contains the exploratory data analysis and model training notebooks. The `evaluate_model.py` script is used to evaluate the model on the test set and save the output in the `output.txt` file.

```
Fashion-Classification
│
├── dataset
│   └── fashion-mnist_test.csv
│
├── models
│   └── model_v2.pkl
│
├── notebooks
│   ├── model_training.ipynb
│   └── exploratory_data_analysis.ipynb
│
├── evaluate_model.py
├── model_summary.txt
├── output.txt
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```


## Dependencies

- fastai
- scikit-learn


## Model

The model used for this project is a ResNet34 model with custom head layer which is trained and dine-tuned on the Fashion MNIST dataset. The model was trained for 5 epochs with a batch size of 32. The model achieved an accuracy of 89.9% on the test set.


## How to run (Windows)

1. Clone the repository and navigate to the project directory.

```bash
git clone https://github.com/abir0/Fashion-Classification.git
cd Fashion-Classification
```

2. Create a virtual environment and activate it.

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install the dependencies

```bash
pip install -r requirements.txt
```

1. Run the `evaluate_model.py` script

```bash
python evaluate_model.py
```

5. The output will be saved in the `output.txt` file.

