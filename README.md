# MNIST Model Comparison

This project compares the performance of three deep learning models on the MNIST handwritten digit dataset:

- Perceptron
- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)

The models are trained and evaluated on the same dataset to compare their classification accuracy.

## Dataset
- MNIST Handwritten Digits Dataset
- 70,000 grayscale images of handwritten digits (0–9)
- Image size: 28 × 28 pixels

## Models Used
- Perceptron
- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)

## Data Preprocessing
- Train-test split
- Pixel value normalization (0–255 → 0–1)
- One-hot encoding of labels
- Reshaping images for CNN input

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras

## Results
The CNN model achieved the highest accuracy among all three models.

## Project Structure
```
mnist-model-comparison/
│── mnist_model_comparison.ipynb
│── mnist.csv
└── README.md
```

## Future Improvements
- Hyperparameter tuning
- Data augmentation
- Confusion matrix visualization
- Model saving and loading
