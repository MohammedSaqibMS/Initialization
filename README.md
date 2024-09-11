# Neural Network Initialization and Training 🧠🚀

Welcome to the Neural Network Initialization and Training project! This project demonstrates the importance of initialization methods when training deep learning models. It showcases how different initialization strategies affect the model's learning and performance. We'll cover three key initialization methods:

- **Zero Initialization** ➡️ All weights set to zero.
- **Random Initialization** ➡️ Weights are randomly initialized.
- **He Initialization** ➡️ Weights are initialized using He et al.'s method, suitable for ReLU activations.

### 📂 Dependencies

Before running the code, make sure you have the following libraries installed:

```bash
pip install numpy matplotlib scikit-learn
```

### 🎯 Objective

We aim to train a 3-layer neural network with different initialization methods and analyze the performance on a classification dataset of blue/red points in circles.

---

## 🚀 Model Architecture

The neural network follows the architecture:

```
Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid -> Output
```

### Key Parameters:
- **Learning Rate**: `0.01`
- **Number of Iterations**: `15,000`
- **Initialization Methods**: Zero, Random, He

---

## ⚙️ How to Use

### 1. Load the Dataset 📊

```python
train_X, train_Y, test_X, test_Y = load_dataset()
```
- `train_X`: Training data (features)
- `train_Y`: Training labels (0 or 1)
- `test_X`: Test data (features)
- `test_Y`: Test labels (0 or 1)

### 2. Define the Model 🏗️

We implement a fully connected 3-layer neural network. The key function `model()` handles forward propagation, backward propagation, and parameter updates.

```python
parameters = model(train_X, train_Y, learning_rate=0.01, num_iterations=15000, initialization="he")
```
- `initialization`: Can be set to `"zeros"`, `"random"`, or `"he"`.

---

## 📝 Initialization Methods

### 1. **Zero Initialization** 🛑

In this method, all the weights are initialized to zero. This leads to symmetry, where the neurons learn the same features, and hence, the model will not perform well.

```python
parameters = initialize_parameters_zeros([3, 2, 1])
```

### 2. **Random Initialization** 🎲

In this method, weights are initialized randomly. This helps break symmetry and allows neurons to learn different features.

```python
parameters = initialize_parameters_random([3, 2, 1])
```

### 3. **He Initialization** ⚡

The He initialization method sets weights based on the number of units in the previous layer. This is particularly useful for ReLU activations.

```python
parameters = initialize_parameters_he([3, 2, 1])
```

---

## 🔍 Results

### Zero Initialization:
- **Cost after 15,000 iterations**: 0.693147
- **Accuracy on training set**: 50%
- **Accuracy on test set**: 50%

### Random Initialization:
- **Cost** and **Accuracy** vary with each run, as the weights are randomly initialized.

### He Initialization:
- **Best performance** in terms of faster convergence and higher accuracy.

---

## 📊 Visualization

After training the model, we plot the decision boundary for each initialization method to visualize how well the model has learned to separate the data points:

```python
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
plt.title("Model Decision Boundary with He Initialization")
```

---

## 🏆 Conclusion

- **Zero Initialization**: Poor performance due to weight symmetry.
- **Random Initialization**: Decent performance but not always reliable.
- **He Initialization**: Fast convergence and higher accuracy, recommended for deep networks with ReLU activations.

---

## 🙏 Credits

This project is based on concepts and methodologies taught in the [Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/) by **Andrew Ng** and **DeepLearning.AI**. 

Feel free to explore, modify, and contribute! 😊
