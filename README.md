# ConvNeuralNetDemo
A simple C++ that project implements a entire convolutional neural network from scratch. 

Originally created in early 2022, this project aims to demonstrate the internal workings of a convolutional neural network (CNN), since many modern libraries tend to abstract away these foundational concepts. The goal is to build a simple CNN capable of recognizing whether a $9 \times 9$ ASCII-based pixelated image contains a drawing of an `X` or `O`. The network performs as intended when the drawings are centered on the canvas, but its accuracy remains low because it does not train over multiple epochs and relies on only one ideal example of each symbol. This project is meant as a conceptual and educational tool for how CNNs operates internally, rather than as a scalable or optimized model.

## Usage
The easiest way to see this demo in action is to deploy on a *nix-like OS (such as directly using GitHub Codespaces). Run the following command in the root of this repository to get some output:
```
$ g++ -I. CNN.cpp cnn_demo.cpp && ./a.out
```

## Inner Workings
This section describes the theoretical constructs of this CNN model. To see the steps listed with concrete values, run the program and it will fill in each layer with actual values. The overall CNN architecture is shown below:

<img width="540" height="169" alt="image" src="https://github.com/user-attachments/assets/a69625c1-8ac3-4042-94cf-ad8b1bcfff6a" />

Presented below is a technical, layer-by-layer walkthrough of the implemented architecture, systematically tracing each stage as depicted in the diagram:

### I. Input/Preprocessing
Scan through `canvas.txt`, constructing a matrix $A \in \mathbb{R}^{9 \times 9}$ such that:

$$
A_{i,j} = \begin{cases}
1 & \text{if pixel }i, j\text{ is filled in (represented by @)} \\
-1 & \text{if pixel }i, j\text{ is not filled in (represented by .)}
\end{cases}
$$
> [!NOTE]
> The `@` symbol is one of the most visually dense characters available on a standard keyboard, occupying a large proportion of the character cell, whereas the `.` symbol occupies the least area. This makes them convenient choices for representing “shaded” and “unshaded” pixels in ASCII-art-based input matrices.

### II. Convolution Layer
In this step, three distinct convolutional kernels, denoted as $K^{(1)}, K^{(2)}, K^{(3)} \in \mathcal{R}^{3 \times 3}$ are applied to $A$ to extract localized spatial features. $K^{(1)}$ focuses on the diagonal from top-left to bottom-right. $K^{(2)}$ focuses on the diagonal from top-right to bottom-left. $K^{(3)}$ focuses on a central cross-shaped pattern. Generate three convolved feature maps $B^{(1)}, B^{(2)}, B^{(3)} \in \mathbb{R}^{7 \times 7}$ using the following process:

$$
B^{(n)}_{i, j} = \sum_{u=0}^2 \sum_{u=0}^2 A_{i+u,j+v} \cdot K^{(n)}_{u,v},
$$

where $n = 1, 2, 3$ and $0 \leq i, j \leq 6$.

### III. Activation Layer
The *rectified linear unit* activation function is applied to each feature map $n = 1, 2, 3$ in the following fashion:

$$
B^{(n)}_{i, j} := \text{ReLU}(B^{(n)}_{i, j}) = \max(0, B^{(n)}_{i, j}), \text{ for all } 0 \leq i, j \leq 6.
$$

### IV. Pooling Layer
Each feature map is then downsampled to $C^{(1)}, C^{(2)}, C^{(3)} \in \mathbb{R}^{4 \times 4}$ in this layer. Using a window size of 2 and stride 2, apply the max pooling operation on each

$$
C^{(n)}_{i, j} = \max(B^{(n)}_{u, v} | u \in [2i - 1, 2i], v \in [2j - 1, 2j]).
$$

### V. Flatten Layer
Each of the three pooled outputs is flattened and concatenated into a single vector

$$
\vec{x} = \begin{bmatrix} C^{(1)}_{0,0} & \dots & C^{(1)}_{3,3} & C^{(2)}_{0,0} & \dots & C^{(2)}_{3,3} & C^{(3)}_{0,0} & \dots & C^{(3)}_{3,3} \end{bmatrix}^\top \in \mathbb{R}^{48}.
$$

### VI. Pseudo-Dense Output Layer
The flattened $\vec{x} \in \mathbb{R}^{48}$ is mapped to class scores as

$$
y_X = \frac{100}{13} \sum_{i \in \mathcal{I}_X} x_i,
$$
$$
y_O = \frac{100}{9} \sum_{i \in \mathcal{I}_O} x_i,
$$

where $\mathcal{I}_X, \mathcal{I}_O$ correspond to neurons that are maximally activated by a “perfect” `X` or `O` drawing, respectively. By summing these selected neurons, the pseudo-dense layer effectively performs a manual aggregation based on a single canonical sample, rather than learning weights from a dataset. Lastly, the final predicted class is given by:

$$
\hat{y} = \begin{cases}
  X & \text{if } y_X > y_O \\
  O & \text{otherwise}
\end{cases}
$$
