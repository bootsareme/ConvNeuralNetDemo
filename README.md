# ConvNeuralNetDemo
A simple C++ that project implements a entire convolutional neural network from scratch. 

Originally created in early 2022, this project aims to demonstrate the internal workings of a convolutional neural network (CNN), since many modern libraries tend to abstract away these foundational concepts. The goal is to build a simple CNN capable of recognizing whether a $9 \times 9$ ASCII-based pixelated image contains a drawing of an `X` or `O`. The network performs as intended when the drawings are centered on the canvas, but its accuracy remains low because it does not train over multiple epochs and relies on only one ideal example of each symbol. This project is meant as a conceptual and educational tool for how CNNs operates internally, rather than as a scalable or optimized model.

## Usage
The easiest way to see this demo in action is to deploy on a *nix-like OS (such as directly using GitHub Codespaces). Run the following command in the root of this repository to get some output:
```
$ g++ -I. CNN.cpp cnn_demo.cpp
$ ./a.out
```

## Inner Workings
<img width="566" height="200" alt="Untitled drawing" src="https://github.com/user-attachments/assets/47bd4c3b-7bb7-4c0d-8f5f-9ae85e944729" />
