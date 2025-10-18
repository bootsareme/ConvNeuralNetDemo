/*
 * cnn_demo.cpp : This standalone file will implement a basic convolutional neural network from the ground up in C++.
 * ============================================================================================================================
 * AUTHOR: Vincent Zhang
 * DATE: 2022-1-7
 * PURPOSE: To demonstrate the internal workings of a CNN as many modern libraries abstract away fundamental concepts.
 * GOAL: Build a convolutional network to recognize whether a 9x9 pixelated black-and-white image contains a drawing of a "X" or "O".
 * NOTE: This CNN works as intended if drawings are close to the center of the canvas. Because it does not go through backpropagation, the accuracy of this CNN is still fairly low as the training set is only based on 1 perfect drawing of 'X' or 'O'. It is designed to demonstrate the inner-workings of a CNN and not serve as scalable, optimized algorithm.
 */

#include <iostream>
#include <iomanip> 
#include <vector>

#include "Util.h"
#include "CNN.h"

int main()
{
    std::cout << std::setprecision(2);
    const std::vector<std::vector<int>> drawing = Util::parse_file("canvas.txt");
    
    std::cout << "C++ Convolutional Neural Network that classifies whether a 9x9 drawing is a 'X' or 'O'.\n";
    std::cout << "View input drawing in 'canvas.txt' for reference.\n";
    
    std::cout << "\nDrawing converted to computer-readable matrix (9x9): \n";
    Util::print_2Dvector(drawing);
    
    std::cout << "\nApplying 3 filters to data...\n";
    std::cout << "\nDiagonal top-left to bottom-right (3x3):\n";
    Util::print_2Dvector(CNN::filterDiagonal1);
    std::cout << "\nDiagonal top-right to bottom-left (3x3):\n";
    Util::print_2Dvector(CNN::filterDiagonal2);
    std::cout << "\nCenterpiece (3x3):\n";
    Util::print_2Dvector(CNN::filterCenter);
    
    std::cout << "\nConvolving original input using filters...\n";
    std::vector<std::vector<std::vector<double>>> featureMaps = CNN::convolve(drawing);
    std::cout << "\nConvolved feature map 1 (7x7):\n";
    Util::print_2Dvector(featureMaps[0]);
    std::cout << "\nConvolved feature map 2 (7x7):\n";
    Util::print_2Dvector(featureMaps[1]);
    std::cout << "\nConvolved feature map 3 (7x7):\n";
    Util::print_2Dvector(featureMaps[2]);

    for (int i = 0; i < 3; ++i)
        CNN::ReLU(featureMap[i]);

    std::cout << "\nApplying ReLU activation layer...\n";
    std::cout << "\nReLU feature map 1 (7x7):\n";
    Util::print_2Dvector(featureMaps[0]);
    std::cout << "\nReLU feature map 2 (7x7):\n";
    Util::print_2Dvector(featureMaps[1]);
    std::cout << "\nReLU feature map 3 (7x7):\n";
    Util::print_2Dvector(featureMaps[2]);
    
    std::cout << "\nPooling layers with window size = 2 and stride size = 2...\n";
    const std::vector<std::vector<std::vector<double>>> pooledLayer = CNN::pool(featureMaps);
    std::cout << "\nReLU Layer 1 with Pooling (4x4):\n";
    Util::print_2Dvector(pooledLayer[0]);
    std::cout << "\nReLU Layer 2 with Pooling (4x4):\n";
    Util::print_2Dvector(pooledLayer[1]);
    std::cout << "\nReLU Layer 3 with Pooling (4x4):\n";
    Util::print_2Dvector(pooledLayer[2]);
    
    std::cout << "\nFlattening into 1 layer...\n";
    std::cout << "\nFeed-forward network layer (1x48):\n";
    std::vector<double> singleLayer;

    for (const auto& featureMap : pooledLayer)
        for (const auto& vector : featureMap)
            for (auto value : vector)
                singleLayer.push_back(value);

    for (const double d : singleLayer)
        std::cout << d << '\n';

    // find all of the most weighted nodes in the perfect 'X' or 'O' and see how new input compares to the perfect model
    std::cout << "\nMaking final predictions and correlations...\n";
    double xPrediction = 0, oPrediction = 0;
    constexpr short xNodes[13] = { 0, 5, 10, 11, 14, 15, 18, 19, 22, 24, 25, 28, 37 },
                    oNodes[9] = { 2, 3, 8, 12, 16, 26, 27, 30, 31 };

    for (const short& i : xNodes)
        xPrediction += singleLayer[i];

    for (const short& i : oNodes)
        oPrediction += singleLayer[i];

    xPrediction = xPrediction / 13 * 100;
    oPrediction = oPrediction / 9 * 100;
    
    std::cout << "\nConfidence that original drawing is 'X': " << std::fixed << xPrediction << "%\n";
    std::cout << "Confidence that original drawing is 'O': " << std::fixed << oPrediction << "%\n";

    if (xPrediction > oPrediction)
        std::cout << "\nFINAL RESULT: The CNN predicts that the original drawing resembles a 'X'.\n";
    else
        std::cout << "\nFINAL RESULT: The CNN predicts that the original drawing resembles a 'O'.\n";
}
