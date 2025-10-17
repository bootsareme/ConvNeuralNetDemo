/*
 * cnn_demo.cpp : This standalone file will implement a basic convolutional neural network from the ground up in C++.
 * ============================================================================================================================
 * AUTHOR: Vincent Zhang
 * DATE: 2022-1-7
 * PURPOSE: To demonstrate the internal workings of a CNN as many modern libraries abstract away fundamental concepts.
 * GOAL: Build a convolutional network to recognize whether a 9x9 pixelated black-and-white image contains a drawing of a "X" or "O".
 * NOTE: This CNN works as intended if drawings are close to the center of the canvas. Because it does not go through generational epochs, the accuracy of this CNN is still very low as the training set is only based on 1 perfect drawing of 'X' or 'O'. It is designed to demonstrate the inner-workings of a CNN, not serve as scalable, optimized algorithm.
 */

#include <iostream>
#include <iomanip> 
#include <vector>
#include <algorithm> 

/* core CNN functions */
namespace CNN
{
    const static std::vector<std::vector<int>> filterDiagonal1 = { {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1} },
        filterDiagonal2 = { {-1, -1, 1}, {-1, 1, -1}, {1, -1, -1} },
        filterCenter = { {1, -1, 1}, {-1, 1, -1}, {1, -1, 1} };

    /// <summary> Applies convolutional layer on drawing. </summary>
    ///     <param name="input"> Raw data of drawing. </param>
    ///     <returns> 3D vector containing 3 2D vectors that represents each layer applied on each filter. </returns>
    std::vector<std::vector<std::vector<double>>> convolute(const std::vector<std::vector<int>>& input)
    {
        std::vector<std::vector<std::vector<double>>> convolutedLayers;
        std::vector<std::vector<double>> firstLayer, secondLayer, thirdLayer;

        // apply first filter
        for (int i = 1; i < 8; ++i)
        {
            std::vector<double> row;
            for (int j = 1; j < 8; ++j)
            {
                // slide filter through input layer and take the average dot product (sum of filter multiplied by data divided by filter size)
                const double dotProd1 = input[i - 1][j - 1] * filterDiagonal1[0][0] + input[i - 1][j] * filterDiagonal1[0][1] + input[i - 1][j + 1] * filterDiagonal1[0][2];
                const double dotProd2 = input[i][j - 1] * filterDiagonal1[1][0] + input[i][j] * filterDiagonal1[1][1] + input[i][j + 1] * filterDiagonal1[1][2];
                const double dotProd3 = input[i + 1][j - 1] * filterDiagonal1[2][0] + input[i + 1][j] * filterDiagonal1[2][1] + input[i + 1][j + 1] * filterDiagonal1[2][2];
                row.push_back((dotProd1 + dotProd2 + dotProd3) / 9);
            }
            firstLayer.push_back(row);
        }

        // apply second filter
        for (int i = 1; i < 8; ++i)
        {
            std::vector<double> row;
            for (int j = 1; j < 8; ++j)
            {
                // slide filter through input layer and take the average dot product (sum of filter multiplied by data divided by filter size)
                const double dotProd1 = input[i - 1][j - 1] * filterDiagonal2[0][0] + input[i - 1][j] * filterDiagonal2[0][1] + input[i - 1][j + 1] * filterDiagonal2[0][2];
                const double dotProd2 = input[i][j - 1] * filterDiagonal2[1][0] + input[i][j] * filterDiagonal2[1][1] + input[i][j + 1] * filterDiagonal2[1][2];
                const double dotProd3 = input[i + 1][j - 1] * filterDiagonal2[2][0] + input[i + 1][j] * filterDiagonal2[2][1] + input[i + 1][j + 1] * filterDiagonal2[2][2];
                row.push_back((dotProd1 + dotProd2 + dotProd3) / 9);
            }
            secondLayer.push_back(row);
        }

        // apply third filter
        for (int i = 1; i < 8; ++i)
        {
            std::vector<double> row;
            for (int j = 1; j < 8; ++j)
            {
                // slide filter through input layer and take the average dot product (sum of filter multiplied by data divided by filter size)
                const double dotProd1 = input[i - 1][j - 1] * filterCenter[0][0] + input[i - 1][j] * filterCenter[0][1] + input[i - 1][j + 1] * filterCenter[0][2];
                const double dotProd2 = input[i][j - 1] * filterCenter[1][0] + input[i][j] * filterCenter[1][1] + input[i][j + 1] * filterCenter[1][2];
                const double dotProd3 = input[i + 1][j - 1] * filterCenter[2][0] + input[i + 1][j] * filterCenter[2][1] + input[i + 1][j + 1] * filterCenter[2][2];
                row.push_back((dotProd1 + dotProd2 + dotProd3) / 9);
            }
            thirdLayer.push_back(row);
        }

        convolutedLayers.push_back(firstLayer);
        convolutedLayers.push_back(secondLayer);
        convolutedLayers.push_back(thirdLayer);
        return convolutedLayers;
    }

    /// <summary> ReLU function Layer, removes all negative numbers and sets them to zero. </summary>
    ///     <param name="convLayer"> Individual 2D convoluted layer. Modifies in-place. </param>
    void ReLU(std::vector<std::vector<double>>& convLayer)
    {
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                if (convLayer[i][j] < 0)
                    convLayer[i][j] = 0;
    }

    /// <summary> Pooling layer shrinks the image stack into a smaller size for the feed-forward network to digest. </summary>
    ///     <param name="convLayers"> Vector of all 2D convoluted layers. </param>
    /// <returns> Shrunken down layers. </returns>
    std::vector<std::vector<std::vector<double>>> pool(const std::vector<std::vector<std::vector<double>>>& convLayers)
    {
        std::vector<std::vector<std::vector<double>>> convolutedLayers;
        std::vector<std::vector<double>> firstLayer, secondLayer, thirdLayer;

        // pool first layer
        for (int i = 0; i < 6; i += 2)
        {
            // stride size of 2
            std::vector<double> row;
            for (int j = 0; j < 6; j += 2)
            {
                // pooling by taking highest value in 2x2 window
                row.emplace_back(std::max({ convLayers[0][i][j], convLayers[0][i][j + 1], convLayers[0][i + 1][j], convLayers[0][i + 1][j + 1] }));
            }
            // account for the fact that a 2x2 window cannot scan a 7x7 with stride-size 2 evenly
            row.emplace_back(std::max({ convLayers[0][i][5], convLayers[0][i][6], convLayers[0][i + 1][5], convLayers[0][i + 1][6] }));
            firstLayer.emplace_back(row);
        }

        // pool second layer
        for (int i = 0; i < 6; i += 2)
        {
            // stride size of 2
            std::vector<double> row;
            for (int j = 0; j < 6; j += 2)
            {
                // pooling by taking highest value in 2x2 window
                row.emplace_back(std::max({ convLayers[1][i][j], convLayers[1][i][j + 1], convLayers[1][i + 1][j], convLayers[1][i + 1][j + 1] }));
            }
            // account for the fact that a 2x2 window cannot scan a 7x7 with stride-size 2 evenly
            row.emplace_back(std::max({ convLayers[1][i][5], convLayers[1][i][6], convLayers[1][i + 1][5], convLayers[1][i + 1][6] }));
            secondLayer.emplace_back(row);
        }

        // pool third layer
        for (int i = 0; i < 6; i += 2)
        {
            // stride size of 2
            std::vector<double> row;
            for (int j = 0; j < 6; j += 2)
            {
                // pooling by taking highest value in 2x2 window
                row.emplace_back(std::max({ convLayers[2][i][j], convLayers[2][i][j + 1], convLayers[2][i + 1][j], convLayers[2][i + 1][j + 1] }));
            }
            // account for the fact that a 2x2 window cannot scan a 7x7 with stride-size 2 evenly
            row.emplace_back(std::max({ convLayers[2][i][5], convLayers[2][i][6], convLayers[2][i + 1][5], convLayers[2][i + 1][6] }));
            thirdLayer.emplace_back(row);
        }

        firstLayer.push_back(std::vector<double>{
            std::max({ convLayers[0][5][0], convLayers[0][5][1], convLayers[0][6][0], convLayers[0][6][1] }),
            std::max({ convLayers[0][5][2], convLayers[0][5][3], convLayers[0][6][2], convLayers[0][6][3] }),
            std::max({ convLayers[0][5][4], convLayers[0][5][5], convLayers[0][6][4], convLayers[0][6][5] }),
            std::max({ convLayers[0][5][5], convLayers[0][5][6], convLayers[0][6][5], convLayers[0][6][6] }),
        });

        secondLayer.push_back(std::vector<double>{
            std::max({ convLayers[1][5][0], convLayers[1][5][1], convLayers[1][6][0], convLayers[1][6][1] }),
            std::max({ convLayers[1][5][2], convLayers[1][5][3], convLayers[1][6][2], convLayers[1][6][3] }),
            std::max({ convLayers[1][5][4], convLayers[1][5][5], convLayers[1][6][4], convLayers[1][6][5] }),
            std::max({ convLayers[1][5][5], convLayers[1][5][6], convLayers[1][6][5], convLayers[1][6][6] }),
        });

        thirdLayer.push_back(std::vector<double>{
            std::max({ convLayers[2][5][0], convLayers[2][5][1], convLayers[2][6][0], convLayers[2][6][1] }),
            std::max({ convLayers[2][5][2], convLayers[2][5][3], convLayers[2][6][2], convLayers[2][6][3] }),
            std::max({ convLayers[2][5][4], convLayers[2][5][5], convLayers[2][6][4], convLayers[2][6][5] }),
            std::max({ convLayers[2][5][5], convLayers[2][5][6], convLayers[2][6][5], convLayers[2][6][6] }),
        });

        convolutedLayers.push_back(firstLayer);
        convolutedLayers.push_back(secondLayer);
        convolutedLayers.push_back(thirdLayer);
        return convolutedLayers;
    }
}

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
    
    std::cout << "\nConvoluting each layer using filters...\n";
    std::vector<std::vector<std::vector<double>>> convolutedLayers = CNN::convolute(drawing);
    std::cout << "\nConvoluted Layer 1 (7x7):\n";
    Util::print_2Dvector(convolutedLayers[0]);
    std::cout << "\nConvoluted Layer 2 (7x7):\n";
    Util::print_2Dvector(convolutedLayers[1]);
    std::cout << "\nConvoluted Layer 3 (7x7):\n";
    Util::print_2Dvector(convolutedLayers[2]);

    for (int i = 0; i < 3; ++i)
        CNN::ReLU(convolutedLayers[i]);

    std::cout << "\nApplying ReLU activation layer...\n";
    std::cout << "\nReLU Layer 1 (7x7):\n";
    Util::print_2Dvector(convolutedLayers[0]);
    std::cout << "\nReLU Layer 2 (7x7):\n";
    Util::print_2Dvector(convolutedLayers[1]);
    std::cout << "\nReLU Layer 3 (7x7):\n";
    Util::print_2Dvector(convolutedLayers[2]);
    
    std::cout << "\nPooling layers with window size 2 and stride size 2...\n";
    const std::vector<std::vector<std::vector<double>>> pooledLayers = CNN::pool(convolutedLayers);
    std::cout << "\nReLU Layer 1 with Pooling (4x4):\n";
    Util::print_2Dvector(pooledLayers[0]);
    std::cout << "\nReLU Layer 2 with Pooling (4x4):\n";
    Util::print_2Dvector(pooledLayers[1]);
    std::cout << "\nReLU Layer 3 with Pooling (4x4):\n";
    Util::print_2Dvector(pooledLayers[2]);
    
    std::cout << "\nConverting 3 layers into 1 layer...\n";
    std::cout << "\nFeed-forward network layer (48x1):\n";
    std::vector<double> singleLayer;

    for (const auto& layer : pooledLayers)
        for (const auto& vector : layer)
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
