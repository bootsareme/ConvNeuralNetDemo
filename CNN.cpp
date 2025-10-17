#include <algorithm>

#include "Util.h"
#include "CNN.h"

std::vector<std::vector<std::vector<double>>> CNN::convolve(const std::vector<std::vector<int>>& input)
{
    std::vector<std::vector<std::vector<double>>> convolvedLayers;
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

    convolvedLayers.push_back(firstLayer);
    convolvedLayers.push_back(secondLayer);
    convolvedLayers.push_back(thirdLayer);
    return convolvedLayers;
}

void CNN::ReLU(std::vector<std::vector<double>>& convLayer)
{
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            if (convLayer[i][j] < 0)
                convLayer[i][j] = 0;
}

std::vector<std::vector<std::vector<double>>> CNN::pool(const std::vector<std::vector<std::vector<double>>>& convLayers)
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