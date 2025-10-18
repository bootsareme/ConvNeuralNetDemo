#include <algorithm>

#include "Util.h"
#include "CNN.h"

std::vector<std::vector<std::vector<double>>> CNN::convolve(const std::vector<std::vector<int>>& input)
{
    std::vector<std::vector<std::vector<double>>> convolvedFeatureMaps;
    std::vector<std::vector<double>> map1, map2, map3;

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
        map1.push_back(row);
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
        map2.push_back(row);
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
        map3.push_back(row);
    }

    convolvedFeatureMaps.push_back(map1);
    convolvedFeatureMaps.push_back(map2);
    convolvedFeatureMaps.push_back(map3);
    return convolvedFeatureMaps;
}

void CNN::ReLU(std::vector<std::vector<double>>& convLayer)
{
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            convLayer[i][j] = std::max(0, convLayer[i][j]);
}

std::vector<std::vector<std::vector<double>>> CNN::pool(const std::vector<std::vector<std::vector<double>>>& convLayer)
{
    std::vector<std::vector<std::vector<double>>> featureMaps;
    std::vector<std::vector<double>> map1, map2, map3;

    // pool first layer
    for (int i = 0; i < 6; i += 2)
    {
        // stride size of 2
        std::vector<double> row;
        for (int j = 0; j < 6; j += 2) // pooling by taking highest value in 2x2 window
            row.emplace_back(std::max({ convLayer[0][i][j], convLayer[0][i][j + 1], convLayer[0][i + 1][j], convLayer[0][i + 1][j + 1] }));
        
        // account for the fact that a 2x2 window cannot scan a 7x7 with stride-size 2 evenly
        row.emplace_back(std::max({ convLayer[0][i][5], convLayer[0][i][6], convLayer[0][i + 1][5], convLayer[0][i + 1][6] }));
        map1.emplace_back(row);
    }

    // pool second layer
    for (int i = 0; i < 6; i += 2)
    {
        // stride size of 2
        std::vector<double> row;
        for (int j = 0; j < 6; j += 2) // pooling by taking highest value in 2x2 window
            row.emplace_back(std::max({ convLayer[1][i][j], convLayer[1][i][j + 1], convLayer[1][i + 1][j], convLayer[1][i + 1][j + 1] }));
        
        // account for the fact that a 2x2 window cannot scan a 7x7 with stride-size 2 evenly
        row.emplace_back(std::max({ convLayer[1][i][5], convLayer[1][i][6], convLayer[1][i + 1][5], convLayer[1][i + 1][6] }));
        map2.emplace_back(row);
    }

    // pool third layer
    for (int i = 0; i < 6; i += 2)
    {
        // stride size of 2
        std::vector<double> row;
        for (int j = 0; j < 6; j += 2) // pooling by taking highest value in 2x2 window
            row.emplace_back(std::max({ convLayer[2][i][j], convLayer[2][i][j + 1], convLayer[2][i + 1][j], convLayer[2][i + 1][j + 1] }));
        
        // account for the fact that a 2x2 window cannot scan a 7x7 with stride-size 2 evenly
        row.emplace_back(std::max({ convLayer[2][i][5], convLayer[2][i][6], convLayer[2][i + 1][5], convLayer[2][i + 1][6] }));
        map3.emplace_back(row);
    }

    map1.push_back(std::vector<double>{
        std::max({ convLayer[0][5][0], convLayer[0][5][1], convLayer[0][6][0], convLayer[0][6][1] }),
        std::max({ convLayer[0][5][2], convLayer[0][5][3], convLayer[0][6][2], convLayer[0][6][3] }),
        std::max({ convLayer[0][5][4], convLayer[0][5][5], convLayer[0][6][4], convLayer[0][6][5] }),
        std::max({ convLayer[0][5][5], convLayer[0][5][6], convLayer[0][6][5], convLayer[0][6][6] }),
    });

    map2.push_back(std::vector<double>{
        std::max({ convLayer[1][5][0], convLayer[1][5][1], convLayer[1][6][0], convLayer[1][6][1] }),
        std::max({ convLayer[1][5][2], convLayer[1][5][3], convLayer[1][6][2], convLayer[1][6][3] }),
        std::max({ convLayer[1][5][4], convLayer[1][5][5], convLayer[1][6][4], convLayer[1][6][5] }),
        std::max({ convLayer[1][5][5], convLayer[1][5][6], convLayer[1][6][5], convLayer[1][6][6] }),
    });

    map3.push_back(std::vector<double>{
        std::max({ convLayer[2][5][0], convLayer[2][5][1], convLayer[2][6][0], convLayer[2][6][1] }),
        std::max({ convLayer[2][5][2], convLayer[2][5][3], convLayer[2][6][2], convLayer[2][6][3] }),
        std::max({ convLayer[2][5][4], convLayer[2][5][5], convLayer[2][6][4], convLayer[2][6][5] }),
        std::max({ convLayer[2][5][5], convLayer[2][5][6], convLayer[2][6][5], convLayer[2][6][6] }),
    });

    featureMaps.push_back(map1);
    featureMaps.push_back(map2);
    featureMaps.push_back(map3);
    return featureMaps;
}
