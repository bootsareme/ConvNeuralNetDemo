#pragma once
#include <vector>

/* core CNN functions */
namespace CNN
{
    const static std::vector<std::vector<int>> filterDiagonal1 = { 
        {1, -1, -1}, 
        {-1, 1, -1}, 
        {-1, -1, 1}
    },
    filterDiagonal2 = { 
        {-1, -1, 1}, 
        {-1, 1, -1}, 
        {1, -1, -1} 
    },
    filterCenter = { 
        {1, -1, 1}, 
        {-1, 1, -1}, 
        {1, -1, 1} 
    };

    /// <summary> Applies convolutional layer on drawing. </summary>
    ///     <param name="input"> Raw data of drawing. </param>
    ///     <returns> 3D vector containing 3 2D vectors that represents each feature map applied on each filter. </returns>
    std::vector<std::vector<std::vector<double>>> convolve(const std::vector<std::vector<int>>& input);

     /// <summary> ReLU function Layer, removes all negative numbers and sets them to zero. </summary>
    ///     <param name="convLayer"> Individual 2D convolved feature maps. Modifies in-place. </param>
    void ReLU(std::vector<std::vector<double>>& convLayer);

    /// <summary> Pooling layer shrinks the image stack into a smaller size for the feed-forward network to digest. </summary>
    ///     <param name="convLayers"> Vector of all 2D convolved feature maps. </param>
    ///     <returns> Shrunken down feature maps. </returns>
    std::vector<std::vector<std::vector<double>>> pool(const std::vector<std::vector<std::vector<double>>>& featureMaps);
}
