#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <string>

/* Utility functions for reading files and printing data to console. */
namespace Util
{
    /// <summary> Utility function to print 2D vectors. </summary>
    /// <param name="vector2D"> 2D vector to be printed. </param>
    template <typename T>
    void print_2Dvector(const std::vector<std::vector<T>>& vector2D)
    {
        for (const std::vector<T>& components : vector2D)
        {
            for (T element : components)
                std::cout << std::right << std::setw(6) << element;
            std::cout << '\n';
        }
    }

    /// <summary>
    ///     Converts human-readable drawing into 2D vectors.
    ///     Use "@" for a shaded black pixel, "." for a empty white pixel.
    ///     Put 1 drawing only, additional drawings will be ignored. Do not go above 9x9.
    /// </summary>
    ///     <param name="file"> name of the file </param>
    ///     <returns> 2D vector of machine-readable format of image </returns>
    inline std::vector<std::vector<int>> parse_file(const char* file)
    {
        std::ifstream input(file);
        std::vector<std::vector<int>> singleDrawing;

        for (std::string line; std::getline(input, line);)
        {
            std::vector<int> row;
            for (const char c : line)
            {
                // if pixel is shaded, set to 1; blank pixels are -1
                switch (c)
                {
                case '\r':
                    continue;
                case '@':
                    row.emplace_back(1);
                    break;
                case '.':
                    row.emplace_back(-1);
                    break;
                default:
                    throw std::logic_error("The drawing has an invalid format. Please fix the drawing in accordance to specifications.");
                }
            }
            singleDrawing.push_back(row);
        }
        return singleDrawing;
    }
}
