#include "T.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <cmath>

T::T()
{
    numImages = 0;
    numPixels = 784;
}

T::T(int n, int d)
{
    numImages = n;
    numPixels = d;
    for (int i = 0; i < numImages; i++)
    {
        for (int j = 0; j < numPixels; j++)
        {
            images[i].push_back(0);
        }
    }
}

void T::setPixel(int imageIndex, int pixelIndex, int pixelValue)
{
    if (imageIndex >= 0 && imageIndex < numImages && pixelIndex >= 0 && pixelIndex < numPixels)
    {
        images[imageIndex][pixelIndex] = pixelValue;
    }
}

int T::getPixel(int imageIndex, int pixelIndex) const
{
    if (imageIndex >= 0 && imageIndex < numImages && pixelIndex >= 0 && pixelIndex < numPixels)
    {
        return images[imageIndex][pixelIndex];
    }
    return 0; // Default value if index is out of range
}

void T::setImage(std::vector<int> images)
{
    this->images.push_back(images);
    numImages++;
}

bool T::load(std::string filepath)
{
    std::ifstream file(filepath);
    if (!file)
    {
        std::cerr << "Failed to load KNNClassifier from file." << std::endl;
        return false;
    }

    std::string line;
    getline(file, line); // skip the rest of the line

    images.clear();
    trainLabels.clear();


    // Load test images
    while (std::getline(file, line) && !line.empty())
    {
        std::vector<int> image;
        std::stringstream ss(line);
        int pixel;
        std::string pixel1;
        int lbl = 0;
        ss >> pixel;
        trainLabels.push_back(pixel);

        while (std::getline(ss, pixel1, ','))
        {
            try {
                if (isdigit(pixel1[0]))
                {
                    int data = std::stoi(pixel1);
                    image.push_back(data);
                }
            }
            catch (const std::exception& e)
            {
                std::cout << "Error converting string to int: " << e.what() << std::endl;
            }
        }

        images.push_back(image);
        numImages++;
    }

    file.close();
    return true;
}