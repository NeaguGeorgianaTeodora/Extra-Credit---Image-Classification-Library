#ifndef EXTRA_CREDIT_IMAGE_CLASSIFICATION_LIBRARY_T_H
#define EXTRA_CREDIT_IMAGE_CLASSIFICATION_LIBRARY_T_H

#pragma once
#include<vector>
#include<string>

class T
{
public:
    std::vector<std::vector<int>> images; // Matrix X to store images
    int numImages; // N: number of images
    int numPixels; // d: number of pixels in an image
    std::vector<int> trainLabels;
    std::string filepath;

public:

    T();

    T(int n, int d);

    void setPixel(int imageIndex, int pixelIndex, int pixelValue);

    int getPixel(int imageIndex, int pixelIndex) const;

    int getNumImages() const { return numImages; }

    int getNumPixels() const { return numPixels; }

    std::vector<std::vector<int>> getImages() const { return images; }

    void setImage(std::vector<int> images);

    std::vector<int> getTrainLabels() const { return trainLabels;}

    bool load(std::string filepath);
};

#endif //EXTRA_CREDIT_IMAGE_CLASSIFICATION_LIBRARY_T_H
