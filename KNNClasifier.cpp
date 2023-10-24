#include "KNNClasifier.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <cmath>


KNNClasifier::KNNClasifier(int k)
{
    this->k = k;
}

// Helper function to calculate distance between two images
double KNNClasifier::calculateDistance(int imageIndex1, int imageIndex2)
{
    double distance = 0.0;
    for (int i = 0; i < trainImages.getNumPixels(); i++)
    {
        int pixel1 = trainImages.getPixel(imageIndex1, i);
        int pixel2 = trainImages.getPixel(imageIndex2, i);
        distance += pow(pixel1 - pixel2, 2);
    }
    return sqrt(distance);
}

// Helper function to find the indices of the k nearest neighbors for a given test image
std::vector<int> KNNClasifier::findNearestNeighbors(int testImageIndex)
{
    std::vector<int> neighbors;
    std::vector<double> distances;

    for (int i = 0; i < trainImages.getNumImages(); i++)
    {
        double distance = calculateDistance(testImageIndex, i);
        neighbors.push_back(i);
        distances.push_back(distance);
    }

    std::partial_sort(neighbors.begin(), neighbors.begin() + k, neighbors.end(),
                      [&distances](int a, int b) { return distances[a] < distances[b]; });

    neighbors.resize(k);
    return neighbors;
}

void KNNClasifier::fit(T trainImages, std::vector<int> trainLabels)
{
    this->trainImages = trainImages;
    this->trainLabels = trainLabels;
}

std::vector<int> KNNClasifier::predict(T& testImages)
{
    std::vector<int> predictions;

    for (int i = 0; i < testImages.getNumImages(); i++)
    {
        std::vector<int> neighbors = findNearestNeighbors(i);
        std::vector<int> classCounts(trainLabels.size(), 0);

        for (int neighbor : neighbors)
        {
            int label = trainLabels[neighbor];
            classCounts[label]++;
        }

        int maxCount = 0;
        int predictedLabel = 0;
        for (int j = 0; j < classCounts.size(); j++)
        {
            if (classCounts[j] > maxCount)
            {
                maxCount = classCounts[j];
                predictedLabel = j;
            }
        }

        predictions.push_back(predictedLabel);
    }

    return predictions;
}

bool KNNClasifier::save(std::string filepath)
{
    std::ofstream file(filepath);
    if (!file)
    {
        std::cerr << "Failed to save KNNClassifier to file." << std::endl;
        return false;
    }


    file << "KNNClassifier" << std::endl;
    file << "k: " << k << std::endl;
    file << "trainImages: " << std::endl;

    for (int i = 0; i < trainImages.getNumImages(); i++)
    {
        for (int j = 0; j < trainImages.getNumPixels(); j++)
        {
            file << trainImages.getPixel(i, j) << " ";
        }
        file << std::endl;
    }

    file << "trainLabels: ";
    for (int i = 0; i < trainLabels.size(); i++)
    {
        file << trainLabels[i] << " ";
    }

    file.close();
    return true;
}

bool KNNClasifier::load(std::string filepath)
{
    std::ifstream file(filepath);
    if (!file)
    {
        std::cerr << "Failed to load KNNClassifier from file." << std::endl;
        return false;
    }

    std::string line;
    getline(file, line); // skip the rest of the line

    trainImages.getImages().clear();
    trainLabels.clear();


    // Load training images
    int numImg = 0;
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

        trainImages.setImage(image);
        numImg++;
    }

    file.close();
    return true;
}

double KNNClasifier::eval(T& testImages)
{
    std::vector<int> predictions = predict(testImages);
    int correct = 0;
    int total = testImages.getNumImages();

    for (int i = 0; i < total; i++)
    {
        if (predictions[i] == testImages.getTrainLabels()[i])
        {
            correct++;
        }
    }

    return static_cast<double>(correct) / total;
}