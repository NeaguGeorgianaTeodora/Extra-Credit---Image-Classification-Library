#include "BayesClassifier.h"
#include "T.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include<algorithm>


BayesClassifier::BayesClassifier()
{

}

void BayesClassifier::fit(T trainImages, std::vector<int> trainLabels)
{
    // Determine the number of classes by finding the maximum label in trainLabels and adding 1
    int numClasses = *std::max_element(trainLabels.begin(), trainLabels.end()) + 1;

    // Get the number of pixels in each image
    int numPixels = trainImages.getNumPixels();

    // Resize the likelihoods vector to store the likelihood values for each class and pixel value
    likelihoods.resize(numClasses, std::vector<double>(numPixels * 256, 0.0));

    // Resize the priorProbabilities vector to store the prior probabilities for each class
    priorProbabilities.resize(numClasses, 0.0);

    // Create a vector to store the count of images in each class, initialized to 0
    std::vector<int> classCounts(numClasses, 0);

    // Iterate over each image in the training set
    for (int i = 0; i < trainImages.getNumImages(); i++)
    {
        int label = trainLabels[i];
        classCounts[label]++;

        // Iterate over each pixel in the current image
        for (int j = 0; j < numPixels; j++)
        {
            int pixelValue = trainImages.getPixel(i, j);

            // Increment the count of occurrences of the pixel value for the corresponding class and pixel index
            likelihoods[label][j * 256 + pixelValue]++;
        }
    }

    //Iterate over each class
    for (int i = 0; i < numClasses; i++)
    {
        // Calculate the prior probability of the class
        priorProbabilities[i] = static_cast<double>(classCounts[i]) / trainImages.getNumImages();

        // Iterate over each element in the likelihoods vector for the current class
        for (int j = 0; j < numPixels * 256; j++)
        {
            // Divide the count of occurrences of the pixel value by the sum of the class
            //count and the number of pixels for additive smoothing

            likelihoods[i][j] /= classCounts[i] + numPixels; // Additive smoothing
        }
    }
}


std::vector<int> BayesClassifier::predict(T& testImages)
{
    std::vector<int> predictions;

    // Iterate over each image in the test set
    for (int i = 0; i < testImages.getNumImages(); i++)
    {
        // Initialize the maximum log probability with negative infinity
        double maxLogProb = -std::numeric_limits<double>::infinity();
        //default label
        int predictedLabel = 0;

        // Iterate over each class
        for (int j = 0; j < priorProbabilities.size(); j++)
        {
            // Calculate the logarithm of the prior probability of the current class
            double logProb = log(priorProbabilities[j]);

            // Iterate over each pixel in the current test image
            for (int k = 0; k < testImages.getNumPixels(); k++)
            {
                int pixelValue = testImages.getPixel(i, k);
                // Calculate the logarithm of the likelihood of the current
                // pixel value given the current class and pixel index and add it to the log probability
                logProb += log(calculateLikelihood(j, k, pixelValue));
            }

            // Check if the current log probability is greater than the maximum log probability so far
            if (logProb > maxLogProb)
            {
                maxLogProb = logProb;
                predictedLabel = j;
            }
        }

        predictions.push_back(predictedLabel);
    }

    return predictions;
}


bool BayesClassifier::save(std::string filepath)
{

    std::ofstream file(filepath);
    if (!file)
    {
        std::cerr << "Failed to save BayesClassifier to file." << std::endl;
        return false;
    }

    file << "BayesClassifier" << std::endl;
    file << "numClasses: " << likelihoods.size() << std::endl;
    file << "numPixels: " << likelihoods[0].size() / 256 << std::endl;
    file << "likelihoods: " << std::endl;

    for (int i = 0; i < likelihoods.size(); i++)
    {
        for (int j = 0; j < likelihoods[i].size(); j++)
        {
            file << likelihoods[i][j] << " ";
        }
        file << std::endl;
    }

    file << "priorProbabilities: ";
    for (int i = 0; i < priorProbabilities.size(); i++)
    {
        file << priorProbabilities[i] << " ";
    }

    file.close();
    return true;
}


bool BayesClassifier::load(std::string filepath)
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


double BayesClassifier::eval(T& testImages)
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
