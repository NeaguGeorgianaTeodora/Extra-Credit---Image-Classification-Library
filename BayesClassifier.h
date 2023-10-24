#ifndef EXTRA_CREDIT_IMAGE_CLASSIFICATION_LIBRARY_BAYESCLASSIFIER_H
#define EXTRA_CREDIT_IMAGE_CLASSIFICATION_LIBRARY_BAYESCLASSIFIER_H

#pragma once
#include "Clasifier.h"

class BayesClassifier : public Clasifier
{
public:
    T trainImages;
    std::vector<int> trainLabels;
    std::vector<std::vector<double>> likelihoods;
    std::vector<double> priorProbabilities;

    std::vector<double> prior;
    std::vector<std::vector<double>> likelihood;


    //Helper function to calculate the likelihood for a given class and pixel value
    double calculateLikelihood(int classIndex, int pixelIndex, int pixelValue)
    {
        return likelihoods[classIndex][pixelIndex * 256 + pixelValue];
    }

public:

    BayesClassifier();

    //will actually train the classifier.
    void fit(T trainImages, std::vector<int> trainLabels) override;

    //will return the predicted labels for all the images in the matrix T
    std::vector<int> predict(T& testImages) override;

    bool save(std::string filepath) override;

    bool load(std::string filepath) override;

    double eval(T& testImages) override;
};



#endif //EXTRA_CREDIT_IMAGE_CLASSIFICATION_LIBRARY_BAYESCLASSIFIER_H
