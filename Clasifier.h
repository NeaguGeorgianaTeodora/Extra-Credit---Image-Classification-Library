#ifndef EXTRA_CREDIT_IMAGE_CLASSIFICATION_LIBRARY_CLASSIFIER_H
#define EXTRA_CREDIT_IMAGE_CLASSIFICATION_LIBRARY_CLASSIFIER_H

#pragma once

#include<vector>
#include<string>
#include "T.h"

class Clasifier
{

public:


    virtual void fit(T trainImages, std::vector<int> trainLabels) = 0;

    virtual std::vector<int> predict(T& imageData) = 0;

    //stores all the information related to the classifier in a file
    virtual bool save(std::string filepath) = 0;

    //reads all the information related to the classifier from the file passed as parameter
    virtual bool load(std::string filepath) = 0;

    //returns the accuracy of the classifier
    //(the number of correct predictions for the images in T divided by the total number of samples in T).
    virtual double eval(T& testImage) = 0;

};


#endif //EXTRA_CREDIT_IMAGE_CLASSIFICATION_LIBRARY_CLASSIFIER_H
