#ifndef EXTRA_CREDIT_IMAGE_CLASSIFICATION_LIBRARY_KNNCLASIFIER_H
#define EXTRA_CREDIT_IMAGE_CLASSIFICATION_LIBRARY_KNNCLASIFIER_H

#pragma once
#include "Clasifier.h"

class KNNClasifier : public Clasifier
{
public:
    T trainImages;
    std::vector<int> trainLabels;
    int k; //number of nearest neighbors to consider
    std::vector<int> findNearestNeighbors(int testImageIndex);
    double calculateDistance(int imageIndex1, int imageIndex2);

public:

    KNNClasifier(int k);
    //will actually train the classifier.
    void fit(T trainImages, std::vector<int> trainLabels) override;

    //will return the predicted labels for all the images in the matrix T
    std::vector<int> predict(T& testImages) override;

    bool save(std::string filepath) override;

    bool load(std::string filepath) override;

    double eval(T& testImages) override;
};



#endif //EXTRA_CREDIT_IMAGE_CLASSIFICATION_LIBRARY_KNNCLASIFIER_H
