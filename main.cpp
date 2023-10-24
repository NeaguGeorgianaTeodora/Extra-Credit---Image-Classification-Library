#include "KNNClasifier.h"
#include "BayesClassifier.h"
#include <iostream>


int main()
{
    KNNClasifier classifier(1);
    classifier.load("source.txt");
    classifier.fit(classifier.trainImages, classifier.trainLabels);

    T testImages;
    testImages.load("Test.txt");
    std::cout<<classifier.eval(testImages)<<std::endl;
    std::vector<int> predictions = classifier.predict(testImages);
    /*
    for(int i = 0; i < predictions.size(); i++)
        std::cout << predictions[i] << std::endl;*/



    BayesClassifier classifierB;
    classifierB.load("source.txt");
    classifierB.fit(classifierB.trainImages, classifierB.trainLabels);
    std::cout << classifierB.eval(testImages) << std::endl;

    std::vector<int> predictionsB = classifierB.predict(testImages);

    /*
    for (int i = 0; i < predictionsB.size(); i++)
        std::cout << predictionsB[i] << std::endl;*/

    classifier.save("rezults.txt");
    return 0;
}