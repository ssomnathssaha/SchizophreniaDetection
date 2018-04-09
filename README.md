# SchizophreniaDetection
# Machine Learning
# ELM

Schizophrenia - A disorder that affects a person's ability to think, feel and behave clearly. More than 1 million cases of schizophrenia disease per year (India).

I have got a dataset of 60 subjects. The data are in fMRI image format. Out of this 60, 30 subjects have Schizophrenia disease and rest 30 do not have Schizophrenia. After pre-processing we have final data of 60 subjects where each subject have 153594 features. So we have a matrix of 60 * 153594.

Our final aim is to train our machine in an optimised way so that the trained machine can guess whether a new data is of Schizophrenia patient or not.

Our target was to use ELM(Extreme Learning Machine) for the classification model.

The main challenge of this project is to reduce the number of features by deleting redundant features and by removing irrelevant features too. For this, I have tried many feature selection techniques. After applying PCC, PCA, LDA, t-test, Relief finally, I have got the combination of t-test and Relief to get the best accuracy. 

Also, I have tried different ELM techniques like ELM, ELM kernel, OS ELM, V ELM, B ELM, Complex ELM, H ELM.

Right now I am getting 95% accuracy. 

In future, I will try to increase the accuracy. Your contribution is always welcome. :)
