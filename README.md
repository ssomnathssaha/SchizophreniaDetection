# SchizophreniaDetection
# Machine Learning
# ELM

Schizophrenia - Schizophrenia is a psychiatric disorder in which a person shows some positive, some negative and cognitive symptoms. Here positive doesn’t means that a person will show some positive sign instead it means a person will show signs some symptoms which are normally not present in human body. Positive symptoms are psychotic in nature like delusions, hallucination, disorganized speech, disorganized behavior, catatonic behavior. Like for example a TV reporter said “There are chances of earthquake”, so a person might think he is directly referring to him. Hallucinations can be a person thinking there might be another person standing beside him though he might not be physically present at the moment. More than 1 million cases of schizophrenia disease per year (India). 

The Problem Statement - Till 2015, 17000 people have died due to schizophrenia. Currently there is no physical or lab test available to detect schizophrenia. Mostly it is diagnosed over a period of 6 months (with 1 month of active symptoms) by a psychiatrist based on clinical symptoms. If we can automate this Schizophrenia detection, then this will help to fight against the problem. One of the fast approach is to use Machine Learning technique in order to make a system learn and then work by itself. Our aim of this project is to classify schizophrenic patients faster than the current diagnoses available and maximize the accuracy of the detection. We have a dataset called "Function Biomedical Informatics Research Network Data Repository" i.e. FBIRN which is obtained from the Function BIRN Data Repository. We want to use good feature selection algorithm to reduce number of features and classify a subject after training a model using classifier.

Objective of Study - Currently there is no physical or lab test available to detect schizophrenia. Mostly it is diagnosed over a period of 6 months (with 1 month of active symptoms) by a psychiatrist based on clinical tests. If we can automate this Schizophrenia detection, then this will help to fight against the problem. One of the fast approach is to use Machine Learning technique in order to make a system learn and then work by itself. Our aim of this project is to classify schizophrenic patients faster than the current diagnoses available and maximize the accuracy of the detection.

Dataset - I have got a dataset of 60 subjects. The data are in fMRI image format. Out of this 60, 30 subjects have Schizophrenia disease and rest 30 do not have Schizophrenia. After pre-processing we have final data of 60 subjects where each subject have 153594 features. So we have a matrix of 60 * 153594.

Our final aim is to train our machine in an optimised way so that the trained machine can guess whether a new data is of Schizophrenia patient or not.

Our target was to use ELM(Extreme Learning Machine) for the classification model.

The main challenge of this project is to reduce the number of features by deleting redundant features and by removing irrelevant features too. For this, I have tried many feature selection techniques. After applying PCC, PCA, LDA, t-test, Relief finally, I have got the combination of t-test and Relief to get the best accuracy. 

Also, I have tried different ELM techniques like ELM, ELM kernel, OS ELM, V ELM, B ELM, Complex ELM, H ELM.

I have used ensemble of good classifiers to get the best performance.
Right now I am getting 96% accuracy. 

In future, I will try to increase the accuracy. Your contribution is always welcome. :)
