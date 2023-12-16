# Understanding the Influence of Qubit Features on Error Rates
This is the compilation of the first of two semesters of a solo reseach project conduct by me under the guidance of Dr.Nicholas LaRacuente (nlaracu@iu.edu) during the fall 2023 semester, which will continue into the spring 2024 semester.


## Abstract
The goal of this project is to assess correlations between qubit features and ultimately predict error probability. In the context of quantum computing, quantum bits (qubits) serve as the fundamental units for information processing, comparable to classical bits in conventional computers that store and process data using 1s and 0s. Quantum gates, analogous to classical logic gates such as NOT and OR, are employed to manipulate qubit states. The project involved the application of time series analysis to examine 7-qubit systems, notably ibm_perth, over a span of 315 days. Additionally, normal regression analysis was conducted for 127-qubit systems over a one-day period, instead treating each individual qubit as a data point, encompassing seven unique systems. Evaluation of prediction accuracy was carried out using root mean square error (RMSE) and R-squared (R2), commmon regression techniques. The results indicate a limited predictive capability of qubit features in relation to error rates when employing linear and low-order polynomials. Notably, a stronger correlation was noted across qubits compared to across time. Future exploration involves investigating the discrepancy between the correlation matrix and the error coefficients, and a comprehensive understanding of the relationship between qubit features and error rates may enhance error mitigation when integrated with existing models.

## Introduction
First, what is a quantum computer [1]? To put it simply, it is a computer that uses quantum bits (qubits) to process information. Two notable features of a quantum computer allow it to be much more powerful than a classical computer:
 * **Superposition** allows the qubit to in a state of one and zero *simultaneously*
 * **Entanglement** "binds" a group of qubits together; their individual states are dependent on each other, no matter how far they are from each other

There are a few challenges that hinders quantum computing performance [2, 3]:
 * It is extremely susceptible to noise, which can be anything from ambient temperature to local radiation from Wi-Fi
 * Scalability issues&mdash;the largest processor was just unveiled by IBM (as of December 2023): the 1,121-qubit Condor processor
 * Relatively new field&mdash;in 1998, the first 2-qubit computer was built

In quantum computing, errors can arise from various sources, including decoherence, environmental factors, and human error on qubit maintenence. These errors pose a significant challenge to the reliability and accuracy of quantum computers. There are many examples of methods that try to solve, or mitigate, error, such as noise intermediate-scale quantum (NISQ) devices, quantum algorithm optimization, and error correction codes.

Predicting gate error probability in the context of quantum computing estimating the likelihood of errors occurring during quantum gate operations. Some reasons why this is important in quantum computing include:
 * Error correction strategies&mdash;understanding and predicting are essential for quantum error correction strategies as they rely on accurate estimates of error probabilities to detect and correct errors effectively
 * Hardware benchmarks&mdash;predicting quantum error probability aids in developing effective calibration protocols; this is mostly what this project is focused on


## Methods

## Results
## Discussion
## References
