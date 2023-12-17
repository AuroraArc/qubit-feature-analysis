# Understanding the Influence of Qubit Features on Error Rates
This is the compilation of the first of two semesters of a solo research project conduct by me under the guidance of Dr.Nicholas LaRacuente (nlaracu@iu.edu) during the fall 2023 semester, which will continue into the spring 2024 semester.


## Abstract
The goal of this project is to assess correlations between qubit features and ultimately predict error probability. In the context of quantum computing, quantum bits (qubits) serve as the fundamental units for information processing, comparable to classical bits in conventional computers that store and process data using 1s and 0s. Quantum gates, analogous to classical logic gates such as NOT and OR, are employed to manipulate qubit states. The project involved the application of time series analysis to examine 7-qubit systems, notably 'ibm_perth', over a span of 315 days. Additionally, normal regression analysis was conducted for 127-qubit systems over a one-day period, instead treating each individual qubit as a data point, encompassing seven unique systems. Evaluation of prediction accuracy was carried out using root mean square error (RMSE) and R-squared (R2), common regression techniques. The results indicate a limited predictive capability of qubit features in relation to error rates when employing linear and low-order polynomials. Notably, a stronger correlation was noted across qubits compared to across time. Future exploration involves investigating the discrepancy between the correlation matrix and the error coefficients, and a comprehensive understanding of the relationship between qubit features and error rates may enhance error mitigation when integrated with existing models.

## Introduction
First, what is a quantum computer [1]? To put it simply, it is a computer that uses quantum bits (qubits) to process information. Two notable features of a quantum computer allow it to be much more powerful than a classical computer:
 * **Superposition** allows the qubit to in a state of one and zero *simultaneously*
 * **Entanglement** "binds" a group of qubits together; their individual states are dependent on each other, no matter how far they are from each other

There are a few challenges that hinders quantum computing performance [2, 3]:
 * It is extremely susceptible to noise, which can be anything from ambient temperature to local radiation from Wi-Fi
 * Scalability issues&mdash;the largest processor was just unveiled by IBM (as of December 2023): the 1,121-qubit Condor processor
 * Relatively new field&mdash;in 1998, the first 2-qubit computer was built

In quantum computing, errors can arise from various sources, including decoherence, environmental factors, and human error on qubit maintenance. These errors pose a significant challenge to the reliability and accuracy of quantum computers. There are many examples of methods that try to solve, or mitigate, error, such as noise intermediate-scale quantum (NISQ) devices, quantum algorithm optimization, and error correction codes.

Predicting gate error probability in the context of quantum computing estimating the likelihood of errors occurring during quantum gate operations. Some reasons why this is important in quantum computing include:
 * Error correction strategies&mdash;understanding and predicting are essential for quantum error correction strategies as they rely on accurate estimates of error probabilities to detect and correct errors effectively
 * Hardware benchmarks&mdash;predicting quantum error probability aids in developing effective calibration protocols; this is mostly what this project is focused on


## Methods
### 1. Data Collection
#### 1.1 Qsikit SDK and IBM Machines
For data collection, I utilized the Qiskit SDK [4], which provided access to IBM quantum computers.

#### 1.2 Quantum Processors and Data Length
One 7-qubit processor and seven 127-qubit processors were utilized:
 * 'ibm_perth': a 7-qubit system based on the Falcon processor, with 315 days of data
 * 'ibm_brisbane', 'ibm_cusco', 'ibm_kawasaki', 'ibm_kyoto', 'ibm_nazca', 'ibm_quebec', 'ibm_sherbrooke': 127-qubit systems based on the Eagle processor, with one day of data for each system compiled into one dataset

### 2. Preprocessing
#### 2.1 Qubit Features
The following qubit features were considered [5]:
 * Decoherence time (T1 and T2)
 * Frequency, anharmonicity
 * Readout assignment error ('prob meas0 prop1' and 'prob meas1 prop0'), readout length
 * gate time
 * ID, rz, sx, Pauli-X (NOT gate), reset, √x gate errors
 * controlled NOT (CNOT) gate error (exclusive to 7-qubit)
 * ECR gate error (exclusive to 127-qubit)

All features were scaled to have a standard deviation of 1.

### 3. Time Series Analysis of CNOT Gate Error on 'ibm_perth'
#### 3.1 LASSO Regression
LASSO Regression was employed for its simplicity and feature selection capabilities [6].
* $\min_{w} { \frac{1}{2n_{\text{samples}}} \|X w - y\|_2 ^ 2 + \alpha \|w\|_1}$
  * $n_{samples}$: number of samples
  * $X$: input features (independent variables)
  * $y$: target value (in this case, CNOT error)
  * $w$: coefficient vector
  * $a\|w\|_1$: LASSO regularization
#### 3.2 Polynomial Regression
* Polynomial regression (quadratic order) was applied to capture nonlinear relationships [6].

Five fold cross-validation was conducted using a traditional 80-20 split:
* The dataset was divided into five subsets of approximately equal size.
* The model was trained on 80% (or 4/5 of the folds) and then tested on the remaining 20% (or the remaining fold).
* This process was repeated five times with different training and testing combinations.

### 4. ECR Gate Error Prediction for 127-qubit Systems
Linear and polynomial LASSO regression were employed again with five-fold cross-validation. Instead of time, predictions for one random 'ECR gate error' value were made across different qubits of all system.

## Results
### 1. Time Series Analysis
#### 1.1 Optimal $\alpha$-Regularization
The optimal $\alpha$-regularization parameter value was determined to be $0.07$.
#### 1.2 Model Performance Metrics
* Average R$^2$ score $\approx -1.252$
  * The model performs worse than just using the average of the predicted value as a prediction.
  * Optimal R$^2$ score = $1$
* Average RMSE score $\approx 4.319 \times 10^{-3}$
  * The inconclusive nature of the RMSE results is attributed to the small range of values.
  * Optimal RMSE score = $0$
#### 1.3 Implications
The results show that qubit data over time is not predictive of error rate.

### 2. ECR Prediction Across Qubits
#### 2.1 Optimal $\alpha$-Regularization
The optimal $\alpha$-regularization parameter value was found to be $0.11$.
#### 2.2 Model Performance Metrics
* Average R$^2$ score $\approx 0.105$
  * The model exhibits weak predictive power but performs better than the time series analysis.
* Average RMSE score $\approx 0.900$
  * The very poor RMSE score is noted due to the small range of values.
#### 2.3 Implications
THe results suggest that qubit data over other qubits is slightly predictive of the error rate but still exhibits a poor score.

### 3. Correlation Matrix
<p align="center">
  <img src="https://github.com/AuroraArc/qubit-feature-analysis/blob/main/images/corrmatrix.png" />
</p>

Some things to note from the visualization of the matrix as a heatmap are the decoherence times(T$_1$, T$_2$) show a negative relationship across the other variables. The anharmonicity values seems to be neutral across all variables. Most notably, the probability errors ('prob meas0 prop1, prob meas1 prop0') and gate errors (Pauli-X, ECR) all show a slight relationship with each other. This is interesting because the correlation matrix seem to tell that there should be some correlation across some of the features, so the predictive power should be higher.

### 4. ECR Plot
<p align="center>
  <img src="https://github.com/AuroraArc/qubit-feature-analysis/blob/main/images/scatterplot.png" />
</p>



## Discussion
## References
