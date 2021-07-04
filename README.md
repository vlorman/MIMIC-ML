# Predicting mortality in electronic health record data

# Introduction

The goal of this project is to apply a variety of machine learning models to the problem of predicting mortality from Electronic Health Record (EHR) data (both structured and unstructured). More specifically, our dataset consists of hourly observations of 104 clinically-aggregated covariates related to vital signs and lab results, static features (age, gender, and ethnicity), as well as free-form patient chart notes from the MIMIC III electronic health record database <cite data-cite="11046887/SG9TBKZH"></cite>. We use this data to predict mortality for Intensive Care Unit (ICU) patients based on the first 12, 24, and 48 hours of their stay, respectively. The models we investigate include variations on logistic regression (with Principal Component Analysis (PCA) preprocessing and regularization), random forests, support vector classifiers, and a handful of deep learning models, including a multi-input model that takes as input both the numeric data consisting of hourly observations as well as free-form text data from chart notes. In each case, we tune hyperparameters through cross-validation and report the results on test datasets.

The MIMIC III database consists of deidentified health-related data associated to over 40,000 patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012. The data was accessed through Physionet, and though publicly available, requires credentialing before access. As such, we do not display the data itself in this document, only the results of our models and the code used to run them, which may be replicated by anyone who has the data. For the first step of processing the data, we use the MIMIC-Extract open-source pipeline <cite data-cite="11046887/I5YVQAME"></cite>, which handles unit conversion, removal of outliers, and accounts for duplication in the original dataset. The remaining processing needed for our models is carried out by the [processing.py](processing.py) script.

We begin by summarizing the structure of our data and the metrics by which we will evaluate the performance of our models. We then present a summary and discussion of our results. The code for running our models and tuning hyperparameters is presented in the final sections.

# The MIMIC III dataset

## Structure of the data

Our starting point is the output of the MIMIC-Extract pipeline, which handles unit conversion, removal of outliers, and accounts for duplication in the original MIMIC III dataset. For each patient's ICU stay, the output of the MIMIC-Extract pipeline contains mean hourly measurements of 104 features, along with age, ethnicity, and gender. For many patients and hours, these features are missing, and we begin by further processing the MIMIC-Extract data by applying the following steps (carried out in the file [processing.py](processing.py)).

- We impute the missing measurements following the "simple imputation" scheme applied in <cite data-cite="11046887/J3W4HI5G"></cite> and also used in <cite data-cite="11046887/I5YVQAME"></cite>, whose code we adapt for this part. For each feature, we first forward-fill missing measurements, then for the remaining missing measurements (those with no prior measurements) we replace them with the individual mean (or, in the case of no measurements of that feature, we fill it by the population mean).

- We encode the categorical variables--age (divided into 10 buckets by quantile), ethnicity, and gender--by dummy variables.

- We append to each patient an additional string variable consisting of the unstructured chart notes for that patient over the relevant time window. Unlike the hourly observation data, we do not keep track of the time series nature of the chart note data at present.

- We construct three datasets, one for each window size of 12, 24, and 48 hours. We include a 6 hour buffer between the period over which we use predictive covariates and the period in which we are predicting mortality, so that patients included in the 12, 24, and 48 hour datasets were in the ICU for at least 18, 30, and 54 hours, respectively.

- We split the data into a training set and test set (80% and 20%, respectively) with stratification to ensure that the mortality rate in the training and test sets is similar. We split the data in such a way that the 12, 24, and 48 hour training and test sets are nested, so that, for instance, a patient in the 24 hour training set appears in the 12 hour training set and not the 12 hour test set.

For now, we focus only on the hourly observations and categorical features, leaving the unstructured chart note data for our neural network models in the second half of this document.

We tune hyperparameters on our training data and only use the test data for the final step of model evaluation. We also note that in splitting the train and test data in the [procesing.py](processing.py) script, we make a point to use a stratified splitting, which ensures the same percentage of positives (patients who die) in the train set as in the test set. Because the two classes in the data are so imbalanced (see below), this turns out to make a big difference, and early attempts to evaluate models that did not use stratification in splitting the data ran into stability issues.

# Discussion of metrics

Our goal in this document is mortality prediction, and we begin by noting that our binary classification problem is imbalanced. Indeed, as a simple baseline for our models, we calculate the percent of patients who did not die during their ICU stay.

A trivial model predicting survival for each patient would have around 93.7% accuracy on the 12 hour dataset, 92.9% accuracy for the 24 hour dataset, and 90.9% accuracy on the 48 hour dataset. This is first baseline against which we compare our subsequent models.

To get a more sophisticated view of the predictive power of our model, we keep track of the following metrics.

- **Accuracy**: percent of ICU stays in which we predicted mortality correctly.

- **AUROC**: area under the ROC curve, which plots false positive rate against true positive rate.

- **AUPR**: area under the precision recall curve, which plots recall (percent of true positives which are predicted positives) against precision (percent of predicted positives which are true positives).

- **F1 score**: the harmonic mean of Precision and Recall.

Because of the low mortality rate in our data, our classes are quite imabalanced. Furthermore, we are in a setting where it is very likely that a false negative (failing to predict mortality for a patient who will die) carries a higher risk than a false positive (predicting mortality for a patient who will not die). As such, evaluating our models based on the AUROC score alone is somewhat misleading as the AUROC score may be overly optimistic in this setting (see <cite data-cite="11046887/C5V6RWQ5"></cite> for an example). We attach more significance to the area under the Precision-Recall curve, AUPR, which frames our problem in terms of the tradeoff between precision and recall: the challenge is to produce sufficiently high recall (i.e. to correctly detect patients who will die) while maintaining sufficiently high precision (identifying patients who will die correctly).

# Summary of results and discussion

## Summary of models evaluated

In this project, we evaluate the following models on each of our 12, 24, and 48 hour datasets.

- **Logistic regression** (with and without categorical features)  


- **Logistic regression with Principal Component Analysis (PCA) preprocessing**. We use cross-validation to select the optimal number of principal components.


- **Logistic regression with regularization**. We evaluate both L1 (lasso) and L2 (ridge) regularization at several values of the regularization parameter to select optimal regularization.  


- **Random forests**. We cross-validate to optimize the number of estimators.  


- **Support Vector Classifiers**. We cross-validate to select the optimal choice of kernel (choosing between linear, polynomial, radial basis functions, and sigmoid), then for the optimal kernel, we cross-validate to select the optimal values of C and gamma.  


- **Multilayer perceptron on structured ICU stay data**. We tested a variety of architectures to settle on one with 3 dense hidden layers (with regularization and dropout). We also experimented with LSTM and GRU layers to account for the time-series nature of our observations, but we exclude these from our analysis since we were unable to achieve optimal results with them (see the Future Work section for more on this).  


- **Neural network on chart note data** To process the input data, we experimented with both word emebeddings (with GRU and 1D Convolutional layer built on top) as well as a simple TF-IDF encoding of the data. We achieved the strongest performance with the TF-IDF encoding of the chart note data followed by a multilayer perceptron with a single dense hidden layer.


- **Multi-input neural network** We combined both of the above neural nets to construct one with two inputs, each fed through dense layers before being concatenated and fed through one more dense hidden layer.

## Results

The table below summarizes the results for each of our models with the optimal choice of hyperparameters (selected through cross-validation), evaluated on our test dataset.

| Model  | Window | Hyperparameters  | Accuracy  | AUROC | AUPR | F1 |
|---|---|---|---|---|---|---|
|Logistic regression (LR)| 12 hr |-|0.942|0.879|0.439|0.384|
|| 24 hr |- |0.930|0.846|0.408|0.397|
|| 48 hr |-|0.890|0.794|0.375|0.347|
|LR w/ PCA| 12 hr |n_components=300|0.945|0.893|0.477|0.378|
|| 24 hr | n_components=300|0.939|0.879|0.480|0.405|
|| 48 hr |n_components=300|0.921|0.873|0.513|0.407|
|LR regression w/ Regularization | 12 hr | L1, C=1.0 |0.944|0.893|0.470|0.355|
|| 24 hr |L2, C=1.0  |0.939|0.879|0.473|0.368|
|| 48 hr | L2, C=0.1 |0.923|0.870|0.510|0.412|
|Random Forests | 12 hr | n_estimators=1800 |0.943|0.892|0.493|0.454|
|| 24 hr | n_estimators=1400 |0.939|0.884|0.501|0.253|
|| 48 hr |n_estimators=400 |0.929|0.876|0.505|0.298|
|Support Vector Classifier | 12 hr | kernel=rbf, gamma=0.000, C=0.056|0.193|0.882|0.388|0.314|
|| 24 hr | kernel=rbf, gamma=0.000, C=0.056 |0.742|0.868|0.364|0.312|
|| 48 hr |kernel=rbf, gamma=0.000, C=0.056 |0.689|0.828|0.336|0.308
|NN w/ numerical input | 12 hr |(see below)|0.946|0.913|0.507|0.422|
|| 24 hr |- |0.948|0.902|0.528|0.456|
|| 48 hr |-|0.948|0.864|0.528|0.469|
|NN w/ chart note data input| 12 hr |- |0.945|0.883|0.452|0.319|
|| 24 hr |-  |0.933|0.858|0.377|0.151|
|| 48 hr |- |0.915|0.811|0.302|0.047|
|Multi-input NN| 12 hr | |0.948|**0.927**|**0.576**|0.378|
|| 24 hr |-  |0.948|**0.917**|**0.555**|0.416|
|| 48 hr |- |0.948|**0.889**|**0.580**|0.485|

## Discussion

For each window size, we find the strongest performance (in terms of both AUROC as well as AUPR) is achieved by the multi-input neural network model. This is the model which takes advantage of all of our data (the hourly observations, the categorical variables, and the chart notes). Compared to the neural net model which uses only the hourly observations, we see that incorporating the unstructured chart notes (via TF-IDF encoding) results in a modest increase in both AUROC and AUPR.

We also note that due to the class imbalance in our problem, the AUROC score is consistently and substantially larger than the AUPR score. Indeed, achieving high recall at a high precision is quite challenging. For instance, the multi-input neural network model on the 12 hour data with the highest AUROC score (0.927) has an AUPR score of 0.576. At a threshhold of 0.5, this corresponds to a recall of only 0.251 with a precision of 0.766. The model on the 48 hour data which achieves the highest AUPR score overall of 0.580 has, at a 0.5 threshhold, a recall of 0.375 with a precision of 0.687. We see that AUPR gives a more accurate assesment of our models as AUROC is somewhat overoptimistic.

It is also interesting to note that on our best model as well as several others, we find that the highest AUROC score is achieved with a 12 hour window size, while the highest AUPR score is achieved with a 48 hour window size. We expect that there are two competing factors that come with increasing the window size: on the one hand, mortality prediction may be more difficult for patients who have been in the ICU longer, and on the other hand, using a larger window size provides us with more data to make predictions.

Besides for the deep learning models, the following three models all achieved comparable performance: random forests, logistic regression with PCA preprocessing, and logistic regression with regularization. If interpretability is of importance to our model, the logistic regression models provide this with only a modest (10-20%, depending on window size) drop in AUPR. 

Our multi-input neural network model trained on both hourly observation, categorical, and chart note data performed favorably compared to the strongest model benchmarked in <cite data-cite="11046887/I5YVQAME"></cite> in the in-ICU mortality prediction task at a window size of 24 hours, which had an AUROC and AUPR scores of 89.1 and 50.9, respectively (compared to 91.7 and 55.5 in our case). We note that our best model had the advantage of being trained on the chart note data in addition to the hourly observations and categorical variables. This adds further evidence to the observation based on our models that incorporating unstructured chart note data from electronic health records improves predictive power. However, an important caviat is that we used stratified splitting in constructing our training and test sets to ensure that the mortality rates were similar in the two, whereas <cite data-cite="11046887/I5YVQAME"></cite> did not. We observed that the splitting into train and test sets without stratification resulted in somewhat unstable mortality rates between the two, which in turn effected the scoring of our models. It is thus likely that the comparison between our results and those benchmarked in <cite data-cite="11046887/I5YVQAME"></cite> is not entirely valid.

Finally, we note that in processing the unstructured chart note data, we took the simplest approach in concatenated all chart notes recorded during the window of interest and encoding these using TF-IDF (Text Frequency-Inverse Document Frequency). This approach only keeps track of which words appear in which chart notes and with what frequency and does not depend on the syntactic structure of the chart notes or on what words appear near other words and in what order. We did initially experiment with embedding layers in our chart note models, which are potentially more sensitive to this kind of structure; however, we found that they did not yield any visible improvement compared to the simple TF-IDF models, suggesting that, at first approximation, perhaps the TF-IDF encoding captures the most important features of the chart notes and that the other structural properties of the chart notes are less important. Whether one could use more sophisticated models based on word embeddings to achieve better results is an interesting question worth further investigation.

# Future work

Since deep learning models achieved the best results on our data, it is natural to expect that more sophisticated neural network architectures might achieve better results. Likely the biggest limitation of our neural network models is that they don't take full advantage of the time series nature of the data, or of the fact that many of observations were missing and had to be imputed. In experimenting with various models, we did do some preliminary evaluation of models with LSTM and GRU layers that could better exploit the time series nature of our hourly observations. However, we omit these from our analysis from the time being, as we have been unable to achieve better results than the models discussed above. However, we do note that <cite data-cite="11046887/J3W4HI5G"></cite> found that such models do lead to improvement in model performance.

Thus, a more optimal approach would take into account the fact that the original dataset from MIMIC-Extract has a substantial amount of missing data, that the missingness of data is in fact biased, and as such, taking into account which observations are missing and the time since the last observation could achieve stronger results than the imputation we used above. Indeed, this is the idea behind the GRU-D (GRU with Decay) layer developed and implemented for the MIMIC data in <cite data-cite="11046887/J3W4HI5G"></cite>. The GRU-D layer functions similarly to a standard GRU layer but includes decay terms related to the time since last observation and a variable indicating whether the observation was missing and imputed. 

A particularly promising area for future work is combining our hybrid structured-unstructured input model with one with GRU-D type layers. In the case of textual chart data, it makes sense to record the time since the last observation and to incorporate a decay term, just as the GRU-D layer do with the numerical hourly observations. Such a model might require development of a 2-dimensional GRU-D layer (one direction for the text and another for time at which notes were recorded).

## Bibliography

<div class="cite2c-biblio"></div>



























