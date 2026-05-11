# Anomaly Detection Using Isolation Forests and Gaussian Mixture Models

## Summary

This project explores the concept of anomaly detection in unlabeled data using Isolation Forests and Gaussian Mixture Models from the Scikit-Learn package. Most machine learning tasks are focused on creating models which create predictions that most closely mimic the underlying dataset with the assumption that almost all of the data is useful or important. However, in certain domains such as fraud detection, irregular and infrequent data points are more important. Most machine learning models are unable to properly pick out anomalies due to a lack of anomalous samples available to extract consistent patterns. Anomaly detection focuses on models which are able to extract these irregular data points, especially those which may not be immediately obvious as anomalous.

For this project, an unlabeled set of financial data was provided. As an unsupervised machine learning task, there is not a set of ground-truth labels to use for directly confirming the effectiveness of each model. Instead, domain knowledge and visual data projection techniques were used to identify a set of candidate anomalies. These candidates were used as pseudo-labels, along with clustering metrics and data distributions, in order to evaluate the potential usefulness of each model type utilizing various hyperparameter configurations.

## Requirements

The libraries and version of Python used to create this project are listed below. The requirements are also available at [requirements.txt](https://github.com/JoshuaGottlieb/Anomaly-Detection-SKLearn/blob/main/requirements.txt).

```
matplotlib==3.10.6
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.6.1
seaborn==0.13.2
```

## Repository Structure

```
├── data                                           # Raw and processed train/test datasets
├── docs                                           # A brief report with an overview and analysis of the work in the project notebook
├── models                                         # Pickled and compressed trained models
├── results                                        # Clustering metrics and predictions
├── src                                            # Project notebooks and source code
│   ├── Anomaly_Detection.ipynb                        # Notebook containing EDA, training, and analysis
│   ├── modules                                        # Source code with custom functions
│   │   ├── io_utils.py                                    # Functions for serialization and deserialization
│   │   ├── plotting.py                                    # Functions for Matplotlib and Seaborn plotting
│   │   ├── plotting_utils.py                              # Helper functions for plotting
│   │   ├── preprocessing.py                               # Functions for preprocessing raw data
│   │   ├── statistics.py                                  # Functions for EDA statistical techniques
│   │   └── training.py                                    # Functions for training models and gathering predictions
├── visualizations                                 # Saved figures from notebook
├── README.md
└── requirements.txt
```
