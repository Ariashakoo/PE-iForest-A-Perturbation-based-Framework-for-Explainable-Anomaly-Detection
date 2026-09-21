# PE-IForest: Perturbation-based Explainer for Anomaly Detection

This repository contains the official Python implementation associated with the research paper **"PE-Iforest"** (authored by Aria Shakoo, published on IEEEXplore). 

## Publication Details

* **Publisher:** IEEE
* **DOI:** [10.1109/IICAI70155.2026.11620789](https://doi.org/10.1109/IICAI70155.2026.11620789)
* **IEEE Xplore Link:** [https://ieeexplore.ieee.org/abstract/document/11620789](https://ieeexplore.ieee.org/abstract/document/11620789)
* **Date of Conference:** 31 May 2026 - 01 June 2026
* **Date Added to IEEE Xplore:** 28 July 2026

## Project Overview

Explaining the decisions of unsupervised anomaly detection models (like Isolation Forests) is notoriously difficult due to the lack of labeled ground truth and the structural opacity of the models. The Perturbation-Based Explainer (PE) framework identifies which features contribute most to an anomaly's score by systematically replacing individual feature values with a baseline reference (e.g., the global mean) and measuring the resulting change (improvement) in the anomaly score.

### The Optimization: Vectorized Perturbation Batching
A naive implementation of this perturbation process iterates through every feature of an anomaly sequentially, calling the model's `decision_function` repeatedly. This results in massive computational bottlenecks, especially in high-dimensional datasets.

This codebase introduces a **Vectorized Perturbation Framework**. Instead of looping through features, the explainer:
1. Replicates the anomalous instance $N$ times (where $N$ is the number of features) to create a matrix using `numpy.tile`.
2. Replaces the diagonal of this matrix with the baseline reference values using `numpy.fill_diagonal`.
3. Passes the entire matrix through the model's `decision_function` in a single, highly optimized batch operation.

This methodology not only exponentially accelerates the explanation generation for Isolation Forests but is also proven to be model-agnostic, seamlessly accelerating explanations for One-Class SVMs and Local Outlier Factor (LOF) models.

## Repository Structure

pe-framework/
├── requirements.txt           # Python dependencies
├── data/
│   ├── __init__.py
│   └── loader.py              # Centralized data ingestion and preprocessing logic
├── explainers/
│   ├── __init__.py
│   ├── pe_iforest.py          # Dedicated Isolation Forest vectorization
│   └── generic_explainer.py   # Model-agnostic wrapper supporting iForest, SVM, and LOF
├── scripts/
│   ├── __init__.py
│   ├── benchmark_speed.py     # Reproduces Table III: Explanation time per 100 anomalies
│   └── benchmark_fidelity.py  # Reproduces sensitivity analysis comparing Mean, Median, and Zero baselines
├── main.py                    # Master execution orchestrator
└── README.md
Installation
Clone the repository:

``` Bash
git clone [https://github.com/Ariashakoo/PE-iForest-A-Perturbation-based-Framework-for-Explainable-Anomaly-Detection.git](https://github.com/Ariashakoo/PE-iForest-A-Perturbation-based-Framework-for-Explainable-Anomaly-Detection.git)
cd pe-framework
```
Install the dependencies:

``` Bash
pip install -r requirements.txt
```
Dataset Configuration
Create a directory named datasets/ in the root of the repository and place the following required benchmark files inside:

breast-cancer.txt

arrhythmia.data

creditcard.csv

covtype.libsvm.binary

Note: If executing within a Google Colab environment, you may modify the DATA_PATH variable in main.py to point directly to your mounted Google Drive directory.

Usage
To execute the full benchmarking suite, run the orchestrator script:

```Bash
python main.py
```
Execution Details
The pipeline will execute two distinct phases:

Speed Benchmarking (benchmark_speed.py)
The script loads the benchmark datasets, applies standard scaling, and initializes three distinct outlier detection models (Isolation Forest, One-Class SVM, and Local Outlier Factor). It isolates a set of anomalies and measures the precise time required to generate explanations using the vectorized batching technique. Results are normalized to represent the total execution time (in seconds) per 100 anomalies.

Sensitivity and Fidelity Analysis (benchmark_fidelity.py)
Utilizing the Credit Card Fraud dataset, this script isolates the top 100 anomalies detected by an Isolation Forest. It then conducts a sensitivity analysis to determine how the choice of baseline replacement value (Global Mean, Global Median, or Zero) impacts the overall fidelity score. Fidelity is measured by calculating the maximum positive gain (movement towards normality) in the model's decision function after perturbation.


Citation
If you find this code useful in your research, please consider citing the paper:
```
تکه‌کد
@INPROCEEDINGS{11620789,
  author={Shakoo, Aria and Riahi-Madvar, Mahboobeh},
  booktitle={2026 International Interdisciplinary Conference on Artificial Intelligence: Engineering, Health, Finance and Humanities (IICAI)}, 
  title={PE-iForest: A Perturbation-based Framework for Explainable Anomaly Detection}, 
  year={2026},
  volume={},
  number={},
  pages={1-6},
  keywords={Modeling;Anomaly detection;Printing;Forests;Breast cancer;Algorithms;Conferences;Arrhythmia;Machine learning;Timing;anomaly detection;explainable AI (XAI);Isolation Forest;perturbation explanation;SHAP;DIFFI;scalability},
  doi={10.1109/IICAI70155.2026.11620789}
}

```
