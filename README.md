# Heart Failure Analysis

-   contributors:Yuhan Fan, Gurmehak Kaur, Ke Gao, Merari Santana

## About

In Milestone project 1, we attempt to build a classification model using logistic regression algorithm to predict patient mortality risk after surviving a heart attack using their medical records. Using patient test results, the final classifier achieves an accuracy of 81.6%. The model’s precision of 70.0% suggests it is moderately conservative in predicting the positive class (death), minimizing false alarms. More importantly, the recall of 73.68% ensures the model identifies the majority of high-risk patients, reducing the likelihood of missing true positive cases, which could have serious consequences. The F1-score of 0.71 reflects a good balance between precision and recall, highlighting the model’s robustness in survival prediction. While promising, further refinements are essential for more reliable predictions and effectively early intervention.

The data set that was used in this project is created by D. Chicco, Giuseppe Jurman in 2020. It was sourced from the UCI Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records). Each row in the data set represents the medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 13 clinical features（age, anaemia, diabetes, platelets, etc.).

## Dependencies

-   `conda` (version 23.9.0 or higher)
-   `conda-lock` (version 2.5.7 or higher)
-   `jupyterlab` (version 4.0.0 or higher)
-   `nb_conda_kernels` (version 2.3.1 or higher)
-   Python and packages listed in [`environment.yml`](environment.yml)

## Usage

First time running the project, run the following from the root of this repository:

``` bash
conda-lock install --name heart-failure-analysis conda-lock.yml
```

To run the analysis, run the following from the root of this repository:

``` bash
jupyter lab 
```

Open `notebooks/heart-failure-analysis.ipynb` in Jupyter Lab and under Switch/Select Kernel choose "Python [conda env:heart_failure_analysis_project]".

Next, under the "Kernel" menu click "Restart Kernel and Run All Cells...".

## License

This dataset is licensed under a [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/legalcode).

If re-using/re-mixing please provide attribution and link to this webpage. The software code contained within this repository is licensed under the MIT license. See [the license file](LICENSE.md) for more information.

## References

Chicco, D., Jurman, G. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Med Inform Decis Mak 20, 16 (2020). <https://doi.org/10.1186/s12911-020-1023-5>

Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.” University of California, Irvine, School of Information; Computer Sciences. <http://archive.ics.uci.edu/ml>.

Heart Failure Clinical Records [Dataset]. (2020). UCI Machine Learning Repository. <https://doi.org/10.24432/C5Z89R>.
