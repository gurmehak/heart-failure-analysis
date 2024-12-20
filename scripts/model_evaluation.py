# model_evaluation.py
# author: Yuhan Fan
# date: 2024-12-06

import pickle
import click
import os
import pandas as pd
import numpy as np
import os
from sklearn import set_config
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score


@click.command()
@click.option('--scaled-test-data', type=str, help="Path to scaled test data")
@click.option('--pipeline-from', type=str, help="Path to directory where the fit pipeline object lives")
@click.option('--results-to', type=str, help="Path to directory where the table will be written to")

@click.option('--seed', type=int, help="Random seed", default=123)
def main(scaled_test_data, pipeline_from, results_to, seed):
    '''Evaluates the health failure classifier on the test data 
    and saves the evaluation results.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # read in data & cancer_fit (pipeline object)
    heart_failure_test = pd.read_csv(scaled_test_data)
    with open(pipeline_from, 'rb') as f:
        heart_failure_fit = pickle.load(f)



    # Confusion Matrix
    heart_failure_predictions = heart_failure_test.assign(
        predicted=heart_failure_fit.predict(heart_failure_test)
    )
    
    cm_crosstab = pd.crosstab(heart_failure_predictions['DEATH_EVENT'], 
                              heart_failure_predictions['predicted'], 
                              rownames=["Actual"], 
                              colnames=["Predicted"]
                             )

    accuracy = accuracy_score(heart_failure_predictions['DEATH_EVENT'], heart_failure_predictions['predicted'])
    precision = precision_score(heart_failure_predictions['DEATH_EVENT'],heart_failure_predictions['predicted'])
    recall = recall_score(heart_failure_predictions['DEATH_EVENT'], heart_failure_predictions['predicted'])
    f1 = f1_score(heart_failure_predictions['DEATH_EVENT'], heart_failure_predictions['predicted'])
    
    cm_crosstab.to_csv(os.path.join(results_to, "confusion_matrix.csv"))
    test_scores = pd.DataFrame({'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': [f1]})
    test_scores.to_csv(os.path.join(results_to, "test_scores.csv"), index=False)

if __name__ == '__main__':
    main()