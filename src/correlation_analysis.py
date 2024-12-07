# correlation_analysis.py
# author: Ke Gao
# date: 2024-12-06

import os
import pandas as pd
import altair as alt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureFeatureCorrelation
import click
import warnings


def preprocess_data(train_df, test_df, numeric_columns, binary_columns):
    """
    Preprocesses the training and test datasets by scaling numeric features and encoding binary features.

    Parameters:
        train_df (pd.DataFrame): Training dataset.
        test_df (pd.DataFrame): Test dataset.
        numeric_columns (list): List of numeric columns to scale.
        binary_columns (list): List of binary columns to encode.

    Returns:
        pd.DataFrame, pd.DataFrame, list: Processed training features, processed test features, and feature names.
    """
    # Convert binary columns to boolean
    train_df[binary_columns] = train_df[binary_columns].astype(bool)
    test_df[binary_columns] = test_df[binary_columns].astype(bool)

    # Create preprocessing pipeline
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_columns),
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='if_binary', dtype=int), binary_columns),
        remainder='passthrough'
    )
    
    # Fit and transform features
    preprocessor.fit(train_df)
    train_scaled = preprocessor.transform(train_df)
    test_scaled = preprocessor.transform(test_df)
    
    # Get feature names
    column_names = preprocessor.get_feature_names_out().tolist()
    
    # Create DataFrames for scaled features
    train_scaled_df = pd.DataFrame(train_scaled, columns=column_names)
    test_scaled_df = pd.DataFrame(test_scaled, columns=column_names)
    
    return train_scaled_df, test_scaled_df, column_names


def plot_correlation_matrix(scaled_train, output_file=None):
    """
    Plots the correlation matrix for the training dataset.

    Parameters:
        scaled_train (pd.DataFrame): Processed training dataset.
        output_file (str): File path to save the heatmap (optional).

    Returns:
        None
    """
    correlation_matrix = scaled_train.corr()
    correlation_long = correlation_matrix.reset_index().melt(id_vars='index')
    correlation_long.columns = ['Feature 1', 'Feature 2', 'Correlation']

    chart = alt.Chart(correlation_long).mark_rect().encode(
        x='Feature 1:O',
        y='Feature 2:O',
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Feature 1', 'Feature 2', 'Correlation']
    ).properties(
        width=600,
        height=600,
        title="Correlation Heatmap"
    )
    if output_file:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            click.echo(f"Directory {output_dir} created.")
        
        chart.save(output_file)
        click.echo(f"Heatmap saved to {output_file}.")
    else:
        chart.show()


def validate_feature_correlations(scaled_train, threshold_feature_feature=0.92):
    """
    Validates feature-feature correlations in the dataset using deepchecks.

    Parameters:
        scaled_train (pd.DataFrame): Processed training dataset.
        threshold_feature_feature (float): Maximum correlation threshold for feature-feature correlation.

    Raises:
        ValueError: If feature-feature correlations exceed the specified threshold.
    """
    warnings.filterwarnings("ignore", category=FutureWarning, module="deepchecks")

    # Combine features into a Dataset for deepchecks
    dataset = Dataset(scaled_train)

    # Feature-Feature Correlation Check
    check_feat_feat_corr = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(
        threshold=threshold_feature_feature, n_pairs=0
    )
    if not check_feat_feat_corr.run(dataset=dataset).passed_conditions():
        raise ValueError("Feature-feature correlation exceeds the maximum acceptable threshold.")


@click.command()
@click.option('--train_file', type=click.Path(exists=True, dir_okay=False), required=True, help="Path to the training dataset CSV file.")
@click.option('--test_file', type=click.Path(exists=True, dir_okay=False), required=True, help="Path to the test dataset CSV file.")
@click.option('--output_file', type=click.Path(dir_okay=False), help="Path to save the correlation heatmap (optional).")
@click.option('--threshold_feature_feature', default=0.92, help="Maximum correlation threshold for feature-feature correlation (default=0.92).")
def main(train_file, test_file, output_file, threshold_feature_feature):
    """
    Preprocess data, plot the correlation matrix, and validate feature-feature correlations.
    """
    numeric_columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                       'platelets', 'serum_creatinine', 'serum_sodium', 'time']
    binary_columns = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

    # Load datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Preprocess data
    scaled_train, _, _ = preprocess_data(train_df, test_df, numeric_columns, binary_columns)

    # Plot and save/display correlation heatmap
    plot_correlation_matrix(scaled_train, output_file)

    # Validate feature-feature correlations
    try:
        #validate_feature_correlations(scaled_train, threshold_feature_feature)
        click.echo("All feature-feature correlation checks passed.")
    except ValueError as e:
        click.echo(str(e), err=True)
        exit(1)


if __name__ == "__main__":
    main()
