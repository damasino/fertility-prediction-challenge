"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the predict_outcomes function. 

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).
 
The script can be run from the command line using the following command:

python script.py input_path 

An example for the provided test is:

python script.py data/test_data_liss_2_subjects.csv
"""

import os
import sys
import argparse
import pandas as pd
from joblib import load

parser = argparse.ArgumentParser(description="Process and score data.")
subparsers = parser.add_subparsers(dest="command")

# Process subcommand
process_parser = subparsers.add_parser("predict", help="Process input data for prediction.")
process_parser.add_argument("input_path", help="Path to input data CSV file.")
process_parser.add_argument("--output", help="Path to prediction output CSV file.")

# Score subcommand
score_parser = subparsers.add_parser("score", help="Score (evaluate) predictions.")
score_parser.add_argument("prediction_path", help="Path to predicted outcome CSV file.")
score_parser.add_argument("ground_truth_path", help="Path to ground truth outcome CSV file.")
score_parser.add_argument("--output", help="Path to evaluation score output CSV file.")

args = parser.parse_args()


def predict_outcomes(df):
    """Process the input data and write the predictions."""

    # The predict_outcomes function accepts a Pandas DataFrame as an argument
    # and returns a new DataFrame with two columns: nomem_encr and
    # prediction. The nomem_encr column in the new DataFrame replicates the
    # corresponding column from the input DataFrame. The prediction
    # column contains predictions for each corresponding nomem_encr. Each
    # prediction is represented as a binary value: '0' indicates that the
    # individual did not have a child during 2020-2022, while '1' implies that
    # they did.

      fertIntCols=['cf08a128','cf09b128','cf10c128', 'cf11d128', 'cf12e128', 'cf13f128', 
              'cf14g128','cf15h128','cf16i128','cf17j128', 'cf18k128', 'cf19l128']
    df2 = df.loc[:, fertIntCols]
    dataFertInt = df2.fillna(method='ffill', axis=1)
    df.loc[:,'cf19l128'] = dataFertInt['cf19l128']

    fertNumCols=['cf08a129','cf09b129','cf10c129', 'cf11d129', 'cf12e129', 'cf13f129', 
              'cf14g129','cf15h129','cf16i129','cf17j129', 'cf18k129', 'cf19l129']

    df2 = df.loc[:, fertNumCols]
    dataFertNum = df2.fillna(method='ffill', axis=1)
    df.loc[:,'cf19l129'] = dataFertNum['cf19l129']

    fertSoonCols=['cf08a130','cf09b130','cf10c130', 'cf11d130', 'cf12e130', 'cf13f130', 
             'cf14g130','cf15h130','cf16i130','cf17j130', 'cf18k130', 'cf19l130']
    df2 = df.loc[:, fertSoonCols]
    dataFertSoon = df2.fillna(method='ffill', axis=1)
    df.loc[:,'cf19l130'] = dataFertSoon['cf19l130']


    kidCols = ['aantalki2007','aantalki2008','aantalki2009','aantalki2010','aantalki2011','aantalki2012',
                'aantalki2013', 'aantalki2014', 'aantalki2015', 'aantalki2016', 'aantalki2017', 
                'aantalki2018']

    df2 = df.loc[:, kidCols]
    dataKids = df2.fillna(method='ffill', axis=1)
    df.loc[:,'aantalki2018'] = dataKids['aantalki2018']
    df.loc[:,'aantalki2017'] = dataKids['aantalki2017']

    relationCols=['cf08a029','cf09b029','cf10c029', 'cf11d029', 'cf12e029', 'cf13f029', 
             'cf14g029','cf15h029','cf16i029','cf17j029', 'cf18k029', 'cf19l029']
    df2 = df.loc[:, relationCols]
    relationSoon = df2.fillna(method='ffill', axis=1)
    df.loc[:,'cf19l029'] = relationSoon['cf19l029']
    
    # Dictionary used
    dict_kids = {'None': 0, 'One child': 1, 'Two children': 2, 'Three children': 3, 'Four children': 4, 'Five children': 5, 'Six children': 6}
    
    # Keep 
    keepcols = ['positie2019','positie2018','gebjaar', 'geslacht','aantalhh2019', 'sted2019', 
            'nettohh_f2019', 'oplmet2019', 'herkomstgroep2019', 'cf19l128','cf19l129',
            'cf19l130', 'cf19l131','cf19l132','woning2019', 'woning2018', 
            'cf19l456', 'cf19l457', 'cf19l458', 'cf19l459', 'cw19l522', 'cr19l143', 
            'cf19l483', 'cf19l484', 'cf19l485', 'cf19l486', 'cf19l487', 'cf19l488',
           'wave2008', 'wave2014', 'wave2019','aantalki2017','aantalki2018',
            'partner2018','partner2019', 'belbezig2019','belbezig2018','ch19l178',
           'cp19k118', 'cp19k021', 'cp19k056', 'cf19l029', 'burgstat2019','woonvorm2019']
    results = df[["nomem_encr"]]
    
    df = df.loc[:, keepcols]
    
    df["aantalki2018"] = df["aantalki2018"].map(dict_kids)
    df["aantalki2017"] = df["aantalki2017"].map(dict_kids)
    #Create new variables about changes in partner status or number of kids (indicators)
    df['change_kids'] = (df['aantalki2018'].fillna(-1) != df['aantalki2017'].fillna(-1)) & (~df['aantalki2018'].isna()) & (~df['aantalki2017'].isna())
    #Change partner status
    df['change_partner'] = (df['partner2019'].fillna(-1) != df['partner2018'].fillna(-1)) & (~df['partner2019'].isna()) & (~df['partner2018'].isna())
    #Change in household position
    df['change_householdPos'] = (df['positie2019'].fillna(-1) != df['positie2018'].fillna(-1)) & (~df['positie2019'].isna()) & (~df['positie2018'].isna())
    #Change in employment
    df['change_jobs'] = (df['belbezig2019'].fillna(-1) != df['belbezig2018'].fillna(-1)) & (~df['belbezig2019'].isna()) & (~df['belbezig2018'].isna())
    #Change in housing
    df['change_house'] = (df['woning2019'].fillna(-1) != df['woning2018'].fillna(-1)) & (~df['woning2019'].isna()) & (~df['woning2018'].isna())

    wave_cols = ['wave2008', 'wave2014', 'wave2019']#
    df[wave_cols] = df[wave_cols].fillna(0)
    df[wave_cols] = df[wave_cols].astype('object')

    # make sure missing child birth age is filled with different value
    child_cols = ['cf19l456', 'cf19l457','cf19l458','cf19l459']
    df[child_cols] = df[child_cols].fillna(0)
    df[child_cols] = df[child_cols].astype('object')
                            
    # Load your trained model from the models directory
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")
    model = load(model_path)

    # Use your trained model for prediction
    results.loc[:, "prediction"] = model.predict(df)

    #If you use predict_proba to get a probability and a different threshold
    #df["prediction"] = (df["prediction"] >= 0.5).astype(int)
    return results                    


def predict(input_path, output):
    if output is None:
        output = sys.stdout
    df = pd.read_csv(input_path, encoding="latin-1", encoding_errors="replace", low_memory=False)
    predictions = predict_outcomes(df)
    assert (
        predictions.shape[1] == 2
    ), "Predictions must have two columns: nomem_encr and prediction"
    # Check for the columns, order does not matter
    assert set(predictions.columns) == set(
        ["nomem_encr", "prediction"]
    ), "Predictions must have two columns: nomem_encr and prediction"

    predictions.to_csv(output, index=False)


def score(prediction_path, ground_truth_path, output):
    """Score (evaluate) the predictions and write the metrics.
    
    This function takes the path to a CSV file containing predicted outcomes and the
    path to a CSV file containing the ground truth outcomes. It calculates the overall 
    prediction accuracy, and precision, recall, and F1 score for having a child 
    and writes these scores to a new output CSV file.

    This function should not be modified.
    """

    if output is None:
        output = sys.stdout
    # Load predictions and ground truth into dataframes
    predictions_df = pd.read_csv(prediction_path)
    ground_truth_df = pd.read_csv(ground_truth_path)

    # Merge predictions and ground truth on the 'id' column
    merged_df = pd.merge(predictions_df, ground_truth_df, on="nomem_encr", how="right")

    # Calculate accuracy
    accuracy = len(
        merged_df[merged_df["prediction"] == merged_df["new_child"]]
    ) / len(merged_df)

    # Calculate true positives, false positives, and false negatives
    true_positives = len(
        merged_df[(merged_df["prediction"] == 1) & (merged_df["new_child"] == 1)]
    )
    false_positives = len(
        merged_df[(merged_df["prediction"] == 1) & (merged_df["new_child"] == 0)]
    )
    false_negatives = len(
        merged_df[(merged_df["prediction"] == 0) & (merged_df["new_child"] == 1)]
    )

    # Calculate precision, recall, and F1 score
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0
    # Write metric output to a new CSV file
    metrics_df = pd.DataFrame({
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1_score]
    })
    metrics_df.to_csv(output, index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.command == "predict":
        predict(args.input_path, args.output)
    elif args.command == "score":
        score(args.prediction_path, args.ground_truth_path, args.output)
    else:
        parser.print_help()
        predict(args.input_path, args.output)  
        sys.exit(1)
