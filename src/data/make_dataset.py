import pandas as pd
from src.features import build_features
from sklearn.model_selection import train_test_split

def make_dataset():
    """
    This function laods the raw data, builds some features and saves the df.
    It is not meant to be called but once to produce the dataset.
    """
    raw_data = pd.read_csv("C:/Users/groes/OneDrive/Documents/701CW_2/Intro_to_AI/data/raw/df_all_data_w_desc_2021-06-22.csv")
    df = build_features.build_features(raw_data)
    df.to_csv("C:/Users/groes/OneDrive/Documents/701CW_2/Intro_to_AI/data/processed/4.0-processed_data_w_listprice.csv")

#Test:
#make_dataset()
#new_df = pd.read_csv("C:/Users/groes/OneDrive/Documents/701CW_2/Intro_to_AI/data/processed/3.0-processed_data.csv")

###### SPLITTING DATASET INTO TRAINING AND VALIDATION #######
# Create an array that for each home makes a boolean. The number of trues are percentage_trainingset %
def create_training_validation_set(df, percentage_trainingset):    
    mask = np.random.rand(len(df)) < percentage_trainingset

    trainDF = pd.DataFrame(df[mask])
    validationDF = pd.DataFrame(df[~mask])

    print(f"Training DF: {len(trainDF)}")
    print(f"Validation DF: {len(validationDF)}")
    
    return trainDF, validationDF 


######## K-FOLD CROSS VALIDATION ########
# Function takes a df and and k (integer) as inputs and returns k training and validation sets
def create_kfold_sets(df, k):
    # Using k fold validation because the dataset is small
    # Shuffling to get representative samples
    df = df.reindex(np.random.permutation(df.index))
    kf = KFold(k)
    fold = 1
    dict_of_sets = {}
    for train_index, validate_index in kf.split(df):
        trainDF = pd.DataFrame(df.iloc[train_index, :])
        validateDF = pd.DataFrame(df.iloc[validate_index])
        dict_name_train = "Fold" + str(fold) + "_training"
        dict_name_validate = "Fold" + str(fold) + "_validation"
        dict_of_sets[dict_name_train] = trainDF
        dict_of_sets[dict_name_validate] = validateDF 
        fold += 1
    return dict_of_sets


def partition_dataset(df, target_name, training_size):
    """
    df : Pandas df
        DataFrame containing the processed data, i.e. all variables and all samples
    target_name : STR
        Name of target variable
    training_ratio : FLOAT
        The percentage of samples to put in training data, e.g. 0.8

    """    
    x = df.drop(target_name, axis=1)
    y = df[target_name]
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=training_size, random_state=101
        )
    
    return x_train, x_test, y_train, y_test
    

"""
# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ """Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed)."""
"""
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
"""