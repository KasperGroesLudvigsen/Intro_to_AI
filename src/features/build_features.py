
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import statistics
import math
from sklearn.decomposition import PCA

COLUMNS_TO_USE = ["Home_type", "rooms", "home_size_m2", "lotsize_m2", 
                  "expenses_dkk", "floor_as_int", "balcony", "zipcodes", 
                  "m2_price", "age", "zipcode_avg_m2_price", "list_price_dkk"]


# remove 'Hometype_Andelsbolig' ? 
UNUSED_VARIABLES = ['ID', 'home_url_boliga', 'home_url_realtor',
       'street_name_number', 'zip_code_town', 'description_of_home', 'floor']


def build_features(df, zip_threshold=0.01):
    """
    Wrapper for the methods that engineers new features

    zip_threshold (float) : 
        If a zip code accounts for less observations than
        the threshold (0.01 = 1%) it will be grouped with another zip code in the
        neighborhood
    """
    
    df = add_balcony_variable(df)
    df = make_floor_int(df)
    df = add_zip_code_variable(df, zip_threshold)
    df = make_m2_price(df)
    df = add_neighboorhood_avg_m2_price(df)
    df = add_age(df)
    df = df.loc[df['age'] < 450]
    df = remove_nans(df)
    df = onehot_encode_variables(df)
    df = remove_unused_variables(df)
    df.drop(
        ["retrieved_on"], axis=1, inplace=True
        )
    return df

def remove_coorporative_homes(df):
    num_datapoints = len(df)
    index = df[df["Home_type"] == "Andelsbolig"].index
    df.drop(index, inplace=True)
    print("{} cooperative dwellings removed".format(num_datapoints-len(df)))
    df = df.reset_index(drop=True)
    return df

def add_balcony_variable(df):
    """
    This function creates a variable "balcony"
    The function adds a column of which each element is an integer:
    0 = no balcony, 1 = possibility of building balcony, 2 = balcony
    It is an ordered variable, not categorical, as balcony is better,
    therefore larger, than no balcony
    """
    list_does_home_have_balcony = []
    num_homes_with_no_description = 0

    for description in df["description_of_home"]: #df.iloc[:, description_column_index]:
        if type(description) != str:
            list_does_home_have_balcony.append(0)
            num_homes_with_no_description += 1
            continue
        
        # If the home does not have a balcony but there is an option of adding
        # one, the realtor will typically write "mulighed for altan" or 
        # "altan projekt" (balcony project)
        if "mulighed for altan" in description or "altanprojekt" in description or "altan projekt" in description:
            list_does_home_have_balcony.append(1)
            continue
        if "altan" in description or "terrasse" in description or "tagterrasse" in description:
            list_does_home_have_balcony.append(2)
            continue
        
        list_does_home_have_balcony.append(0)
        
    df["balcony"] = list_does_home_have_balcony
    print("{} homes had no description".format(num_homes_with_no_description))
    return df


def make_floor_int(df):
    floor_as_int = []
    for i in range(len(df)):
        # Only house types "villalejlighed" (flat in villa) and "ejerlejlighed" (flat)
        # has floor numbers 
        if df["Home_type"][i] == "Villalejlighed" or df["Home_type"][i] == "Ejerlejlighed":
            try:
                floor_as_int.append(int(df["floor"][i][0]))
            except:
                median_value = int(round(statistics.median(floor_as_int)))
                floor_as_int.append(median_value)
                print("Error converting floor to int in line {}. Inserting median value: {}".format(i, median_value))
        else:
            floor_as_int.append(0)
    df["floor_as_int"] = floor_as_int
    return df #floor_as_int


def get_zips_to_be_grouped(df, threshold):
    """
    Helper function for add_zip_code_variable()
    
    If an area has more than one zipcode (e.g. Frederiksberg C), those of 
    the zipcodes that account for less than 1 % (per default) of datapoints,
    all zip codes within the area will be grouped into 1 zipcode

    Parameters
    ----------
    df : Pandas dataframe
        Df with data
        
    threshold : Float
        If a zip code accounts for fewer than 'threshold' datapoints, it will
        be grouped into one

    Returns
    -------
    zips_to_be_grouped : SET

    """
    zip_code_occurences = df.zip_code_town.value_counts()
    zips_to_be_grouped = []

    threshold = len(df) * threshold
    print("Grouping zip codes with fewer than {} datapoints".format(threshold))
    for i in range(len(zip_code_occurences)):
        area = zip_code_occurences.index[i]

        if zip_code_occurences[i] < threshold:
            zips_to_be_grouped.append(area)
    
    # using set() for higher look up speed
    return set(zips_to_be_grouped)


def add_zip_code_variable(df, threshold=0.01):
    """
    Some zip codes in Copenhagen cover very small areas whereas others cover
    very large areas. The zip codes covering small areas are not well repre-
    sented in the dataset. Therefore, I group zip codes that have few datapoints
    in groups that represent the area of Copenhagen the zip code belongs to. 
    E.g. 

    Parameters
    ----------
    df : PANDAS DATAFRAME
        df with data
        
    threshold : FLOAT
        If a zip code accounts for fewer than 'threshold' datapoints, it will
        be grouped into one

    Returns
    -------
    Enriched df

    """
    # Identifying zip codes that account for fewer observations than the threshold
    zips_to_be_grouped = get_zips_to_be_grouped(df, threshold)
    zipcodes = []
    
    for i in range(len(df)):
        area = df.zip_code_town[i] # e.g. "2000 Frederiksberg"
        if area in zips_to_be_grouped:
            if "København V" in area:
                zipcodes.append("1600")
            if "København K" in area:
                zipcodes.append("1300")
            if "Frederiksberg C" in area:
                zipcodes.append("1900")
            if "Rødovre" in area:
                zipcodes.append("2610")
        else:
            # The first word of the string 'area' is the zipcode
            zipcode = area[:4]
            try:
                int(zipcode)
            except:
                print("{} in row {} of zip_code_town is not a number. Appending NaN".format(zipcode, i))
                zipcodes.append("NaN")
            zipcodes.append(zipcode)
           
    assert len(zipcodes) == len(df)
    df["zipcodes"] = zipcodes
    return df


def make_m2_price(df):
    df["m2_price"] = df.list_price_dkk / df.home_size_m2
    return df

    
def add_neighboorhood_avg_m2_price(df):
    grouped_df = df.groupby("zipcodes")
    mean_df = grouped_df.mean()
    mean_df = mean_df.reset_index()
    zipcode_avg_m2_price = []
    for i in range(len(df)):
        #print(i)
        zipcode = df["zipcodes"][i]
        #print("zipcode")
        #print(zipcode)
        index = mean_df.index[mean_df["zipcodes"] == zipcode] #mean_df["zipcodes"].index(zipcode)
        avg_price = float(mean_df.iloc[index, -1])
        #print(avg_price)
        zipcode_avg_m2_price.append(avg_price)
    df["zipcode_avg_m2_price"] = zipcode_avg_m2_price
    return df 

def add_age(df):
    # Adding age feature and removing "built_year"
    df["age"] = 2021 - df["built_year"]
    df.drop(["built_year"], axis=1, inplace=True)
    return df

def remove_nans(df):
    # Using heuristics to replace nans with reasonable values
    for index in df.index:
        if math.isnan(df.loc[index, "rooms"]):
            if df.loc[index, "home_size_m2"] < 40:
                df.loc[index, "rooms"] = 1 #df.at[home_size_m2_idx, i] = 1
            elif df.loc[index, "home_size_m2"] < 70:
                df.loc[index, "rooms"] = 2
            elif df.loc[index, "home_size_m2"] < 100:
                df.loc[index, "rooms"] = 3
            else: 
                df.loc[index, "rooms"] = 4
        if math.isnan(df.loc[index, "lotsize_m2"]):
            if df.loc[index, "Home_type"] == "Ejerlejlighed" or df.loc[index, "Home_type"] == "Andelsbolig":
                df.loc[index, "lotsize_m2"] = 0
            else:
                df.loc[index, "lotsize_m2"] = df["lotsize_m2"].mean()
        if math.isnan(df.loc[index, "expenses_dkk"]):
            df.loc[index, "expenses_dkk"] = df["expenses_dkk"].mean()
        if math.isnan(df.loc[index, "age"]):
            df.loc[index, "age"] = round(df["age"].mean())
            
    # Removing observations for which the any of the relevant variables are null
    variables_to_remove_if_null = ["list_price_dkk", "rooms", "home_size_m2"]
    for variable in variables_to_remove_if_null:
        nulls = pd.isnull(df[variable])
        df = df[~nulls]
        
    return df

def onehot_encode_variables(df):
    onehot_encoded_variables = []
    
    # One hot encoding zipcodes
    zipcodes_onehot = pd.get_dummies(df.zipcodes, prefix="Zipcode")
    df = pd.concat([df, zipcodes_onehot], axis=1)
    onehot_encoded_variables.append("zipcodes")
    
    # One hot encoding home_type
    hometype_onehot = pd.get_dummies(df.Home_type, prefix="Hometype")
    df = pd.concat([df, hometype_onehot], axis=1)
    onehot_encoded_variables.append("Home_type")
    
    # Removing variables that have been one hot encoded
    df = df.drop(onehot_encoded_variables, axis=1)
    
    return df

def remove_unused_variables(df):
    # Drop the unused variables
    df = df.drop(UNUSED_VARIABLES, axis=1)
    return df

def pca_dim_reduction(df, explained_variance=0.85):
    """
    Reduces the dimensionality of the dataset (df) into the smallest number
    of components required to explain a given level of variance.
    
    df (Pandas DataFrame) : the training data df
    explained_variance (float) : the level of variance that the components must
    be able to explain
    
    Adapted from Baruah (2020)
    https://towardsdatascience.com/one-hot-encoding-standardization-pca-data-preparation-steps-for-segmentation-in-python-24d07671cf0b
    """
    # Loop Function to identify number of principal components that explain at least 85% of the variance
    for comp in range(df.shape[1]):
        pca = PCA(n_components= comp, random_state=42)
        pca.fit(df)
        comp_check = pca.explained_variance_ratio_
        final_comp = comp
        if comp_check.sum() > explained_variance:
            break
            
    final_PCA = PCA(n_components=final_comp, random_state=42)
    final_PCA.fit(df)
    cluster_df = final_PCA.transform(df)
    print("Using {} components, we can explain {}% of the variance in the original data.".format(final_comp,comp_check.sum()*100))
    return cluster_df








    