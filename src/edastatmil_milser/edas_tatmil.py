############################################################################
#                       Functions for complete EDA                         #
# ------------------------------------------------------------------------ #
#   - get_column_type for discrimine variables in numerical or categorical #
#   - explore for get basic info and both list of numerical and            #
#     categorical variables                                                #
#   - FindDuplicates for find and drop (or not) duplicates based in one    #
#     given variable                                                       #
#   - univariate_hist for plot histograms of a list of variables_          #
#   - univariate_histbox for plot histograms and boxplotof a list of       #
#     variables                                                            #
#   - multivariate_barplots for multivariate barplots by count or mean     #
#   - factorize_categorical for factorize list of categorical variables    #
#   - correlation_matrix for build correlation matix in heatmap            #
#   - numerical_box for plot boxplot of numerical variables from a list    #
#   - outliers_iqr for drop or sustitute outliers in a iqr*sigma           #
#   - splitter for split in train and test all csv from a root and save    #
#   - normalize for normalice X_train and X_test by any scaler of          #
#     sklearn.preprocessing module                                         #
#   - feature_sel for make feature selection by any method and with any    #
#     test included in sklearn.feature_selection                           #
############################################################################

############################################################################
#                                Authors                                   #
#                         @TatianaCC  @milser                              #
############################################################################

###############################Imports#####################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import importlib
import tabulate
from typing import List, Tuple, Literal, Union, Dict

############################################################################


def get_column_type(series: str) -> Literal['Numeric','Categorical']:
    """
    Determines the column type of a pandas series.

    This function takes a pandas series as input and determines whether the data
    in the series are `numeric` or `categorical`.

    Parameters::

        series (pandas.Series): The pandas series from which the column type will be determined.

    Returns::
    
        str: 'Numeric': If they are numeric.
             'Categorical': If they are categorical.
    """
    series_ = series.copy()
    
    if pd.api.types.is_numeric_dtype(series_):
        return 'Numeric'
    else:
        return 'Categorical'

def explore(data_frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Explores a pandas DataFrame and prints information about its characteristics.

    This function `prints` useful `information` about the provided pandas DataFrame,
    including the number of `rows` and `columns`, the count of `null` and `non-null` values,
    the `data type` of each column, and whether each column is `categorical` or `numeric`.

    Parameters::

        data_frame (pandas.DataFrame): The pandas DataFrame to be explored.

    Returns:

        tuple: A tuple containing two lists. The first list contains the names of categorical columns.
        The second list contains the names of numeric columns.
    """
    data_frame_ = data_frame.copy()
    
    # Get shape
    num_rows_, num_columns_ = data_frame_.shape
    print('Rows:', num_rows_)
    print('Columns:', num_columns_)

    # Get null and type info
    column_data_ = pd.DataFrame({
        'Non-Null Count': data_frame_.count(),
        'Null Count': data_frame_.isnull().sum(),
        'Data Type': data_frame_.dtypes
    }) 
    # Add if a variable is categorical or numerical  
    column_data_['Data Category'] = data_frame_.apply(get_column_type)
    print(tabulate(column_data_, headers='keys', tablefmt='pretty'))

    # Get list of categorical and numerical variables_
    categorical_columns_ = list(column_data_[column_data_['Data Category'] == 'Categorical'].index)
    numeric_columns_ = list(column_data_[column_data_['Data Category'] == 'Numeric'].index)
    
    return categorical_columns_, numeric_columns_

def FindDuplicates(data_frame: pd.DataFrame, id_col: str, Drop: bool = False) -> Union[pd.DataFrame, int]:
    """
    Finds and handles duplicates in a pandas DataFrame.

    This function finds and handles `duplicates` in a pandas DataFrame,
    either by `removing` them or simply `counting` them.

    Parameters::

        data_frame (pandas.DataFrame): The pandas DataFrame to be searched for duplicates.
        id_col (str): The name of the column to be used to identify duplicates.
        Drop (bool, optional): A boolean indicator specifying whether duplicates should be removed.
        If True, duplicates will be removed; if False, only counted.

    Returns::

        Union[pandas.DataFrame, int]: If Drop is True, returns a new DataFrame without duplicates and the number of duplicates removed.
        If Drop is False, returns the original DataFrame and 0.
    """
    data_frame_ = data_frame.copy()
    id_col_ = id_col
    Drop_ = Drop
    
    duplicates_count_:int = data_frame_.duplicated(subset=[id_col_]).sum()
    removed_ = int(0)
    print(f" {duplicates_count_} duplicates have been found")
    if Drop_ == 'True':
        deduplicated_df_ = data_frame_.drop_duplicates(data_frame_.columns.difference([id_col_]))
        if len(deduplicated_df_) < len(data_frame_):
            removed_ = len(data_frame_) - len(deduplicated_df_)
            print(f" {removed_} duplicates have been removed")
        return deduplicated_df_,removed_
    else:
        return data_frame_,removed_
    
def Find_over_50_percent_value(df: pd.DataFrame) -> None:
    """
    Finds values in a DataFrame that occupy more than 50% of the table and prints their details.

    This function calculates the `occurrences` of each unique value in each `column` of the DataFrame provided.
    It then determines the percentage of each value's occurrence relative to the total number of rows in the DataFrame.
    Values with occurrences greater than `50%` are identified, and their details are printed.

    Parameters::
    
        df (pd.DataFrame): The pandas DataFrame to analyze.

    Returns::
    
        None
    """
    df_ = df.copy()
    
    columns_over_50_percent_value = {}
    no_columns_over_50_percent = True  # Variable de control

    for col in df_.columns:
        # Calculamos la suma de repeticiones de cada valor único en la columna actual
        value_counts_df = df_[col].value_counts().reset_index()
        percentages = (value_counts_df['count'] / len(df_)) * 100

        # Agregar la columna de porcentaje
        value_counts_df['percentages'] = percentages  

        # Verificar si algún valor tiene más del 50% de la tabla
        over_50_percent_values = value_counts_df[percentages > 50]
        if not over_50_percent_values.empty:
            columns_over_50_percent_value[col] = over_50_percent_values
            no_columns_over_50_percent = False  # Hay al menos una columna con más del 50% de participación

    if no_columns_over_50_percent:
        print("No columns have values occupying more than 50% of the table.")
    else:
        print("Columns with values occupying more than 50% of the table:")
        for col, values_df in columns_over_50_percent_value.items():
            print(f"Column: {col}")
            for index, row in values_df.iterrows():
                print(f"Value: {row[f'{col}']}, Count: {row['count']}, Percentage: {row['percentages']:.2f}%")
            print()

def univariate_hist(variables: List[str], data_frame: pd.DataFrame, color: str = '#1295a6', kde: bool = False) -> None:
    """
    Creates a univariate histogram for each specified variable in a pandas DataFrame.

    This function creates a `univariate` `histogram` for each specified variable in the provided pandas DataFrame.
    The histograms are displayed in a grid, with up to 3 plots per row.

    Parameters::

        variables (List[str]): List of variable names for which histograms will be created.
        data_frame (pandas.DataFrame): The pandas DataFrame containing the data.
        color (str, optional): Color for the histograms. Default is '#1295a6'.
        kde (bool, optional): Boolean indicator specifying whether to plot a kernel density estimation. Default is False.

    Returns::

        None
    """
    variables_ = variables.copy()
    data_frame_ = data_frame.copy()
    color_ = color
    kde_ = kde
    
    
    num_plots_ = len(variables_)
    num_rows_ = (num_plots_ - 1) // 3 + 1  # N rows
    
    fig_, axes_ = plt.subplots(num_rows_, 3, figsize=(15, 5*num_rows_))
    axes_ = axes_.flatten()  

    for i_, var_ in enumerate(variables_):
        ax_ = axes_[i_]
        sns.histplot(data_frame_[var_], ax=ax_, kde=kde_, color=color_)
        ax_.set_title(var_)
        ax_.set_xlabel('')
        ax_.set_ylabel('')
        
    # Remove empty axes_
    for j_ in range(i_+1, len(axes_)):
        fig_.delaxes(axes_[j_])
    
    plt.tight_layout()

def univariate_histbox(variables: List[str], data_frame: pd.DataFrame, color: str = '#1295a6') -> None:
    """
    Creates a univariate histogram and boxplot for each specified variable in a pandas DataFrame.

    This function creates a `univariate` `histogram` and `boxplot` for each specified variable in the provided pandas DataFrame.
    The plots are displayed in a grid, with up to 3 plots per row.

    Parameters::
        
        variables (List[str]): List of variable names for which plots will be created.
        data_frame (pandas.DataFrame): The pandas DataFrame containing the data.
        color (str, optional): Color for the plots. Default is '#1295a6'.

    Returns::

        None
    """    
    variables_ = variables.copy()
    data_frame_ = data_frame.copy()
    color_ = color
    
    num_variables_ = len(variables_)
    num_rows_ = math.ceil(num_variables_ / 3)
    
    _, axis_ = plt.subplots(2 * num_rows_, 3, figsize=(18, 6 * num_rows_), gridspec_kw={'height_ratios': [8, 4] * num_rows_})

    for i_, col_ in enumerate(variables_):
        row_ = i_ // 3
        col_in_row_ = i_ % 3
        
        sns.histplot(data_frame_, x=col_, ax=axis_[row_ * 2, col_in_row_], color=color_)
        sns.boxplot(data_frame_, x=col_, ax=axis_[row_ * 2 + 1, col_in_row_], color=color_)
    
    if num_variables_ % 3 != 0:
        for j_ in range(num_variables_ % 3, 3):
            axis_[num_rows_ * 2 - 1, j_].remove()
            axis_[num_rows_ * 2 - 2, j_].remove()

    plt.tight_layout()

def multivariate_barplots(df: pd.DataFrame, variable_lists: List[List[str]], y: str = 'count', palette: str = 'Set2') -> None:
    """
    Creates multivariate bar plots for the specified variables in a pandas DataFrame.

    This function creates `multivariate` bar plots for the specified variables in the provided pandas DataFrame.
    The plots are displayed one below the other.

    Parameters::

        df (pandas.DataFrame): The pandas DataFrame containing the data.
        variable_lists (List[List[str]]): List of lists containing variable names.
        Each sublist should contain three elements: the x variable, the y variable, and the hue variable.
        y (str, optional): Specifies whether to compute y values as 'count' or 'mean'. Default is 'count'.
        palette (str, optional): Color palette to use in the plots. Default is 'Set2'.

    Returns::

        None
    """
    df_ = df.copy()
    variable_lists_ = variable_lists.copy()
    y_ = y
    palette_ = palette
        
    num_plots_ = len(variable_lists_)
    _, axes_ = plt.subplots(num_plots_, 1, figsize=(10, 5 * num_plots_))

    for i_, variables_ in enumerate(variable_lists_):
        if y_ == 'count':
            df_ = df_.groupby(variables_).size().reset_index(name='count')

            x_, hue_ = variables_[0], variables_[2]
            ax_ = axes_[i_]
            sns.barplot(data=df_, x=x_, y='count', hue=hue_, ax=ax_,palette=palette_,errorbar=None)
            ax_.set_xlabel(variables_[0])
            ax_.set_ylabel('Count')
        elif y_ == 'mean':
            try:
                mean_ = df_.groupby([variables_[0], variables_[2]])[variables_[1]].mean().reset_index()
                x_, hue_ = variables_[0], variables_[2]
                ax_ = axes_[i_]
                sns.barplot(data=mean_, x=x_, y=variables_[1], hue=hue_, ax=ax_,palette=palette_,errorbar=None)
                ax_.set_xlabel(variables_[0])
                ax_.set_ylabel('Mean')
            except:
                print('y variable is not numerical')
    plt.tight_layout()

def factorize_categorical(df: pd.DataFrame, cols_to_factor: List[str]) -> pd.DataFrame:
    """
    Factorizes categorical variables in a pandas DataFrame and saves the mappings to CSV files.

    This function `factorizes` the categorical `variables` specified in a pandas DataFrame, assigning a unique integer number to each unique value of the variable.
    Additionally, it saves the mappings between the original values and the factorized values to CSV files in
        >>> ..\\data\\processed\\factorized_mapping\\

    Parameters::

        df (pd.DataFrame): The pandas DataFrame containing the data.
        cols_to_factor (List[str]): A list of column names corresponding to the categorical variables to be factorized.

    Returns::

        pd.DataFrame: The pandas DataFrame with the factorized categorical variables.

    Example::

        If 'cols_to_factor' contains ['Gender', 'Region'], and the original DataFrame contains these two columns, the function will factorize these columns and create two CSV files named 'Gender_mappings.csv' and 'Region_mappings.csv' inside the 'factorized_mapping' directory, containing the mappings for each column.
    """
    df_ = df.copy()
    cols_to_factor_ = cols_to_factor.copy()
    
    mapping_dir = '../data/processed/factorized_mapping/'

    if not os.path.exists(mapping_dir): os.makedirs(mapping_dir)
        

    for col_ in cols_to_factor_:
        mappings_df = pd.DataFrame(columns=['Original_Value', 'Factorized_Value'])
        original_col_ = f"{col_}_O"
        df_[original_col_] = df_[col_]
        df_[col_] = pd.factorize(df_[col_])[0]
        
        mappings = pd.DataFrame({'Original_Value': df_[original_col_], 'Factorized_Value': df_[col_]})
        mappings_df = pd.concat([mappings_df, mappings])
        mappings_df = mappings_df.drop_duplicates(subset=['Original_Value', 'Factorized_Value'])
        
        mappings_df.to_csv(os.path.join(mapping_dir, f'{col_}_mappings.csv'), index=False)

    return df_

def correlation_matrix(df: pd.DataFrame, variables_list: List[str]) -> None:
    """
    Creates a correlation matrix for the specified variables in a pandas DataFrame.

    This function creates a `correlation` matrix for the specified variables in the provided pandas DataFrame.
    CATEGORICAL variables are FACTORIZED before calculating the correlation.

    Parameters::

        df (pandas.DataFrame): The pandas DataFrame containing the data.
        variables_list (List[str]): List of variable names for which correlation will be calculated.

    Returns::

        None
    """
    df_ = df.copy()
    variables_list_ = variables_list.copy()
    
    fz_df_ = factorize_categorical(df_, variables_list_)
    sns.heatmap(fz_df_.corr(), annot = True, fmt = ".2f")

def numerical_box(variables: List[str], data_frame: pd.DataFrame, color: str = '#1295a6') -> None:
    """
    Crea diagramas de caja para variables numéricas en un DataFrame de pandas.

    Esta función crea `diagramas` de caja para variables `numéricas` en el DataFrame de pandas proporcionado.
    Los diagramas se muestran en una cuadrícula, con hasta 3 gráficos por fila.

    Parámetros::

        variables (List[str]): Lista de variables numéricas para las que se crearán los diagramas de caja.
        data_frame (pandas.DataFrame): El DataFrame de pandas que contiene los datos.
        color (str, opcional): Color para los diagramas de caja. Por defecto, '#1295a6'.

    Retorna::

        None
    """
    variables_ = variables.copy()
    data_frame_ = data_frame.copy()
    color_ = color
    
    num_plots_ = len(variables_)
    num_rows_ = (num_plots_ - 1) // 3 + 1  # N rows
    
    fig_, axes_ = plt.subplots(num_rows_, 3, figsize=(15, 5*num_rows_))
    axes_ = axes_.flatten()  

    for i_, var_ in enumerate(variables_):
        ax_ = axes_[i_]
        sns.boxplot(x=data_frame_[var_], ax=ax_, color=color_)
        ax_.set_title(var_)
        ax_.set_xlabel('')
        ax_.set_ylabel('')

    for j_ in range(i_+1, len(axes_)):
        fig_.delaxes(axes_[j_])
    
    plt.tight_layout()

def outliers_iqr(df: pd.DataFrame, var: str, sigma: float, Do: str = 'nothing') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies and handles outliers of a variable in a pandas DataFrame using the interquartile range (IQR).

    This function identifies `outliers` of a variable in a pandas DataFrame using the interquartile range (IQR).
    Optionally, outliers can be `removed` or `replaced` with the `mode`, `mean`, or `median` value.

    Parameters::

        df (pandas.DataFrame): The pandas DataFrame containing the data.
        var (str): Name of the variable for which outliers will be identified.
        sigma (float): Tolerance for the interquartile range (IQR).
        Do (str, optional): Action to take with the outliers.
            'nothing' (default): No action is taken.
            'drop': Outliers are DROPPED.
            'mode': Outliers are REPLACED by the MODE.
            'mean': Outliers are REPLACED by the MEAN.
            'median': Outliers are REPLACED by the MEDIAN.

    Returns::

        tuple: A tuple containing two DataFrames.
               The first DataFrame contains the identified outliers.
               The second DataFrame contains the original data with outliers treated according to the option specified in 'Do_'.
    """    
    df_ = df.copy()
    var_ = var
    sigma_ = sigma
    Do_ = Do
  
    descr_ = df_[var_].describe()
 
    iqr_ = descr_["75%"] - descr_["25%"]
    upper_l_ = descr_["75%"] + sigma_*iqr_
    lower_l_ = descr_["25%"] - sigma_*iqr_
    print(upper_l_, lower_l_)

    outliers_ = df_[(df_[var_] >= upper_l_) | (df_[var_] < lower_l_)]
    num_outliers_ = outliers_.shape[0]     

    if Do_ == 'nothing':
        print(str(num_outliers_)+' outliers have been found')
        pass
    else:
        if Do_ != 'drop':
            if Do_ == 'mode':
                replacer_ = df_[var_].mode()                    
            elif Do_ == 'mean':
                replacer_ = df_[var_].mean()
            elif Do_ == 'median':
                replacer_ = df_[var_].median()

            replace_func_ = lambda x_: x_ if lower_l_ <= x_ < upper_l_ else replacer_
            df_[var_] = df_[var_].apply(replace_func_)        
            print(str(num_outliers_) + ' outliers have been treated by replacing them with the ' + Do_)
        else:
            df_ = df_[var_].between(lower_l_, upper_l_)
            print(str(num_outliers_)+' outliers have been treated by dropping')
    return outliers_, df_
    
def splitter(origin_root: str, predictors: List[str], target: str) -> Dict[str, pd.DataFrame]:
    """
    Splits datasets into training and testing sets and saves them as CSV files in
    >>> origin_root\\SplitData\\

    This function `splits` datasets into `training` and `testing` sets using sklearn's train_test_split,
    and returns a dictionary containing the split datasets.

    Parameters::

        origin_root (str): Path of the folder containing the original CSV files.
        predictors (List[str]): List of predictor variable names.
        target (str): Name of the target variable.

    Returns::

        Dict[str, pd.DataFrame]: A dictionary containing the split datasets.
            The `keys` are the names of the datasets ("dataset_name_Xtrain").
            The `values` are the datasets themselves.
    """
    origin_root_ = origin_root
    predictors_ = predictors.copy()
    target_ = target
    
    from sklearn.model_selection import train_test_split
    csv_files_ = []
    datasets_ = {}
    for file in os.listdir(origin_root_):
        if file.endswith(".csv"):
            csv_files_.append(os.path.join(origin_root_, file))
    
    for df_root_ in csv_files_ :
        df_ = pd.read_csv(df_root_)
        X_ = df_[predictors_]
        Y_ = df_[target_]

        X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, Y_, test_size = 0.3, random_state = 42)
        
        name_ = (df_root_.split('/'))[-1].split('.')[0]       
        destino_ =origin_root_+'SplitData/'
        if not os.path.exists(destino_):
            os.makedirs(destino_)

        X_train_.to_csv(destino_ + name_ + '_Xtrain.csv', index=False)
        X_test_.to_csv(destino_ + name_ + '_Xtest.csv', index=False)
        y_train_.to_csv(destino_ + name_ + '_ytrain.csv', index=False)
        y_test_.to_csv(destino_ + name_ + '_ytest.csv', index=False)

        datasets_[name_ + '_Xtrain'] = X_train_
        datasets_[name_ + '_Xtest'] = X_test_
        datasets_[name_ + '_ytrain'] = y_train_
        datasets_[name_ + '_ytest'] = y_test_
    
    return datasets_
    
                      
def normalize(origin_root: str, predictors: List[str], scaler: str = 'StandardScaler') -> Dict[str, pd.DataFrame]:
    """
    Normalizes datasets and saves them as CSV in
    >>> origin_root_\\NormData\\.

    This function normalizes datasets using a specified `scaler` from sklearn.preprocessing
    and returns a `dictionary` containing the normalized `datasets`.

    Parameters::

        origin_root (str): Path of the folder containing the CSV files of the datasets to normalize.
        predictors (List[str]): List of predictor variable names (excluding the target variable) in the datasets.
        scaler (str, optional): Name of the scaler from sklearn.preprocessing to use. Default is 'StandardScaler'.

    Returns::

        Dict[str, pd.DataFrame]: Dictionary containing the normalized datasets.
            The `keys` are the names of the normalized datasets ("dataset_name_norm").
            The `values` are the normalized datasets themselves.
    """
    origin_root_ = origin_root
    predictors_ = predictors.copy()
    scaler_ = scaler
    
    module_ = importlib.import_module('sklearn.preprocessing')
    submodule_ = getattr(module_, scaler_)
    scaler_ = submodule_()
    csv_files_ = []
    normalized_datasets_ = {}

    for file in os.listdir(origin_root_):
        if file.endswith(".csv"): csv_files_.append(os.path.join(origin_root_, file))
    
    for df_root_ in csv_files_:
        if '_X' in str(df_root_):
            df_ = pd.read_csv(df_root_)

            scaler_.fit(df_)
            scaler_df_ = scaler_.transform(df_)
            scaler_df_ = pd.DataFrame(scaler_df_, index = df_.index_, columns = predictors_)
            
            name_ = (df_root_.split('/'))[-1].split('.')[0]       
            destino_ =origin_root_+'NormData/'
            if not os.path.exists(destino_): os.makedirs(destino_)

            scaler_df_.to_csv(destino_ + name_ + '_norm.csv', index=False)
            normalized_datasets_[name_ + '_norm'] = scaler_df_
    
    return normalized_datasets_


def feature_sel(X_train: pd.DataFrame, y_train: pd.Series, k: int, file_name: str, method: str = 'SelectKBest', test: str = 'mutual_info_classif') -> pd.DataFrame:
    """
    Performs feature selection and saves the selected dataset to a CSV file in
    >>> \\data\\processed\\SplitData\\FeatureSel\\

    This function performs feature selection using a specified `method` and a `feature selection test`,
    and saves the selected dataset to a CSV file.

    Parameters::

        X_train (pd.DataFrame): Training dataset (predictor variables).
        y_train (pd.Series): Training dataset (target variable).
        k (int): Number of features to select.
        file_name (str): Name of the resulting CSV file.
        method (str, optional): Feature selection method. Default is 'SelectKBest'.
        test (str, optional): Feature selection test. Default is 'mutual_info_classif'.

    Returns::

        pd.DataFrame: Selected dataset.
    """
    X_train_ = X_train.copy()
    y_train_ = y_train.copy()
    k_ = k
    file_name_ = file_name
    method_ = method
    test_ = test
    
    module_ = importlib.import_module('sklearn.feature_selection')
    method_ = getattr(module_, method_)
    test_ = getattr(module_, test_)
        
    selection_model_ = method_(test_, k = k_)
    selection_model_.fit(X_train_, y_train_)
    ix_ = selection_model_.get_support()
    X_train_sel_ = pd.DataFrame(selection_model_.transform(X_train_), columns = X_train_.columns.values[ix_])

    if not os.path.exists('../data/processed/SplitData/FeatureSel/'):
            os.makedirs('../data/processed/SplitData/FeatureSel/')
    X_train_sel_.to_csv('../data/processed/SplitData/FeatureSel/'+str(file_name_)+'_FeatureSel.csv', index=False)

    return X_train_sel_