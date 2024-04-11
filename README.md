<!-- Replace edastatmil with your library's name on PyPI -->
[![PyPI version](https://badge.fury.io/py/edastatmil-milser.svg)](https://badge.fury.io/py/edastatmil-milser)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/edastatmil-milser.svg)](https://pypi.org/project/edastatmil-milser/)

This library is [`edastatmil-milser`](https://github.com/milser/Python-library-template/search?q=edas_tatianamilser&unscoped_q=edas_tatianamilser) 

## General Description  
This library provides useful tools to conduct a comprehensive EDA.

> **⚠️ Note:** This module is in development phase and may contain errors.

> **⚠️ Note:** Some functions only apply to very general cases. To perform a good EDA, it is necessary to understand the specific case being worked on, and often actions not covered in this module will be needed.

> **⚠️ Note:** Please read the function descriptions carefully, as some require a specific file system. The final file structure if EDA is done with edastatmil-milser is shown in the following figure:

![files](./images/final.png)

## Installation
1. **Requirements:**  
   - tabulate  
   - pandas  
   - matplotlib.pyplot  
   - seaborn  
   - math  
   - os  
   - sklearn.model_selection  
   - importlib  
   These can be installed from the terminal with `pip install`.

2. **Install the library:**  
   ```bash
   pip install edastatmil-milser

3. **Import the library:**  
   ```bash
   from edastatmil_milser import edas_tatmil as EDA

4. **Example function call**
   ```bash
   EDA.function_example

## Functions
### `build_directory(data_root)`

This function creates the recommended folder system for conducting a comprehensive EDA with this library.

- **Attributes:** 
   - data_root: the path to the main data file, with '/' at the end.

- **Usage Example:**
  ```python
   EDA.build_directory('../data/')

- **Return:**
   If they do not exist, it adds the 'processed' folder in the 'data' folder. Inside 'processed', it creates the folders 'factorized_mapping', 'NonSplit', and 'SplitData'. Inside 'SplitData', it creates the folders 'FeatureSel' and 'NormData'.

### `get_column_type(series)`

This function studies if a feature is numeric or categorical.

- **Attributes:** not required.

- **Usage Example:**
  ```python
  variables = pd.DataFrame({'Data Type': data_frame.dtypes})
  variables['Data category'] = df.apply(EDA.get_column_type)

Adds a new column to the 'variables' dataframe called 'Data type' indicating if the variable is categorical or numeric.
- **Return:**
   'Categorical' if the variable is categorical 'Numerical' if it is numeric.

### `explore(data_frame)`

This function gives a general idea of the content of the dataframe.

- **Attributes:** 
   - data_frame: the name of the dataframe to explore.

- **Usage Example:**
  ```python
   categorical_list, numerical_list = EDA.explore(df_example)

Shows the number of columns and rows and a table with information about non-null, null values, data type, and category of the variables.

- **Return:**
   List of categorical variables and list of numerical variables.

### `FindDuplicates(data_frame, id_col, Drop=False)`

This function looks for duplicates in a column and gives the option to remove duplicate rows or not.

- **Attributes:** 
   - data_frame: the name of the dataframe to explore.
   - id_col: name of the column in which duplicates are to be searched.
   - Drop: if True, it will remove duplicate lines; if False, it will leave them. By default, it is False.

- **Usage Example:**
  ```python
   df_without_duplicates = EDA.FindDuplicates(df_example, 'id_host', Drop=True)
Returns the dataframe without duplicate rows according to the 'id_host' column.

- **Return:**
   If Drop=True, dataframe without duplicates. If Drop=False, dataframe with duplicates. In both cases, it prints the number of duplicates found.

### `Find_over_50_percent_value(df)`

This function looks for elements that occupy more than 50% of their column and shows information about them.

- **Attributes:** 
   - data_frame: the name of the dataframe to explore.

- **Usage Example:**
  ```python
   EDA.Find_over_50_percent_value(df_example)
Shows information about irrelevant values in the dataframe.

- **Return:**
   None.

### `univariate_hist(variables, data_frame, color='#1295a6', kde=False)`

This function creates a histogram for each variable in the 'variables' list. They will be displayed in a figure with three histograms per row.

- **Attributes:** 
   - data_frame: the name of the dataframe to explore.
   - variables: the list of variables for which histograms are to be created.
   - color: color. By default, it is turquoise.
   - kde: if True, the kernel density estimation line is shown on the graph; if False, it is not shown. By default, it is False.

- **Usage Example:**
  ```python
   list = ['age,'smoke','region','children']
   EDA.univariate\_hist(list,df_example)
Draws a figure with 4 histograms, three in the first row and one in the second, without kernel density estimation line and in turquoise.

- **Return:**
   Shows the figure.

### `univariate_histbox(variables, data_frame, color='#1295a6')`

This function creates a histogram and a box plot for each variable in the 'variables' list. They will be displayed in a figure with three histograms+box plots per row.

- **Attributes:** 
   - data_frame: the name of the dataframe to explore.
   - variables: the list of variables for which histograms and box plots are to be created.
   - color: color. By default, it is turquoise.

- **Usage Example:**
  ```python
   list = ['age,'smoke','charges','bmi']
   EDA.univariate\_histbox(list,df_example)

![histbox](./images/histbox.png)

- **Return:**
   Shows the figure.

### `multivariate_barplots(df, variable_lists,y='count',palette='Set2')`

This function creates a multivariable bar plot for each set of variables in the 'variable_lists'. They will be displayed in a figure with one graph per row.

- **Attributes:** 
   - df: the name of the dataframe to explore.
   - variable_lists: it is a list of lists such as ['variable on x-axis','variable on y-axis','discrimination variable']. The 'y' variable must be numeric.
   - y: what the height of the bars should represent. If 'count', the height of the bars represents the number of elements of the group with the 'y' variable. If 'mean', the height will represent the mean. By default, it is 'count'.
   - palette: Color palette. By default, it is the seaborn default 'Set2'.

- **Usage Example:**
  ```python
   variable_lists=[['age','charges','smoker'], ['sex','charges','children]]
   EDA.multivariate_barplots(df, variable_lists,y='mean')

![multivariate](./images/multivariate.png)
Draws a figure with 2 bar plots. In the first one, age is represented on the x-axis, there is one bar for each value of the smoker column, and the height of the bar gives the mean of the 'charges' variable for each group. In the second one, sex is represented on the x-axis, there is one bar for each value of the 'children' column, and the height of the bar represents the mean of 'charges' for each group.

- **Return:**
   Shows the figure.

### `factorize_categorical(df,cols_to_factor)`

This function factorizes the categorical variables included in the 'cols_to_factor' list and replaces the value in the column with the one assigned in the factorization.

- **Attributes:** 
   - df: the name of the dataframe to explore.
   - cols_to_factor: it is a list of categorical variables.

- **Usage Example:**
  ```python
   fz_df = factorize_categorical(df,variables_list)

- **Return:**
   Returns the dataframe with the indicated variables factorized.

### `correlation_matrix(df, variables_list, size)`

This function constructs and displays a correlation matrix heatmap.
> **⚠️ Note:** It is not necessary to factorize the categorical variables before calling this function. Just include them in the list.

- **Attributes:** 
   - df: the name of the dataframe to explore.
   - variables_list: list of the categorical variables from your dataframe that are not factorized. 
   - size: tuple with the width and height of the figure. By default (20,16).

- **Usage Example:**
  ```python
   df_factorized, df_factorized_onlynumerical  = EDA.correlation_matrix(raw_df, categorical_list,(10,7))
![heatmap](./images/heatmap.png)
- **Return:**
   Shows the figure and returns two dataframes. The first dataframe has the categorical variables factorized as 'variable' and the categorical equivalent as 'variable_0'. The second dataframe contains only the numerical variables and the factorized categorical ones.

### `numerical_box(variables, data_frame, color='#1295a6')`

This creates a box plot for each variable in the list and displays them in a figure with three plots per row.

- **Attributes:** 
   - data_frame: the name of the dataframe to explore.
   - variables: list of the variables for which a box plot is to be created. They must be numeric.
   - color: color. By default, it is turquoise. 

- **Usage Example:**
  ```python
   numerical = ['age','bmi','children','charges']
   EDA.numerical_box(numerical, raw_df)
![box](./images/box.png)
- **Return:**
   Shows the figure.

### `outliers_iqr(df,var,sigma,Do=Do_enum.NOTHING)`

This function looks for outliers in the indicated column, using the interquartile range (75%-25%) criterion. The upper and lower limits of the accepted value range are adjusted with a sigma parameter. There are different options for handling the outliers found.
> **⚠️ Note:** This function requires importing Do_enum.

> **⚠️ Note:** This function is not the only way to handle outliers and for specific cases not covered here, manual handling should be done.

> **⚠️ Note:** Keep in mind that this function will try to handle ALL outliers in the same column as indicated.

> **⚠️ Note:** This function is useful for counting and finding outliers by interquartile range and then, once found, being able to use other methods for their treatment.

- **Attributes:** 
   - df: the name of the dataframe to explore.
   - var: the variable on which outliers are being searched.
   - sigma: the parameter adjusting the accepted value interval.
   - Do: what to do with the outliers. If 'nothing', it only counts them but does nothing. If 'mode', 'median', or 'mean', it replaces them with the mode, median, or mean, respectively. If 'drop', it removes the rows with outliers from the dataframe. By default, it is 'nothing'.

- **Usage Example:**
  ```python
   from edastatmil_milser.edas_tatmil import Do_enum
   outliers, cleaned_df = EDA.outliers_iqr(raw_df,'bmi',1,Do=Do_enum.DROP)
In this case, cleaned\_df will not have the outlier rows because 'drop' option is chosen.
- **Return:**
  Returns a dataframe with the outliers and another dataframe with the outliers treated according to the chosen procedure. In any case, it prints the number of outliers found.

### `splitter(origin_root,predictors,target)`

This function divides all the dataframes found in the indicated path into train and test.
> **⚠️ Note:** Make sure that only the dataframes to be divided are in the indicated path. The rest should be stored in another folder.

> **⚠️ Note:** Make sure to name the initial files correctly so that you can later differentiate all the resulting files from the division.

- **Attributes:** 
   - origin_root: the path where the dataframes to be divided are located, including / at the end.
   - predictors: list of predictor variables.
   - target: name of the target variable. 

- **Usage Example:**
  ```python
   predictors = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
   target = 'charges'
   EDA.splitter('../data/processed/',predictors,target)
All the dataframes saved in the directory '../data/processed/' will be split into train and test.
- **Return:**
  Creates a folder in the indicated path called SplitData. Inside are all the resulting dataframes from the division of each file. The original names of the files will have the suffix _Xtrain, _Xtest, _ytrain, _ytest appended to differentiate them.

### `normalize(origin_root,predictors,scaler='StandardScaler')`

This function normalizes the data of the predictor variables.
> **⚠️ Note:** Remember that the target is not normalized.

> **⚠️ Note:** Before using this function, categorical variables must be factorized.

> **⚠️ Note:** Make sure that the files to be normalized include the suffix _X

- **Attributes:** 
   - origin_root: the path where the dataframes to be normalized are located, including / at the end.
   - predictors: list of predictor variables.
   - scaler: name of the scaling function to be used. It can be any from the sklearn.preprocessing library. By default, it is StandardScaler.

- **Usage Example:**
  ```python
   predictors = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
   EDA.normalize('../data/processed/SplitData/',predictors,scaler='StandardScaler')
The predictor variables of all dataframes found in the directory '../data/processed/SplitData/' and containing the suffixes _Xtrain,_Xtest in their file names will be normalized.
- **Return:**
  Creates a folder in the indicated path called NormData. Inside are all the resulting dataframes from the normalization of each dataframe. The original names of the files will have the suffix _norm appended.

### `feature_sel(X_train,y_train,k,file_name,method='SelectKBest', test='mutual_info_classif')`

This function performs feature selection on the training datasets, leaving a number k of variables and using the indicated method and test.
> **⚠️ Note:** Remember that feature selection is not performed on the testing data.

- **Attributes:** 
   - X_train: the dataframe containing the training data of the predictor variables.
   - y_train: the dataframe containing the training data of the target.
   - k: the number of variables to leave.
   - file_name: the name under which the resulting dataframe from feature selection is to be saved.
   - method: the method to be used for selection.
   - test: the test on which the selection is based.

- **Usage Example:**
  ```python
   All_X_train = pd.read_csv('../data/processed/SplitData/NormData/ All_factorize_Xtrain_norm.csv')
   All_y_train = pd.read_csv('../data/processed/SplitData/All_factorize_ytrain.csv')    
   EDA.feature_sel(All_X_train,All_y_train,k=4,file_name='All_Xtrain', method='SelectKBest', test='mutual_info_regression')
Normalized dataframes can be loaded from wherever they were saved.
A training dataframe will be created that contains only the most relevant columns according to the indicated method and test.  
- **Return:**
  Creates a folder in the SplitData folder called FeatureSel and the dataset with only the selected columns will be saved with the indicated name and the _FeatureSol suffix. 

## Version

Version 1.0 (12/04/2024)

When you want to increment the version number for a new release use [`bumpversion`](https://github.com/peritus/bumpversion) to do it correctly across the whole library.
For example, to increment to a new patch release you would simply run

```
bumpversion patch
```

which given the [`.bumpversion.cfg`](https://github.com/milser/Python-library-template/blob/master/.bumpversion.cfg) makes a new commit that increments the release version by one patch release.
