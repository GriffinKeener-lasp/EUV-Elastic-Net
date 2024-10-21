# EUV-Elastic-Net
Contains packages and notebooks for the statistical analysis of an Elastic Net model of the solar atmosphere.

# Jupyter Notebook

The Jupyter notebook contains the code necessary to run the package scripts. It also contains some plotting code to visualize the results

# Data Files
1. Arrays_compressed.npz - Large file containing the EUV spectral data, and the DEM spectrum from the EVE, EXIS, and AIA instruments. This data is used for the Elastic Net Regularization to obtain linear equations/coefficients for each wavelength of the spectrum.

2. Ion_line_list.xlsx - The Chianti line list in Excel spreadsheet format. Read in as a pandas dataframe. Used for determining the locations of the emitting ions in the solar atmosphere.

3. wavelength_array - An array of the wavelengths measured by EVE. This is used to assign named Chianti ions to specific EVE wavelengths.


# Classes in the Module 

## NetRegressor 
Class to run Elastic Net Regularizations for time-series spectral measurements from EVE. Uses the ChiantiPrep method contained in this package to "match up" the regularization results to the Chianti atomic database. 
First, initialize the class with the required inputs. Then run NetRegressor.prep_data() function. 
Required Inputs: 
  - file_path: the file path to the large array dataset, called /.arrays_compressed.npz
  - num_nets: The number of regressions to run. This decides how many time chunks are divided from the array. Default is 100. 
        

(The following classes are used within the NetRegressor class, but can be used independently)

## ChiantiPrep 
This class contains methods to align the EVE Elastic net regression (or L1 or L2 regularizations) with the chianti line list. The chianti line list is converted into a pandas dataframe. It then uses this to compute the ratios of contribution for each ion. For example, say HeII has 10 contributing ions, 6 from the Transition region, 4 from the Corona. The function region_ratios() would add 0.6 to the n_t column of the dataframe in the HeII row, and assigns 0.4 to the n_c column.
    
  ### Inputs
    ___________________________________________
    
    net_predict_eve: [numpy.ndarray] The regression-produced spectra, it is the array produced by Regressor.predict()
    net_coefs:       [numpy.ndarray] The coefficient matrix, obtained by calling Regressor.coef_
    linelist_file: The file path to the excel-formatted chianti line list


### Instructions
    

The functions in this class are meant to be run all at once. After inputting the desired parameters, call ChaintiPrep.instructions() to print out the code. Copy and paste the code into a cell, and you're good to go. 
    
## RegressionCoefs 
This method obtains the array of wavelengths produced by the Elastic Net Regression

If you have already run a regression, this can be used to obtain the coefficient matrix, then create a plot to visualize the contributions to each wavelength, from other wavelengths in the spectrum. If not, ignore this as it is just a tool used in later classes in this module. 


### Inputs: 
       
  target_wvl: the wavelength for which you want to see the contributions/get the coefficients for

  coefs: the coefficient matrix from the regression. Usually something like ElasticNet.coef_ or Lasso.coef_

  spectrum: A spectrum produced by the regression. Usually takes the form: regressor.predict()

  wvl_file: the file path to the wavelength array on your computer

### Functions: 
       
get_coefs: Obtains the coefficient matrix, and creates an array of which EVE lines have nonzero coefficients for a line of interest. For example, HeII is the 245 index of the 1012 line EVE spectrum. So calling the get_coefs function where target_wvl = 245 will obtain the index of the lines that have nonzero coefficients, i.e contribute to HeII.

plot_dots: Plotting code that will output an interactive plot that shows the contributing lines to target_wvl with red dots.
       
