'''This module contains the essential tools to recreate the work done with the Elastic Net Model of the EUV spectrum. 

---------------------------
Developed by Griffin Keener and Shah Bahauddin at the Laboratory for Atmospheric and Space Physics

'''

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go 
import warnings
import seaborn as sns
import pickle



class RegressionCoefs:
    '''This method obtains the array of wavelengths produced by the Elastic Net Regression

       If you have already run a regression, this can be used to obtain the coefficient matrix, then create a plot to visualize the contributions to each wavelength, from other wavelengths in the spectrum. If not, ignore this as it is just a tool used in later classes in this module. 


       Inputs: 
       -----------------------------
       target_wvl: the wavelength for which you want to see the contributions/get the coefficients for
       coefs: the coefficient matrix from the regression. Usually something like ElasticNet.coef_ or Lasso.coef_
       spectrum: A spectrum produced by the regression. Usually takes the form: regressor.predict()
       wvl_file: the file path to the wavelength array on your computer

       Functions: 
       -----------------------------
       get_coefs: Obtains the coefficient matrix, and creates an array of which EVE lines have nonzero coefficients for a line of interest. For example, HeII is the 245 index of the 1012 line EVE spectrum. So calling the get_coefs function where target_wvl = 245 will obtain the index of the lines that have nonzero coefficients, i.e contribute to HeII.

       plot_dots: Plotting code that will output an interactive plot that shows the contributing lines to target_wvl with red dots.
       
    
    
    '''
    def __init__(self, wvl_file, target_wvl, coefs, spectrum):
        self.target_wvl = target_wvl
        self.coefs = coefs
        self.spectrum = spectrum
        self.wvls = None
        self.nonzero_coefs = None
        # Check if the wavelength file path exists
        if not os.path.exists(wvl_file):
            warnings.warn(f"Warning: The file '{wvl_file}' does not exist. Please check the file path.")
        else:
            self.wvl_arr = np.load(wvl_file) * 10  # Convert to angstrom
        
        
    def get_coefs(self):
        '''
        Extracts the wavelengths of nonzero coefficients for a specified target wavelength.

        Target wvl: 

        Outputs:
        - wvls: Array of wavelengths with nonzero coefficients
        - nonzero_coefs: Array of nonzero coefficients contributing to the target peak
        '''
        # get the coefficients of the target wvl 
        target_coefs = self.coefs[self.target_wvl]
        # find the nonzero ones 
        self.nonzero_coefs = target_coefs[target_coefs != 0]
        # Get the index of the nonzero coefs 
        self.wvls = np.where(target_coefs != 0)[0]
        # return array of indices 
        return self.wvls
        # This is an array of the EVE index of the nonzero coefs

    def plot_dots(self):
        '''
        Plots the spectrum and highlights the lines with non-zero contributions.
        Displays the wavelength and associated coefficient in scientific notation on hover.
        '''
        if self.wvls is None or self.nonzero_coefs is None:
            raise ValueError("Call get_coefs() before plotting.")

        # Mask wavelengths to include only DEM
        dem_wvls = self.wvls[self.wvls <= 1012]
        dem_coefs = self.nonzero_coefs[self.wvls <= 1012]
        wvls_actual = self.wvl_arr 
        
        #wvls_actual = np.linspace(5.42, 108.42, self.spectrum.shape[0]) * 10 

        # Get actual wavelengths corresponding to non-zero coefficients
        dem_actual_wvls = wvls_actual[dem_wvls]

        # Create the figure
        fig = go.Figure()
        fig.update_layout(
            autosize=True
        )

        # Plot the spectrum as a line
        fig.add_trace(go.Scatter(
            x=wvls_actual,  # Use the full wavelength range for the spectrum
            y=self.spectrum,
            mode='lines',
            name='Spectrum',
            hoverinfo='x+y'
        ))

        # Add the dots for DEM wavelengths using actual wavelengths
        fig.add_trace(go.Scatter(
            x=dem_actual_wvls,  # Use actual wavelengths corresponding to non-zero coefficients
            y=self.spectrum[dem_wvls],
            mode='markers',
            marker=dict(color='red', size=10),
            name=f'Contributions for λ = {wvls_actual[self.target_wvl]:.2f} Å',
            text=[f'Wavelength: {wv:.2f} nm, Coefficient: {coef:.2e}' for wv, coef in zip(dem_actual_wvls, dem_coefs)],
            hoverinfo='text+y'
        ))

            
        # Update layout for better visuals
        fig.update_layout(
            title=f'<b>Spectrum Contributions for λ = {wvls_actual[self.target_wvl]:.2f} Å</b><br>'
                  f'<span style="font-size:14px">Number of Nonzero Coefficients: {self.nonzero_coefs.shape[0]}</span>',
            xaxis_title='Wavelength (Å)',
            yaxis_title='Intensity',
            yaxis_type='log',
            hovermode='closest'
        )


        # Show the plot
        fig.show()



class ChiantiPrep: 
    
    '''This class contains methods to align the EVE Elastic net regression (or L1 or L2 regularizations) with the chianti line list. The chianti line list is converted into a pandas dataframe. It then uses this to compute the ratios of contribution for each ion. For example, say HeII has 10 contributing ions, 6 from the Transition region, 4 from the Corona. The function region_ratios() would add 0.6 to the n_t column of the dataframe in the HeII row, and assigns 0.4 to the n_c column.
    
    Inputs
    ___________________________________________
    
    net_predict_eve: [numpy.ndarray] The regression-produced spectra, it is the array produced by Regressor.predict()
    net_coefs:       [numpy.ndarray] The coefficient matrix, obtained by calling Regressor.coef_
    linelist_file: The file path to the excel-formatted chianti line list


    Instructions
    ___________________________________________

    The functions in this class are meant to be run all at once. After inputting the desired parameters, call ChaintiPrep.instructions() to print out the code. Copy and paste the code into a cell, and you're good to go. 
    
    '''

    

    
    def __init__(self, net_predict_eve, net_coefs, linelist_file, wvl_file): 
        
        # make sure file paths are entered correctly 
        if not os.path.exists(linelist_file):
            warnings.warn(f"Warning: The file '{linelist_file}' does not exist. Please check the file path.")
        else:
            self.chianti_df = pd.read_excel(linelist_file)

        if not os.path.exists(wvl_file):
            warnings.warn(f"Warning: The file '{wvl_file}' does not exist. Please check the file path.")
        else:
            self.wvl_array = np.load(wvl_file)
        
        self.linelist_file = linelist_file
        self.net_predict_eve = net_predict_eve
        self.net_coefs =  net_coefs
        self.index_df = None 
        self.wvl_file = wvl_file
        self.wvl_array = np.load(self.wvl_file)
        self.chianti_df = pd.read_excel(linelist_file)
        
        
    def instructions(self): 
        '''Prints code for you to copy and paste to do all of the chianti prep at once '''
        print('chianti_prep.wvl_arr() \nchianti_prep.align_chianti() \nchianti_prep.nonzero_coefs(measurement=0) \nchianti_prep.region_classifier() \nchianti_prep.region_ratio')

        print(self.chianti_df.columns)

    
    def reset_for_measurement(self):
        # Optionally, re-read the excel file to reset chianti_df for each new measurement
        self.chianti_df = pd.read_excel(self.linelist_file)
        
    
    def wvl_arr(self): 
        '''Creates the wavelength - index dataframe and stores it as self '''
        
        
    
        x = np.linspace(0, 1012, 1012).astype(int)
        y = self.wvl_array.round(2) * 10
        
        # Create a DataFrame
        self.index_df = pd.DataFrame({
            'index': x,
            'wavelength': y
        })
        return self.index_df
    
    
    def align_chianti(self): 
        '''Function to align the eve data with chianti'''
        
        # Drop the Unnecessary Columns for Readability 
        # Check if the columns exist before attempting to drop them
        columns_to_drop = ['Column4', 'Column5', 'Column6', 'Column7', 'Column8', 'Transition']
        self.chianti_df = self.chianti_df.drop(columns=[col for col in columns_to_drop if col in self.chianti_df.columns])

        
        
        # Ensure index_df is already created by calling wvl_arr first
        if self.index_df is None:
            raise ValueError("index_df is not initialized. Run wvl_arr() first.")
        
        # Function to align the data 
        chianti_wavelengths = self.chianti_df['λ (A) ˚']
        index_wavelengths = self.index_df['wavelength']
        
        # Function to find the index of the closest wavelength in index_df
        def find_eve_index(wavelength, index_wavelengths):
            eve_index = (index_wavelengths - wavelength).abs().idxmin()
            return eve_index
        
        # Apply the function to each wavelength in chianti_df
        self.chianti_df['eve_index'] = chianti_wavelengths.apply(find_eve_index, index_wavelengths=index_wavelengths)

        self.chianti_df = self.chianti_df



    def nonzero_coefs(self, measurement=0): 

        '''Function to get the indices of the contributions into the dataframe

        Measurement is the spectrum number, defaulted at 0 for testing, but left as variable for a loop. 
        
        '''

        
        # for every eve line, get the nonzero coefficients' indices 
        nonzero_wvls = []
        
        
        # Loop through each EVE line
        for i in range(len(self.net_coefs[:, 0])): 
            
            analyzer = RegressionCoefs(self.wvl_file, target_wvl = i, coefs = self.net_coefs, spectrum = self.net_predict_eve[measurement])
            
            # Get the indices of the nonzero coefficients
            getcoefs_wvls = analyzer.get_coefs()
            
            # Append the result to the list
            nonzero_wvls.append(getcoefs_wvls)
        
        # Now nonzero_wvls is a LIST of arrays, 1 array for each of the 1012 eve lines that contains the 
        # index of the contributing ion 
        
        # Get rid of EXIS contributions 
        nonzero_wvls = [np.array([val for val in arr if val <= 1012]) for arr in nonzero_wvls]
        
        
        # lasso_coefs = np.load('/users/griffinkeener/LASP/DataFiles/lasso_coefs.npy')
        # nonzero_wvls = np.array([])
        
        self.chianti_df['nonzero_wvls'] = pd.Series(nonzero_wvls[:549])

        self.chianti_df = self.chianti_df




    def region_classifier(self): 
        def regionclassifier(temps): 
            '''Takes the temperature column from the chianti dataframe, and adds a column with the solar region in which the ion exists'''
            regions = []
        
            # Temperature ranges of the 3 solar regions [K]
            chromo_threshold = 4.55 # max temp of solar corona 
            trans_threshold = 6  # max temp of transition region 
            
            for temp in temps: 
                if temp < chromo_threshold: 
                    regions.append('Chromosphere')
        
                elif chromo_threshold <= temp < trans_threshold: 
                    regions.append('Transition')
        
                else: 
                    regions.append('Corona')
            return regions 
    
        
        region_list = regionclassifier(self.chianti_df['T_{max}'].values)

        self.chianti_df['Solar Region'] = region_list

        self.chianti_df = self.chianti_df




    def region_ratio(self):
        '''Finds the ratio of contribution from the TR and Corona, places the values in the n_t and n_c columns respectively.'''
        if self.chianti_df is None:
            raise ValueError("chianti_df is not initialized. Run process_chianti_df() first.")

        def count_corona_wavelengths(row):
            nonzero_wvls = row['nonzero_wvls']
            
            # Convert to a set to remove duplicates
            nonzero_wvls = set(nonzero_wvls)
            
            # Find rows in chianti_df that match the nonzero wavelengths and drop duplicates based on 'eve_index'
            corona_matches = self.chianti_df.loc[self.chianti_df['eve_index'].isin(nonzero_wvls)].drop_duplicates(subset='eve_index')
            
            # Count how many of these wavelengths correspond to 'Corona'
            corona_count = sum(corona_matches['Solar Region'] == 'Corona')
            
            # Return the fraction of 'Corona' wavelengths
            return corona_count / len(nonzero_wvls) if nonzero_wvls else 0

        # Apply the function to calculate 'n_c'
        self.chianti_df['n_c'] = self.chianti_df.apply(count_corona_wavelengths, axis=1)
        
        # Calculate 'n_t' as 1 - 'n_c', or 0 if 'nonzero_wvls' is empty
        self.chianti_df['n_t'] = self.chianti_df.apply(lambda row: 1 - row['n_c'] if len(row['nonzero_wvls']) > 0 else 0, axis=1)
        
        return self.chianti_df



###################################################################################################



class NetRegressor:
    '''Class to run Elastic Net Regularizations for time-series spectral measurements from EVE. Uses the ChiantiPrep method contained in this package to "match up" the regularization results to the Chianti atomic database. 
        -------------------------------------------------------------------------------------------
        
        
        First, initialize the class with the required inputs. Then run NetRegressor.prep_data() function. 


        Required Inputs: 

        - file_path: the file path to the large array dataset, called /.arrays_compressed.npz
       
        - num_nets: The number of regressions to run. This decides how many time chunks are divided from the array. Default is 100. 
        
        '''

    def __init__(self, file_path='./arrays_compressed.npz', num_nets = 100,):
        
       
        self.num_nets = num_nets
        self.file_path = file_path
        self.spec_eve = None
        self.date_eve = None
        self.spec_dem = None
        self.spec_exis = None
        self.aia_data = None
        self.emT_data = None
        self.phyinf = None
        self.phyinf_split = None
        self.spec_eve_split = None
    
    def prep_data(self):
        '''Loads data from the compressed .npz file'''
        loaded = np.load(self.file_path)
        self.spec_eve  = loaded['array1']
        self.date_eve  = loaded['array2']
        self.spec_dem  = loaded['array3']
        self.spec_exis = loaded['array4']
        self.aia_data  = loaded['array5']
        self.emT_data  = loaded['array6']
    
        
        self.phyinf = np.zeros((self.spec_dem.shape[0], self.spec_dem.shape[1] + self.spec_exis.shape[1]))
        for i in range(self.spec_dem.shape[0]):
            self.phyinf[i, 0:self.spec_dem.shape[1]] = self.spec_dem[i, :]
            self.phyinf[i, self.spec_dem.shape[1]:] = self.spec_exis[i, :]
    
    
        
        self.phyinf_split = np.array_split(self.phyinf, self.num_nets, axis=0)
        self.spec_eve_split = np.array_split(self.spec_eve, self.num_nets, axis=0)
        
    
        print('Data ready for regression. Call run_regression_pipeline()')
           
    
    def run_regression_pipeline(self, ion_list_file, wavelength_file, output_dir = None, t_ranges=None):
        '''
        Runs the entire regression pipeline with ElasticNet, ChiantiPrep, and classification.
        
        Arguments:
        
        - ion_list_file: Path to the Chianti ion line list Excel file.
        
        - wavelength_file: Path to the file containing the wavelength array.
       
        - output_dir: Directory where the output CSV files will be saved, if no argument is passed, the dataframes will not be saved. 
                    This is to make it such that you don't need run the regressions over again if you need to use the dataframes for 
                    something. 

        - t_ranges: Specify the temperature zones that you want to obtain ratios for
        
        
        '''
        # Set temperature ranges
        if t_ranges == None: 
            t_ranges = np.array([[4, 5], [5, 5.5], [5.5, 6], [6, 6.5], [6.5, 7], [7, 8]])
    
        else: t_ranges = ranges
        
        fractions_all = np.array([])
    
        for i in range(len(self.phyinf_split)):
            # Train Test split 
            phyinf_train, phyinf_test, spec_eve_train, spec_eve_test = train_test_split(self.phyinf_split[i], self.spec_eve_split[i], test_size=0.25, random_state=42, shuffle=True)
            
            # Standardize the training data
            scaler = StandardScaler()
            phyinf_train_scaled = scaler.fit_transform(phyinf_train)
            phyinf_test_scaled = scaler.transform(phyinf_test)
    
            print(f'Mean: {phyinf_train_scaled.mean():.2e} \n Variance: {phyinf_train_scaled.var():.2f}')
            
            # Center y data 
            eve_train_centered = spec_eve_train - spec_eve_train.mean()
            
            
            
            # Perform ElasticNet regression
            net = ElasticNet(fit_intercept=False, max_iter=100, random_state=42, alpha=1e-6, l1_ratio=0.5, selection='random', warm_start=True)
            net.fit(phyinf_train_scaled, eve_train_centered)
            
            # Predict and test accuracy 
            net_predict_eve = net.predict(phyinf_test_scaled)
            net_predict_eve += spec_eve_train.mean(axis=0)  # Add mean back in to unstandardize intensity
            
           
            
            #Initialize and run ChiantiPrep methods
            chianti_prep = ChiantiPrep(net_predict_eve, net.coef_, ion_list_file, wavelength_file)
            chianti_prep.wvl_arr()
            chianti_prep.align_chianti()
            chianti_prep.nonzero_coefs()
            chianti_prep.region_classifier()
            chianti_prep.region_ratio()
            
            chianti_df = chianti_prep.chianti_df
            
            
            
            # Save the dataframes if output_dir is provided
            if output_dir is not None:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                file_name = os.path.join(output_dir, f'chianti_df_{i}.csv')
                chianti_df.to_csv(file_name, index=False)


            
    
            # Loop through temperature ranges
            for j in range(len(t_ranges)):
                sun_region = chianti_df[(chianti_df['T_{max}'] > t_ranges[j][0]) & (chianti_df['T_{max}'] <= t_ranges[j][1])]
                
                # Find the coronal & TR contributions. Corona is n_c > 0.5 and TR is n_t >= 0.5
                n_c_greater_05 = sun_region[sun_region['n_c'] > 0.5]
                n_c_less_equal_05 = sun_region[sun_region['n_t'] > 0.5]
                
                # Get the number of values for each condition 
                counts = [len(n_c_less_equal_05), len(n_c_greater_05)]
                
                # Normalize the counts so we can see the percentage of contribution
                total_count = sum(counts)
                fractions = [count / total_count for count in counts]
                fractions_all = np.append(fractions_all, fractions)
    
            print(f'Regression {i+1} / {self.num_nets} Complete')
            chianti_prep.reset_for_measurement()
        
        return fractions_all
    




class Plot:
    def __init__(self, file_path):
        # Initialize with the file path and load the data
        self.file_path = file_path
        self.fractions_all = np.load(self.file_path)
        self.n_t = self.fractions_all[::2]

    def ratio_hist(self):
        # Create the histogram plot with KDE and mean/std lines
        sns.histplot(self.n_t, kde=True)
        plt.axvline(self.n_t.mean(), c='red', label='$\mu$')
        plt.axvline(self.n_t.mean() + self.n_t.std(), c='orange', label='1 $\sigma$')
        plt.axvline(self.n_t.mean() - self.n_t.std(), c='orange')
        plt.title('Contribution Fraction Distribution from the Transition Region to All Regions')
        plt.legend(loc='upper left')
        plt.figtext(0.2, -0.05, f'Mean Ratio: {self.n_t.mean():.2f} \nSTD: {self.n_t.std():.3f}')
        plt.show()

    
   
    
    def plot_contribution_ratios(self, time_file):
        '''
        Creates a plot that shows how n_t changes over time. That is, how much the Transition Region influences the specific t_ranges over time. 

        Inputs: 
            time_file: The file path to the file: gregorian_dates.pkl
        
        
        ----------------------------------------------------
        '''
        # Reshape the n_t array to (100, 6)
        n_t_reshaped = self.n_t.reshape(100, 6)
    
        # Load the Gregorian dates (time data) from the pickle file
        with open(time_file, 'rb') as file:
            eve_times = pickle.load(file)
    
        # Resample the times to have 100 time points
        resampled_times = np.linspace(0, len(eve_times) - 1, 100, dtype=int)
        selected_times = [eve_times[i] for i in resampled_times]
    
        # Create the subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    
        # Plot for Zones 1, 2, 3
        ax1.plot(selected_times, n_t_reshaped[:, 0], color='blue', label='T = 4 - 5', alpha=0.8)
        ax1.plot(selected_times, n_t_reshaped[:, 1], color='orange', label='T = 5 - 5.5', alpha=1, linestyle='dotted')
        ax1.plot(selected_times, n_t_reshaped[:, 2], color='red', label='T = 5.5 - 6', alpha=0.7, linestyle='dashed')
        ax1.set_ylim(0, 1)
        ax1.legend(title='Temp Range [log(T)]')
        ax1.set_title('Zones 1, 2, 3')
    
        # Plot for Zones 4, 5, 6
        ax2.plot(selected_times, n_t_reshaped[:, 3], color='gold', label='T = 6 - 6.5')
        ax2.plot(selected_times, n_t_reshaped[:, 4], color='orange', label='T = 6.5 - 7')
        ax2.plot(selected_times, n_t_reshaped[:, 5], color='red', label='T = 7 - 8')
        ax2.set_ylim(0, 1)
        ax2.legend(title='Temp Range [log(T)]')
        ax2.set_title('Zones 4, 5, 6')
    
        # Set overall plot titles and labels
        fig.suptitle('Ratios of Transition Region Contributions Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Ratio of Contribution From Transition Region [%]')
        ax2.set_xlabel('Time')
    
        # Rotate the x-axis labels
        ax1.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='x', rotation=45)
    
        # Adjust the layout and display the plot
        plt.tight_layout()
        plt.show()
    
    







