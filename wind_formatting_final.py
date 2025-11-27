# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#IMPORTING RELEVANT LIABRARIES:
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime, timedelta

#UPLOADING NC FILE:
f = netCDF4.Dataset(r'C:\Users\Besitzer\Desktop\downloaded wind\mesotimeseries-Point 3.nc')

#UPLOADING THE STANDARD TABLE FROM EXCEL
#r converts normal string to raw string 
components_discription=pd.read_excel(r'C:\Users\Besitzer\Desktop\master thesis\meeting wind and python\standard table.xlsx' , index_col=0)
components_discription.reset_index(inplace=True)
#not needed but can be useful
#unique_heights=components_discription["Height"].unique()

#FORMATTING DATA:

# CDF4 Arrays
WS = f.variables['WS']
WS10 = f.variables['WS10']
crs = f.variables['crs']
time = f.variables['time']
XLAT = f.variables['XLAT']
XLON = f.variables['XLON']
South_North = f.variables['south_north']
West_East = f.variables['west_east']
height = f.variables['height']

# Python Arrays
WS = WS[:]
WS10 = WS10[:]
time = time[:]
XLAT = XLAT[:]
XLON = XLON[:]
South_North = South_North[:]
West_East = West_East[:]
height = height[:]
time = np.round(time / 30) * 30
# DateTime jeder Spalte berechnen
Start_Time = datetime.strptime('1989.01.01 00:00:00', "%Y.%d.%m %H:%M:%S")
DateTime = np.array([Start_Time + timedelta(minutes=float(t)) for t in time])
secs_passed=np.arange(0,len(WS)*1800,1800)


# Stacking numeric arrays then form them as numeric data frame seperately because of numerical missing values 
numeric_data = np.column_stack([WS10, WS])
numeric_data = pd.DataFrame(numeric_data, columns= [10, height[0], height[1], height[2],height[3]])

#FILLING MISSING VALUES IN NUMERIC DATA WITH SUITABLE METHOD: 

#numeric_data.fillna(numeric_data.mean(numeric_only=True).round(1), inplace=True)
#Interpolate backwardly across the column:
numeric_data.interpolate(method ='linear', limit_direction ='forward', inplace=True)

#Stacking and transforming other data into other dataframe later concatenating 
other_data = np.column_stack([secs_passed,time, DateTime])
other_data = pd.DataFrame(other_data, columns= ['secs_passed','Min','DateTime',])

df_uploaded_data = pd.concat([other_data, numeric_data], axis=1) 
#To insert the mean value of each column into its missing rows:


#to check if missing vales are tackeld with or not
#print(df_uploaded_data.isnull().sum().sum() )

#selecting columns
f150m = df_uploaded_data.loc[:,150.0]
f100m = df_uploaded_data.loc[:,100.0]
f75m = df_uploaded_data.loc[:,75.0 ]
f10m = df_uploaded_data.loc[:,10.0]
columns=[f10m,f75m,f100m,f150m]
#DYNAMIC INTERPOLATION BASED ON COMPONENTS IN STANDARD TABLE
for i in range(len(components_discription["Height"])):
    current_height = components_discription["Height"][i]
    array = [10, 75, 100, 150]
    index = min(range(len(array)), key=lambda i: abs(array[i] - current_height))
    
    df_uploaded_data[str(components_discription["Description"][i]) + str(components_discription["Height"][i]) + " m"] = columns[index]

#DELETING UNNESSCARY DATA           
df_uploaded_data.pop(10) 
df_uploaded_data.pop(100) 
df_uploaded_data.pop(150) 
df_uploaded_data.pop(75)  
df_uploaded_data.pop('Min')
#df_uploaded_data.pop('Bottom17 m')
#df_uploaded_data.pop('MID140 m')
#df_uploaded_data.pop('MID270 m')
#df_uploaded_data.pop('MID3100 m')
#df_uploaded_data.pop('TOP133 m')
#df_uploaded_data.pop('Nacelle135 m')
#df_uploaded_data.pop('Drive Train135 m')
#df_uploaded_data.pop('Hub135 m')
#df_uploaded_data.pop('Blades135 m')





# Fit the Weibull distribution to the histogram data
hist_data, bin_edges, _ = plt.hist(df_uploaded_data['Hub135 m'], bins='auto', density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Fit the Weibull distribution to the histogram data
shape, loc, scale = stats.weibull_min.fit(df_uploaded_data['Hub135 m'], floc=0)

# Calculate the Weibull PDF using the fitted parameters
weibull_pdf = stats.weibull_min.pdf(bin_centers, shape, loc, scale)

# Calculate the standard deviation using the Weibull parameters
weibull_std = scale * np.sqrt(stats.weibull_min.var(shape, loc, scale))

# Plot the histogram and the fitted Weibull curve
plt.bar(bin_centers, hist_data, width=bin_edges[1]-bin_edges[0], label='Histogram')
plt.plot(bin_centers, weibull_pdf, 'r-', label='Weibull Fit')

plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram with Weibull Fit')
plt.legend()
plt.show()

# Print the Weibull parameters and standard deviation
print("Shape parameter (k):", shape)
#print("Location parameter (λ):", loc)
print("Scale parameter (β):", scale)
#print("Standard Deviation (σ):", weibull_std)

#Statistics data:
print("mean at hub height is:  ",df_uploaded_data['Hub135 m'].mean())
print("standard_deviation at hub height is:  ",df_uploaded_data['Hub135 m'].std())
print("max speed at hub height is:  ",df_uploaded_data['Hub135 m'].max())

df_uploaded_data['Month'] = df_uploaded_data['DateTime'].dt.month
monthly_means = df_uploaded_data.groupby('Month')['Hub135 m'].mean()

plt.bar(monthly_means.index, monthly_means)
plt.xlabel('Month')
plt.ylabel('Mean Wind Speed')
plt.title('Monthly Variations in Mean Wind Speed')
plt.xticks(range(1, 13))
plt.show()


# Calculate the hour component of the DateTime column
df_uploaded_data['Hour'] = df_uploaded_data['DateTime'].dt.hour

# Calculate the hourly means of the wind speed data
hourly_means = df_uploaded_data.groupby('Hour')['Hub135 m'].mean()

# Create a line plot to visualize the hourly variations
plt.plot(hourly_means.index, hourly_means)
plt.xlabel('Hour')
plt.ylabel('Mean Wind Speed')
plt.title('Hourly Variation of Mean Wind Speed')
plt.xticks(range(24))
plt.show()


# Define a function to map months to seasons
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'

# Extract the month from the DateTime column
df_uploaded_data['Month'] = df_uploaded_data['DateTime'].dt.month

# Map the months to seasons using the custom function
df_uploaded_data['Season'] = df_uploaded_data['Month'].apply(get_season)

# Calculate the seasonal means
seasonal_means = df_uploaded_data.groupby('Season')['Hub135 m'].mean()

# Define the season labels
season_labels = ['Spring', 'Summer', 'Autumn', 'Winter']

# Create the bar graph
plt.bar(season_labels, seasonal_means)

# Set the labels and title
plt.xlabel('Season')
plt.ylabel('Mean Wind Speed')
plt.title('Mean Wind Speed by Season')

# Show the plot
plt.show()







# WRITING IN EXCEL FILE:        



#df_uploaded_data.to_excel("10e.xlsx",engine='xlsxwriter')

#df_uploaded_data.to_csv("wind_csv.csv")


