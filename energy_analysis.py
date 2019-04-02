import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display up to 60 columns of a dataframe
pd.set_option('display.max_columns', 60)

# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

sns.set(font_scale = 2)

########## Functions #####################

#calculate missing values by column
def missing_values_table(df):
	#total missing
	mis_val = df.isnull().sum()

	#percentage missing
	mis_val_percent = 100 * df.isnull().sum() / len(df)

	#results table
	mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

	#rename columns
	mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

	#descending sort by missing percentage
	mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)

	#summary
	print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n" + "There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")

	return mis_val_table_ren_columns

#calculate correlations coeffictient between two colums
def corr_func(x, y, **kwargs):
	r = np.corrcoef(x, y)[0][1]
	ax = plt.gca()
	ax.annotate("r = {:.2f}".format(r), xy=(.2, .8), xycoords=ax.transAxes, size=20)


#########################################################


########## Data Cleaning ##################

#read data into dataframe
data = pd.read_csv('data/Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv')

#column data types and non-missing vals
# data.info()

#replace not available with not a number
data = data.replace({'Not Available': np.nan})

#iterate through columns
for col in list(data.columns):
	#columns that should be numeric
	if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in col or 'therms' in col or 'gal' in col or 'ENERGY STAR Score' in col):
		#convert to float
		data[col] = data[col].astype(float)

#statistics
# data.describe()

#missing data table
# print(missing_values_table(data))


#columns with > 50% missing
missing_df = missing_values_table(data)
missing_columns = list(missing_df[missing_df['% of Total Values'] > 50].index)
print('We will remove %d columns.' % len(missing_columns))

#drop columns
data = data.drop(columns = list(missing_columns))


##################################################


####### Exploritory Data Analysis ################


####### Single Variable Plots

#energy star score histogram
# plt.style.use('fivethirtyeight')
# plt.hist(data['ENERGY STAR Score'].dropna(), bins = 100, edgecolor = 'k');
# plt.xlabel('Score'); plt.ylabel('Number of Buildings');
# plt.title('Energy Star Score Distribution');


#site EUI histogram
# plt.figure(figsize = (8,8))
# plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins = 20, edgecolor = 'black');
# plt.xlabel('Site EUI');
# plt.ylabel('Count');
# plt.title('Site EUI Distribution');


#find outlier
#print(data['Site EUI (kBtu/ft²)'].dropna().sort_values().tail(10))

#outlier detail
#print(data.loc[data['Site EUI (kBtu/ft²)'] == 869265, :])

#first and third quartile
first_quartile = data['Site EUI (kBtu/ft²)'].describe()['25%']
third_quartile = data['Site EUI (kBtu/ft²)'].describe()['75%']

#interquartile range
iqr = third_quartile - first_quartile

#remove outliers
data = data[(data['Site EUI (kBtu/ft²)'] > (first_quartile - 3 * iqr)) & (data['Site EUI (kBtu/ft²)'] < (third_quartile + 3 * iqr))]

#updated EUI histogram(without outliers)
# plt.figure(figsize = (8,8))
# plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins = 20, edgecolor = 'black');
# plt.xlabel('Site EUI');
# plt.ylabel('Count');
# plt.title('Site EUI Distribution');

#plt.show()

########## Relationships

#list of buildings with more than 100 measurments
types = data.dropna(subset=['ENERGY STAR Score'])
types = types['Largest Property Use Type'].value_counts()
types = list(types[types.values > 100].index)

#plot distribution of scores for building categories
# plt.figure(figsize = (12,10))

# #plot each building
# for build_type in types:
# 	#select building type
# 	subset = data[data['Largest Property Use Type'] == build_type]

# 	#density plot of Energy Star Scores
# 	sns.kdeplot(subset['ENERGY STAR Score'].dropna(), label = build_type, shade = False, alpha = 0.8);

# #plot labels
# plt.xlabel('Energy Star Score', size = 20); plt.ylabel('Density', size = 20);
# plt.title('Density Plot of Energy Star Scores by Building Type', size = 28);

# plt.show()

#list of boroughs with more than 100 observations
boroughs = data.dropna(subset=['ENERGY STAR Score'])
boroughs = boroughs['Borough'].value_counts()
boroughs = list(boroughs[boroughs.values > 100].index)

#plot distrubution of scores for boroughs
# plt.figure(figsize = (12,10))

# #plot each borough distribution of scores
# for borough_type in boroughs:
# 	#select borough type
# 	subset = data[data['Borough'] == borough_type]
# 	#density plot of Energy Star Scores
# 	sns.kdeplot(subset['ENERGY STAR Score'].dropna(), label = borough_type);

# #plot labels
# plt.xlabel('Energy Star Score', size = 20); plt.ylabel('Density', size = 20);
# plt.title('Density Plot of Energy Star Scores by Borough', size = 28);

# plt.show()

########### Correlations between Features and Target

#sorted correlations
correlations_data = data.corr()['ENERGY STAR Score'].sort_values()

#most negative correlations
# print(correlations_data.head(15), '\n')

#most positive correlations
# print(correlations_data.tail(15))

#non linear correlations

#select numeric columns
numeric_subset = data.select_dtypes('number')

#create columns with square root and log of numeric columns
for col in numeric_subset.columns:
	#skip Score
	if col == 'ENERGY STAR Score':
		next
	else:
		numeric_subset['sqrt_' + col] = np.sqrt(numeric_subset[col])
		numeric_subset['log_' + col] = np.log(numeric_subset[col])

#select categorical columns
categorical_subset = data[['Borough', 'Largest Property Use Type']]

#one hot encode
categorical_subset = pd.get_dummies(categorical_subset)

#concat dataframs (axis = 1 for column bind)
features = pd.concat([numeric_subset, categorical_subset], axis = 1)

#drop buildings without energy score
features = features.dropna(subset = ['ENERGY STAR Score'])

#find correlations 
correlations = features.corr()['ENERGY STAR Score'].dropna().sort_values()

#most negative correlations
#print(correlations.head(15))

#most positive correlations
#print(correlations.tail(15))

############## Two-Variable Plots

#Scatter Plot
# plt.figure(figsize = (10,8))

# #building types
# features['Largest Property Use Type'] = data.dropna(subset = ['ENERGY STAR Score'])['Largest Property Use Type']

# #limit to building types with > 100 observations
# features = features[features['Largest Property Use Type'].isin(types)]

# #scatterplot of Score vs Log Source EUI
# sns.lmplot('Site EUI (kBtu/ft²)', 'ENERGY STAR Score', hue = 'Largest Property Use Type', data = features, scatter_kws = {'alpha': 0.8, 's' : 60}, fit_reg = False, size = 10, aspect = 1.2)
# plt.xlabel('Site EUI', size = 12);plt.ylabel('Energy Star Score', size = 12);
# plt.title('Energy Star Score vs Site EUI', size = 28)

# plt.show()

#Pairs Plot

#extract columns to plot
plot_data = features[['ENERGY STAR Score', 'Site EUI (kBtu/ft²)', 'Weather Normalized Source EUI (kBtu/ft²)', 'log_Total GHG Emissions (Metric Tons CO2e)']]

#replace inf with nan
plot_data = plot_data.replace({np.inf: np.nan, -np.inf: np.nan})

#rename calumns
plot_data = plot_data.rename(columns = {'Site EUI (kBtu/ft²)': 'Site EUI', 'Weather Normalized Source EUI (kBtu/ft²)': 'Weather Norm EUI', 'log_Total GHG Emissions (Metric Tons CO2e)' : 'log GHG Emissions'})

#drop na vals
plot_data = plot_data.dropna()

#create pairgrid object
grid = sns.PairGrid(data = plot_data, size=3)

#upper is scatter plot
grid.map_upper(plt.scatter, color='red', alpha=0.6)

#diagonal is histogram
grid.map_diag(plt.hist, color ='red', edgecolor='black')

#bottom is correlation and density plot
grid.map_lower(corr_func);
grid.map_lower(sns.kdeplot, cmap=plt.cm.Reds)

plt.suptitle('Pairs Plot of Energy Data', size=36, y=1.02);

plt.show()

#########################################################

