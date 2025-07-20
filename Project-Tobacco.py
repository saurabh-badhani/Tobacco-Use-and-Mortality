import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100 # Adjust for higher resolution plots if needed

# --- Data Loading ---
print("--- Data Loading ---")
dfs = {}
try:
    dfs['smokers'] = pd.read_csv('smokers.csv')
    dfs['prescriptions'] = pd.read_csv('prescriptions.csv')
    dfs['metrics'] = pd.read_csv('metrics.csv')
    dfs['fatalities'] = pd.read_csv('fatalities.csv')
    dfs['admissions'] = pd.read_csv('admissions.csv')
    for name in dfs:
        print(f"Loaded {name}.csv")
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Ensure all CSV files are in the same directory as this script.")
    exit() # Exit if essential files are not found

# --- Data Cleaning and Preprocessing ---
print("\n--- Data Cleaning and Preprocessing ---")

# 1. Clean Smokers DataFrame
print("\nCleaning 'smokers' DataFrame...")
dfs['smokers']['Sex'] = dfs['smokers']['Sex'].fillna('Unknown')
# Convert '16 and Over' to numeric, coercing errors, then fill NaNs with mean
dfs['smokers']['16 and Over'] = pd.to_numeric(dfs['smokers']['16 and Over'], errors='coerce')
dfs['smokers']['16 and Over'] = dfs['smokers']['16 and Over'].fillna(dfs['smokers']['16 and Over'].mean())

print(f"Missing values in 'smokers' after cleaning:\n{dfs['smokers'].isnull().sum()}")

# 2. Clean Prescriptions DataFrame
print("\nCleaning 'prescriptions' DataFrame...")
dfs['prescriptions']['Year'] = dfs['prescriptions']['Year'].astype(str).str.split('/').str[0].astype(int)
for col in ['Varenicline (Champix) Prescriptions', 'Net Ingredient Cost of Varenicline (Champix)']:
    if col in dfs['prescriptions'].columns:
        dfs['prescriptions'][col] = pd.to_numeric(dfs['prescriptions'][col], errors='coerce') # Ensure numeric
        dfs['prescriptions'][col] = dfs['prescriptions'][col].fillna(dfs['prescriptions'][col].mean())

print(f"Missing values in 'prescriptions' after cleaning:\n{dfs['prescriptions'].isnull().sum()}")

# 3. Clean Metrics DataFrame
print("\nCleaning 'metrics' DataFrame...")
dfs['metrics'].columns = dfs['metrics'].columns.str.replace('\n', '')
for col in ['Household Expenditure on Tobacco', 'Household Expenditure Total', 'Expenditure on Tobacco as a Percentage of Expenditure']:
    if col in dfs['metrics'].columns:
        dfs['metrics'][col] = pd.to_numeric(dfs['metrics'][col], errors='coerce') # Ensure numeric
        dfs['metrics'][col] = dfs['metrics'][col].fillna(dfs['metrics'][col].mean())

print(f"Missing values in 'metrics' after cleaning:\n{dfs['metrics'].isnull().sum()}")

# 4. Clean Fatalities DataFrame
print("\nCleaning 'fatalities' DataFrame...")
dfs['fatalities']['Sex'] = dfs['fatalities']['Sex'].fillna('Unknown')
dfs['fatalities']['Value'] = pd.to_numeric(dfs['fatalities']['Value'], errors='coerce')
dfs['fatalities']['Value'] = dfs['fatalities']['Value'].fillna(dfs['fatalities']['Value'].mean())

print(f"Missing values in 'fatalities' after cleaning:\n{dfs['fatalities'].isnull().sum()}")

# 5. Clean Admissions DataFrame
print("\nCleaning 'admissions' DataFrame...")
dfs['admissions']['Year'] = dfs['admissions']['Year'].astype(str).str.split('/').str[0].astype(int)
dfs['admissions']['Value'] = dfs['admissions']['Value'].replace('.', np.nan) # Replace '.' with NaN first
dfs['admissions']['Value'] = pd.to_numeric(dfs['admissions']['Value'], errors='coerce')
dfs['admissions']['Value'] = dfs['admissions']['Value'].fillna(dfs['admissions']['Value'].mean())
dfs['admissions']['Sex'] = dfs['admissions']['Sex'].fillna('Unknown')

print(f"Missing values in 'admissions' after cleaning:\n{dfs['admissions'].isnull().sum()}")
print("\n--- Data Cleaning Complete ---")

# --- Data Integration (Merging) ---
print("\n--- Data Integration: Merging DataFrames ---")

# Prepare smokers data for merging: filter for 'Weighted' method and relevant columns
smokers_prevalence = dfs['smokers'][dfs['smokers']['Method'] == 'Weighted'].copy()
smokers_prevalence = smokers_prevalence.rename(columns={'16 and Over': 'Adult Smoking Prevalence'})[['Year', 'Sex', 'Adult Smoking Prevalence']]

# Merge Fatalities with Smokers, Prescriptions, and Metrics
# Use outer merge for prescriptions and metrics to keep all years, then fill NaNs for new columns if any
merged_fatalities = pd.merge(dfs['fatalities'], smokers_prevalence, on=['Year', 'Sex'], how='left')
merged_fatalities = pd.merge(merged_fatalities, dfs['prescriptions'], on='Year', how='left')
merged_fatalities = pd.merge(merged_fatalities, dfs['metrics'], on='Year', how='left')
# Fill any NaNs that result from merging non-overlapping years (especially for metrics/prescriptions)
merged_fatalities.fillna(merged_fatalities.mean(numeric_only=True), inplace=True)


# Merge Admissions with Smokers, Prescriptions, and Metrics
merged_admissions = pd.merge(dfs['admissions'], smokers_prevalence, on=['Year', 'Sex'], how='left')
merged_admissions = pd.merge(merged_admissions, dfs['prescriptions'], on='Year', how='left')
merged_admissions = pd.merge(merged_admissions, dfs['metrics'], on='Year', how='left')
# Fill any NaNs that result from merging non-overlapping years
merged_admissions.fillna(merged_admissions.mean(numeric_only=True), inplace=True)


print("\nMerged Fatalities DataFrame Head:")
print(merged_fatalities.head())
print(f"\nMerged Fatalities DataFrame Info:")
merged_fatalities.info()

print("\nMerged Admissions DataFrame Head:")
print(merged_admissions.head())
print(f"\nMerged Admissions DataFrame Info:")
merged_admissions.info()

print("\n--- Data Integration Complete ---")

# --- Exploratory Data Analysis (EDA) ---
print("\n--- Exploratory Data Analysis (EDA) ---")

# 1. Trend of Adult Smoking Prevalence
print("\nGenerating plot: Trend of Adult Smoking Prevalence...")
# Filter for overall (Unknown Sex) prevalence for a clearer trend, and common years
overall_prevalence_plot = smokers_prevalence[smokers_prevalence['Sex'] == 'Unknown'].copy()
overall_prevalence_plot = overall_prevalence_plot[overall_prevalence_plot['Year'].isin(merged_fatalities['Year'].unique())] # Align years for consistent plotting range

plt.figure(figsize=(12, 6))
sns.lineplot(data=overall_prevalence_plot, x='Year', y='Adult Smoking Prevalence', marker='o')
plt.title('Trend of Adult Smoking Prevalence (Overall)')
plt.xlabel('Year')
plt.ylabel('Adult Smoking Prevalence (%)')
plt.grid(True)
plt.xticks(overall_prevalence_plot['Year'].unique(), rotation=45)
plt.tight_layout()
plt.show()
print("Plot generated: Adult Smoking Prevalence Trend.")

# 2. Trends in Smoking-Related Fatalities
print("\nGenerating plot: Trends in Fatalities caused by smoking...")
fatalities_smoking_related = merged_fatalities[
    merged_fatalities['Diagnosis Type'] == 'All deaths which can be caused by smoking'
].groupby(['Year', 'Sex'])['Value'].sum().unstack()

plt.figure(figsize=(14, 7))
fatalities_smoking_related.plot(kind='line', marker='o', ax=plt.gca())
plt.title('Trends in Fatalities Caused by Smoking (by Sex)')
plt.xlabel('Year')
plt.ylabel('Number of Fatalities')
plt.grid(True)
plt.xticks(fatalities_smoking_related.index, rotation=45)
plt.legend(title='Sex')
plt.tight_layout()
plt.show()
print("Plot generated: Fatalities Caused by Smoking (by Sex).")

# 3. Trends in Smoking-Related Admissions
print("\nGenerating plot: Trends in Admissions caused by smoking...")
admissions_smoking_related = merged_admissions[
    merged_admissions['Diagnosis Type'] == 'All diseases which can be caused by smoking'
].groupby(['Year', 'Sex'])['Value'].sum().unstack()

plt.figure(figsize=(14, 7))
admissions_smoking_related.plot(kind='line', marker='o', ax=plt.gca())
plt.title('Trends in Admissions Caused by Smoking (by Sex)')
plt.xlabel('Year')
plt.ylabel('Number of Admissions')
plt.grid(True)
plt.xticks(admissions_smoking_related.index, rotation=45)
plt.legend(title='Sex')
plt.tight_layout()
plt.show()
print("Plot generated: Admissions Caused by Smoking (by Sex).")

# 4. Trends in Smoking Cessation Prescriptions
print("\nGenerating plot: Trends in All Pharmacotherapy Prescriptions...")
# Corrected line: Use dfs['prescriptions'] instead of merged_prescriptions
prescriptions_trend = dfs['prescriptions'].groupby('Year')['All Pharmacotherapy Prescriptions'].sum()
# Filter for common years available in merged data (e.g., in admissions)
prescriptions_trend = prescriptions_trend[prescriptions_trend.index.isin(merged_admissions['Year'].unique())]

plt.figure(figsize=(12, 6))
prescriptions_trend.plot(kind='line', marker='o')
plt.title('Trend of All Pharmacotherapy Prescriptions')
plt.xlabel('Year')
plt.ylabel('Number of Prescriptions')
plt.grid(True)
plt.xticks(prescriptions_trend.index, rotation=45)
plt.tight_layout()
plt.show()
print("Plot generated: All Pharmacotherapy Prescriptions Trend.")


# 5. Trends in Tobacco Expenditure
print("\nGenerating plot: Trends in Household Expenditure on Tobacco...")
expenditure_trend = dfs['metrics'].groupby('Year')[['Household Expenditure on Tobacco', 'Expenditure on Tobacco as a Percentage of Expenditure']].mean()
# Filter for common years available in merged data
expenditure_trend = expenditure_trend[expenditure_trend.index.isin(merged_admissions['Year'].unique())]


plt.figure(figsize=(14, 7))
ax1 = expenditure_trend['Household Expenditure on Tobacco'].plot(kind='line', marker='o', color='tab:blue')
ax1.set_title('Trends in Household Tobacco Expenditure')
ax1.set_xlabel('Year')
ax1.set_ylabel('Household Expenditure on Tobacco', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

ax2 = ax1.twinx() # Create a second y-axis
sns.lineplot(data=expenditure_trend, x=expenditure_trend.index, y='Expenditure on Tobacco as a Percentage of Expenditure', marker='x', color='tab:red', ax=ax2, legend=False)
ax2.set_ylabel('Expenditure on Tobacco as % of Total Expenditure', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, ['Household Expenditure on Tobacco', 'Expenditure on Tobacco as % of Total Expenditure'], loc='upper left')

plt.xticks(expenditure_trend.index, rotation=45)
plt.tight_layout()
plt.show()
print("Plot generated: Trends in Household Tobacco Expenditure.")

# 6. Correlation Analysis
print("\nGenerating Correlation Heatmap for key numerical variables...")
# Select numerical columns from merged_fatalities for correlation
# Exclude identifier columns and ensure only truly numerical data is included
numerical_cols = merged_fatalities.select_dtypes(include=np.number).columns.tolist()
# Filter out columns that are not relevant for direct correlation analysis or are ID-like
correlation_data = merged_fatalities[numerical_cols].drop(columns=['Year'], errors='ignore') # Year is a time index, not a feature for direct correlation

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Key Variables in Merged Fatalities Data')
plt.tight_layout()
plt.show()
print("Correlation Heatmap generated.")

print("\n--- EDA Complete ---")

# --- Feature Engineering (Preparation for Machine Learning) ---
print("\n--- Feature Engineering (Preparation for Machine Learning) ---")

# For the purpose of demonstration, let's prepare a dataset for predicting Fatalities.
# We'll use merged_fatalities, focusing on 'All deaths which can be caused by smoking' for the target.
# Select features that are relevant for predicting smoking-related fatalities.
# Target variable: 'Value' (number of fatalities)
# Features: Smoking Prevalence, Prescription counts, Economic metrics, Year

# Filter merged_fatalities for the specific diagnosis type to create a focused dataset
df_model_fatalities = merged_fatalities[
    merged_fatalities['Diagnosis Type'] == 'All deaths which can be caused by smoking'
].copy()

# Drop columns not suitable as features or that are highly redundant
columns_to_drop_fatalities = [
    'ICD10 Code', 'ICD10 Diagnosis', 'Diagnosis Type', 'Metric', # Categorical identifiers for diagnosis/metric
    'All Pharmacotherapy Prescriptions', # Keep specific ones or sum them if needed
    'Net Ingredient Cost of All Pharmacotherapies',
    'Net Ingredient Cost of Nicotine Replacement Therapies (NRT)',
    'Net Ingredient Cost of Bupropion (Zyban)',
    'Net Ingredient Cost of Varenicline (Champix)',
    'Household Expenditure Total' # Might be too broad, Expenditure on Tobacco as % is more specific
]

# Create new features (example: ratio or simpler aggregates)
# For now, we'll keep the direct columns. More complex feature engineering would go here.

# Select final features (X) and target (y)
features_fatalities = [
    'Year',
    'Adult Smoking Prevalence',
    'Nicotine Replacement Therapy (NRT) Prescriptions',
    'Bupropion (Zyban) Prescriptions',
    'Varenicline (Champix) Prescriptions',
    'Tobacco PriceIndex',
    'Retail PricesIndex',
    'Tobacco Price Index Relative to Retail Price Index',
    'Real Households\' Disposable Income',
    'Affordability of Tobacco Index',
    'Household Expenditure on Tobacco',
    'Expenditure on Tobacco as a Percentage of Expenditure'
]

# Corrected line: Fixed SyntaxError by closing the bracket and completing the condition
df_model_fatalities_overall = df_model_fatalities[df_model_fatalities['Sex'] == 'Unknown'].copy()
# Or, if we want to include sex as a feature, we'd one-hot encode it here:
# df_model_fatalities = pd.get_dummies(df_model_fatalities, columns=['Sex'], drop_first=True)
# Adjust features_fatalities list if 'Sex' is one-hot encoded.

X_fatalities = df_model_fatalities_overall[features_fatalities]
y_fatalities = df_model_fatalities_overall['Value']

print(f"\nFeatures (X) for Fatalities Model (first 5 rows):\n{X_fatalities.head()}")
print(f"\nTarget (y) for Fatalities Model (first 5 rows):\n{y_fatalities.head()}")
print(f"\nShape of X_fatalities: {X_fatalities.shape}")
print(f"Shape of y_fatalities: {y_fatalities.shape}")

# Similar preparation for Admissions
df_model_admissions = merged_admissions[
    merged_admissions['Diagnosis Type'] == 'All diseases which can be caused by smoking'
].copy()

columns_to_drop_admissions = [
    'ICD10 Code', 'ICD10 Diagnosis', 'Diagnosis Type', 'Metric',
    'All Pharmacotherapy Prescriptions',
    'Net Ingredient Cost of All Pharmacotherapies',
    'Net Ingredient Cost of Nicotine Replacement Therapies (NRT)',
    'Net Ingredient Cost of Bupropion (Zyban)',
    'Net Ingredient Cost of Varenicline (Champix)',
    'Household Expenditure Total'
]

# Corrected line for admissions: Fixed SyntaxError similar to fatalities
df_model_admissions_overall = df_model_admissions[df_model_admissions['Sex'] == 'Unknown'].copy()

X_admissions = df_model_admissions_overall[features_fatalities] # Using same features as fatalities for consistency
y_admissions = df_model_admissions_overall['Value']

print(f"\nFeatures (X) for Admissions Model (first 5 rows):\n{X_admissions.head()}")
print(f"\nTarget (y) for Admissions Model (first 5 rows):\n{y_admissions.head()}")
print(f"\nShape of X_admissions: {X_admissions.shape}")
print(f"Shape of y_admissions: {y_admissions.shape}")


print("\n--- Feature Engineering Preparation Complete ---")
print("\nThis script has loaded, cleaned, merged your data, performed extensive EDA, and prepared initial feature sets (X and y) for machine learning models.")
print("The next steps would involve selecting and training machine learning models, evaluating their performance, and interpreting their results.")