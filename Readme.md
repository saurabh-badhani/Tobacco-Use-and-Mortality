# Tobacco Use and Mortality Analysis (2004–2015) 🚬📉

This repository contains a data analysis project focused on understanding the multifaceted impacts of tobacco use, its prevalence, associated healthcare burdens, and intervention efforts in England between 2004 and 2015.

---

## 📌 Project Overview

This project analyzes five distinct datasets to provide a comprehensive picture of the health issues related to smoking in England. The analysis aims to:

- Analyze trends in tobacco use, healthcare burden (admissions, fatalities), and cessation efforts (prescriptions).
- Visualize key statistics and patterns over time across different demographic groups.
- Identify correlations between smoking prevalence and various health metrics.

---

## 📂 Datasets Used

- **`smokers.csv`** – Smoking prevalence by year, sex, and age groups.
- **`prescriptions.csv`** – Prescriptions for cessation aids and costs.
- **`metrics.csv`** – Economic indicators like tobacco price index and expenditure.
- **`fatalities.csv`** – Deaths related to smoking, categorized by ICD10 codes and sex.
- **`admissions.csv`** – Hospital admissions caused by smoking-related conditions.

---

## 🧹 Data Cleaning & Preprocessing

- Filled missing values with appropriate strategies (e.g., mean or 'Unknown').
- Converted year formats from `YYYY/YY` to `YYYY`.
- Handled non-numeric entries like `.` in `Value` columns.
- Ensured all numerical columns were in correct data types for analysis.
- Cleaned newline characters and fixed inconsistent column names.

---

## 📊 Key Visualizations

- **Smoking Prevalence Trend (2004–2015)**: Line plot showing decline over time.
- **Fatalities & Admissions by Year & Sex**: Clear differences between male and female trends.
- **Cessation Prescriptions**: Increase or decline in Varenicline usage and cost.
- **Expenditure Analysis**: Household spending on tobacco and its percentage of total expenditure.
- **Correlation Heatmap**: Relationships among smoking rates, costs, fatalities, and admissions.

---

## 📈 Key Findings

- 📉 **Decline in Smoking**: Gradual decrease in adult smoking prevalence.
- 🚹 **Gender Disparity**: Males show consistently higher smoking rates and related deaths.
- ⚰️ **Smoking & Mortality Link**: Positive correlation between prevalence and fatalities.
- 💊 **Prescriptions vs Prevalence**: Slight upward trend in prescriptions during years of higher prevalence.
- 💸 **Tobacco Still a Burden**: High household expenditure despite falling usage.

---

## ⚙️ How to Run the Project

1. **Clone this repository**:
   ```bash
   git clone https://github.com/saurabh-badhani/Tobacco_Mortality_Analysis.git
   cd Tobacco_Mortality_Analysis
2.Place all datasets (smokers.csv, prescriptions.csv, etc.) in the root folder.

3.Install required libraries:
  
  pip install pandas numpy matplotlib seaborn

4.Run the analysis script:
  python tobacco_mortality_analysis.py

🛠️ Tools & Technologies Used

i-Python (Pandas, NumPy)

ii-Matplotlib & Seaborn (for plotting)

iii-VS Code (for development)

📁 Project Structure

Tobacco_Mortality_Analysis/
│

├── tobacco_mortality_analysis.py          # Main Python script

├── smokers.csv

├── prescriptions.csv

├── metrics.csv

├── fatalities.csv

├── admissions.csv

└──README.md                         # You're here!

📌 Author

Saurabh Badhani

GitHub: saurabh-badhani


