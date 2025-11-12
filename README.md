# Algerian Forest Fire Prediction using Machine Learning

This project predicts the **Fire Weather Index (FWI)** and classifies regions into *Fire* and *Not Fire* categories using environmental and meteorological data from Algeria.
It covers the full **machine learning pipeline** — from data preprocessing and visualization to model training and evaluation using **Linear Regression**, **Lasso Regression**, and **Ridge Regression**.

## Dataset Information

**Dataset:** Algerian Forest Fires Dataset (UCI Machine Learning Repository)
**Total Samples:** 244 (122 from Bejaia region and 122 from Sidi Bel-Abbes region)
**Duration:** June 2012 – September 2012

The dataset contains 11 input attributes and 1 output attribute (class).
The data represent weather conditions and fire indices recorded during the forest fire season in Algeria.

**Features include:**

* Date (Day, Month, Year)
* Temperature (°C)
* Relative Humidity (%)
* Wind Speed (km/h)
* Rainfall (mm)
* Fine Fuel Moisture Code (FFMC)
* Duff Moisture Code (DMC)
* Drought Code (DC)
* Initial Spread Index (ISI)
* Buildup Index (BUI)
* Fire Weather Index (FWI)
* Class (Fire / Not Fire)

**Class Distribution:**

* Fire: 138 samples
* Not Fire: 106 samples

## Project Workflow

### Step 1: Data Preprocessing (`my_code.ipynb`)

* Imported the dataset using pandas
* Handled missing values and removed unwanted spaces in column names
* Converted columns to numeric types where required
* Added a new feature “Region” (0 for Bejaia, 1 for Sidi Bel-Abbes)
* Encoded the “Classes” column (Fire → 1, Not Fire → 0)
* Performed exploratory data analysis (EDA) using matplotlib and seaborn
* Created visualizations such as:

  * Distribution plots for weather attributes
  * Pie chart showing Fire vs Not Fire ratio
  * Correlation heatmap for features


### Step 2: Model Development (`my_model_training.ipynb`)

* Split the dataset into training and testing sets (75% / 25%)
* Standardized the data using **StandardScaler**
* Trained three regression models:

  1. **Linear Regression** – baseline model to understand relationships
  2. **Lasso Regression (L1 regularization)** – reduces overfitting and performs feature selection
  3. **Ridge Regression (L2 regularization)** – manages multicollinearity and improves stability
* Evaluated the models using:

  * R² Score
  * Mean Absolute Error (MAE)
  * Mean Squared Error (MSE)
* Visualized results with comparison plots and error charts

## Technologies and Libraries Used

* Python 3.x
* Jupyter Notebook
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

## Model Evaluation Summary

* **Linear Regression:** R² ≈ 0.97, good baseline performance
* **Lasso Regression:** R² ≈ 0.96, slightly higher error but effective feature selection
* **Ridge Regression:** R² ≈ 0.97, stable and well-balanced model

