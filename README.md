# Airline Fare Optimization Using Machine Learning

This project builds a dynamic pricing prediction engine using machine 
learning techniques on a dataset of over 200,000 flight records. The 
goal was to identify the key factors that influence ticket pricing and 
build a model that can predict optimal fares with high accuracy.


## Project Overview

Airline ticket pricing is one of the most complex pricing problems in 
the travel industry. Fares change based on seasonality, route frequency, 
seat class, booking window, and competitor activity. This project applies 
regression modelling and feature engineering to understand and predict 
these pricing patterns.

The final model achieved 90% accuracy after hyperparameter tuning and 
provided actionable insights that could enable a 5 to 8 percent revenue 
uplift for airline operations teams.

## Key Results

- Model accuracy of 90% after hyperparameter tuning
- Over 200,000 flight records analysed
- 5 to 8 percent revenue uplift potential identified
- Key pricing variables isolated through feature importance analysis
- Full data privacy and QA compliance maintained throughout


## Problem Statement

Airlines lose significant revenue through inconsistent and reactive 
pricing strategies. Static pricing models fail to account for:

- Seasonal demand fluctuations across different routes
- The relationship between booking window and willingness to pay
- Seat class availability and its effect on last-minute pricing
- Route frequency and competition on popular corridors

This project builds a predictive model to address each of these factors 
systematically.


## Technologies Used

- Python — main language for data processing and modelling
- Pandas and NumPy — data manipulation and feature engineering
- Scikit-learn — regression models, hyperparameter tuning, evaluation
- Matplotlib and Seaborn — exploratory visualisation
- Jupyter Notebook — analysis and documentation environment


## Methodology

### Step 1 — Data Collection and Cleaning

The dataset contained over 200,000 flight records including route 
information, departure times, seat class, booking window, and final 
fare. Cleaning involved handling missing fare values, removing duplicate 
bookings, and encoding categorical variables such as airline and 
destination.

### Step 2 — Feature Engineering

New features were created to capture pricing signals not directly 
present in the raw data:

- Seasonality flags based on departure month and school holiday periods
- Route frequency score indicating how competitive a route is
- Booking window category grouping early, standard, and last-minute bookings
- Seat class weighting combining class type with availability ratio

### Step 3 — Model Building

Three regression approaches were tested:

- Linear Regression as the baseline model
- Random Forest Regression to capture non-linear relationships
- Gradient Boosting to improve on Random Forest with sequential learning

Models were evaluated using RMSE and R-squared scores on a 20% 
held-out test set.

### Step 4 — Hyperparameter Tuning

Grid search cross-validation was applied to the best performing model 
to optimise parameters including number of estimators, max depth, 
and learning rate. This improved model accuracy from 84% to 90%.

### Step 5 — Insights and Recommendations

Feature importance analysis revealed that booking window and route 
frequency were the two strongest predictors of fare variation. 
Recommendations were provided to operations teams on dynamic pricing 
windows that could capture 5 to 8 percent additional revenue.

## Key Findings

Booking window was the single most important pricing variable, 
accounting for a significant portion of fare variance across all routes.

Seasonal demand patterns showed clear peaks around school holidays 
and summer months, with fares rising 30 to 45 percent above baseline.

Route frequency had a strong inverse relationship with price on 
competitive corridors, confirming that competition suppresses fares 
on popular routes.

The final model provides a reliable pricing signal that airlines 
could integrate into existing revenue management systems.

---

## Repository Structure
