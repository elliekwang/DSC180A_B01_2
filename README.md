# Categorizing Banking Transaction Memos
**Contributors:** Jasmine Hong, Heidi Tam, David Tsukamoto, Ellie Wang <br>
**Creation Date:** October 8, 2025 <br>
**Last Updated:** February 14, 2026 <br>
**Website:** https://heiditam.github.io/fairer-credit-cashflow/

## Overview
Currently, one of the most widely used metrics for evaluating an individual's likelihood to pay back a loan is through the credit score, such as FICO or VantageScore. However, these metrics have their own limitations. Elderly people, for instance, may have not made purchases in the recent past, which can lower their credit score and make it more difficult for them to make large purchases, even if they previously maintained high credit scores and paid all their bills on time. On the other side of the spectrum, younger people may be reliable individuals but have a low credit score due to limited credit history. This project aims to use natural language processing to better understand the likelihood of people paying off their loans in two parts: <br> <br>

1) From October to December 2025, we use the text from banking transaction memos (about 2 million records) and build a strong and reliable model, aiming to classify the spending category each transaction memo falls under. Some example categories include education, food and beverages, and general merchandise. 

2) From January to March 2026, our mission is to use natural language processing to develop a more advanced machine learning model that provides a reliable score that estimates credit risk.  <br>

## Running the Project
1) Navigate into the respective folder and run the following command in your command line or terminal: <br>
```git clone https://github.com/elliekwang/DSC180A_B01_2.git``` <br>

2) Set up the environment and activate it: <br>
 ```conda env create -f environment.yml``` <br>
 ```conda activate dsc180-q2``` <br>

Dependencies (see environment.yml)
- ```python=3.12```  
- ```numpy```  
- ```pandas```  
- ```scikit-learn``` 
- ```matplotlib```  
- ```plotly```  
- ```xgboost```  
- ```lightgbm```
- ```shap```

3) If you have access to our data, place the ```q2-ucsd-consDF.pqt```, ```q2-ucsd-acctDF.pqt```, ```q2-ucsd-trxnDF.pqt```, and ```q2-ucsd-cat-map.csv``` files in a new folder called ```data/``` in the main directory. <br>

4) Run the entire pipeline with ```python3 run.py```
This will:

- build features  
- select features  
- train models  
- generate reason codes  
- score holdout consumers  
- save outputs in results

---

## Expected Outputs
```
results/
│
├── roc_curve.png
├── lift_curve.png
├── classification_report.txt
├── holdout_scores.csv
└── reason_codes.csv
```

Outputs include:

- model metrics
- ROC / lift curves
- delinquency scores
- reason codes
- holdout predictions

---
## File Structure
```project-root/

│
├── README.md                               # Project overview and documentation
├── .gitignore                              # Marks data not tracked in this repository
├── poster.pdf                              # Project poster with visualizations and results
├── report.pdf                              # Final project report
├── environment.yml                         # Conda environment for reproducibility
├── results/                                # Saved figures / outputs
│
├── notebooks/                              # All notebooks organized by workflow stage
│
│   ├── feature_engineering/                # Created features based on categories, income, balance, etc. 
│   │   ├── ellie_features.ipynb
│   │   ├── features_balance.ipynb
│   │   ├── features_categories.ipynb
│   │   ├── ht_q2w1_v1.ipynb
│   │   ├── ht_q2w1_v2.ipynb
│   │   ├── ht_w2_part1.ipynb
│   │   ├── ht_w2_part2.ipynb
│   │   ├── jasmine_features (3).ipynb
│   │   ├── jasmine_features (4).ipynb
│   │   └── tsu_w2.ipynb
│
│   ├── feature_selection/                  # Selected top 50 best performing features for the final model
│   │   ├── ellie_w4_vis.ipynb
│   │   ├── ellie_w4.ipynb
│   │   ├── ht_w3_part1.ipynb
│   │   ├── ht_w4_graphs_v2.ipynb
│   │   ├── ht_w4_graphs.ipynb
│   │   ├── ht_w4.ipynb
│   │   ├── jasminefeatures.ipynb
│   │   ├── tsu_w3.ipynb
│   │   └── tsu_w4.ipynb
│
│   ├── model_creation/                     # Trained LogReg, XGBoost, Random Forest, and LightGBM and performed hyperparameter tuning
│   │   ├── ellie_w5.ipynb
│   │   ├── ellie_w6.ipynb
│   │   ├── ht_w5.ipynb
│   │   ├── jh.ipynb
│   │   └── tsu_w6.ipynb
│
│   ├── reason_codes/                       # Created 3 reasons for why we would predict delinquency
│   │   ├── ellie_w7.ipynb
│   │   └── tsu_w7.ipynb
│
│   ├── scoring/                            # Applied our model on a holdout set and created visualizations
│   │   ├── ellie_w8.ipynb
│   │   └── tsu_w8.ipynb
│
│   └── run.ipynb                           # Notebook version of pipeline
│
└── run.py                                  # Main script to reproduce results                 

```
Directory description:

```feature_engineering``` → create financial features  
```feature_selection``` → choose top predictors  
```model_creation``` → train ML models  
```reason_codes``` → explain predictions  
```scoring``` → apply model to holdout set  

## Conclusion
From October-December 2025, our goal was to create a strong model that could accurately predict the categories people's spendings would be attributed to. After applying Regex processing and thoroughly testing a variety of models, we discovered DistilBERT to be the most reliable, offering an accuracy of 97%. <br>
From January 2026-present, we are building on our previous findings to build a model that uses consumer spending categories to help predict whether the respective consumer is delinquent. Subsequently, we will cast our predictions of the probability a consumer is delinquent and scale them into an understandable credit risk score. 