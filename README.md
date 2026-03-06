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

3) If you have access to our data, place the ```q2-ucsd-consDF.pqt```, ```q2-ucsd-acctDF.pqt```, ```q2-ucsd-trxnDF.pqt```, and ```q2-ucsd-cat-map.csv``` files in a new folder called ```data/``` in the main directory. <br>

4) Run the entire pipeline with ```python3 run.py```

## File Structure
```project-root/
│
├── README.md                               # Project overview and documentation
│
├── .gitignore                              # Marks data not tracked in this repository
│
├── poster.pdf                              # Project Poster showing visualizations and results
│
├── report.pdf                              # Project Report detailing methodology and analysis
│
├── notebooks/                              # Each notebook of all our progress 
│   ├── ellie_features.ipynb                # Ellie's notebooks
│   ├── ellie_w4_vis.ipynb
│   ├── ellie_w4.ipynb
│   ├── ellie_w5.ipynb
│   ├── ellie_w6.ipynb
│   ├── ellie_w7.ipynb
│   ├── ellie_w8.ipynb
│   │
│   ├── features_balance.ipynb
│   ├── features_categories.ipynb
│   │
│   ├── ht_q2w1_v1.ipynb                    # Heidi's notebooks
│   ├── ht_q2w1_v2.ipynb
│   ├── ht_w2_part1.ipynb
│   ├── ht_w2_part2.ipynb
│   ├── ht_w3_part1.ipynb
│   ├── ht_w4_graphs.ipynb
│   ├── ht_w4_graphs_v2.ipynb
│   ├── ht_w4.ipynb
│   ├── ht_w5.ipynb
│   │
│   ├── jasmine_features (3).ipynb          # Jasmine's notebooks
│   ├── jasmine_features (4).ipynb
│   ├── jasminefeatures.ipynb
│   │
│   ├── jh.ipynb
│   ├── run.ipynb
│   │
│   ├── tsu_w2.ipynb                        # David's notebooks
│   ├── tsu_w3.ipynb
│   ├── tsu_w4.ipynb
│   ├── tsu_w6.ipynb
│   ├── tsu_w7.ipynb
│   └── tsu_w8.ipynb                        
│
├── environment.yml                         # Conda environment specification for reproducibility 
│
├── results/                                # Holds all visual outputs like visualizations
│
└── run.py                                  # Main script for replicating all analysis and models                             

```

## Conclusion
From October-December 2025, our goal was to create a strong model that could accurately predict the categories people's spendings would be attributed to. After applying Regex processing and thoroughly testing a variety of models, we discovered DistilBERT to be the most reliable, offering an accuracy of 97%. <br>
From January 2026-present, we are building on our previous findings to build a model that uses consumer spending categories to help predict whether the respective consumer is delinquent. Subsequently, we will cast our predictions of the probability a consumer is delinquent and scale them into an understandable credit risk score. 