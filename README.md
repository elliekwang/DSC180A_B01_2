# Categorizing Banking Transaction Memos
**Contributors:** Jasmine Hong, Heidi Tam, David Tsukamoto, Ellie Wang <br>
**Creation Date:** October 8, 2025 <br>
**Last Updated:** February 13, 2026 <br>

## Overview
Currently, one of the most widely used metrics for evaluating an individual's likelihood to pay back a loan is through the credit score, such as FICO or VantageScore. However, these metrics have their own limitations. Elderly people, for instance, may have not made purchases in the recent past, which can lower their credit score and make it more difficult for them to make large purchases, even if they previously maintained high credit scores and paid all their bills on time. On the other side of the spectrum, younger people may be reliable individuals but have a low credit score due to limited credit history. This project aims to use natural language processing to better understand the likelihood of people paying off their loans in two parts: <br> <br>

1) From October to December 2025, we use the text from banking transaction memos (about 2 million records) and build a strong and reliable model, aiming to classify the spending category each transaction memo falls under. Some example categories include education, food and beverages, and general merchandise. 

2) From January to March 2026, our mission is to use natural language processing to develop a more advanced machine learning model that provides a reliable score that estimates credit risk.  <br>

## Running the Project
1) Navigate into the respective folder and run the following command in your command line or terminal: <br>
```git clone https://github.com/elliekwang/DSC180A_B01_2.git``` <br>

2) Set up the environment and activate it: <br>
 ```conda env create -f environment.yml``` <br>
 ```conda activate dsc180-q1``` <br>

3) If you have access to our data, place the ```q2-ucsd-consDF.pqt```, ```q2-ucsd-acctDF.pqt```, ```q2-ucsd-trxnDF.pqt```, and ```q2-ucsd-cat-map.csv``` files in the ```data/``` directory if you have access to them. <br>

4) Run the entire pipeline with ```python3 run.py```

## File Structure
```project-root/
│
├── README.md                               # Project overview and documentation
│
├── Winter_2026/                              # Weekly notebooks and progress checkpoints
│   ├── EDA_w2.ipynb                        # EDA
│
│
├── images/                           
│   ├── fig1.png                     
│   ├── fig2.png              
│   ├── fig3.png            
│   ├── fig4.png                 
│   ├── fig5.png   
│   ├── fig6.png   
│   ├── fig7.png   
│   ├── fig8.png   
│   ├── fig9.png   
│
│
├── Fall_2025/                              # Weekly notebooks and progress checkpoints
│   ├── EDA_w2.ipynb                        # EDA
│   ├── Memo_Cleaning_w3.ipynb              # Text preprocessing and memo cleaning
│   ├── Feature_Creation_w4.ipynb           # Feature engineering and baseline models
│   ├── q1_checkpoint.ipynb                 # Quarter 1 checkpoint code
│   ├── q1_report.ipynb                     # Quarter 1 report code
│
├── environment.yml                         # Conda environment specification for reproducibility 
│
└── run.py                                  # Main script for replicating all analysis and models                             

```

## Conclusion
From October-December 2025, our goal was to create a strong model that could accurately predict the categories people's spendings would be attributed to. After applying Regex processing and thoroughly testing a variety of models, we discovered DistilBERT to be the most reliable, offering an accuracy of 97%. 
From January 2026-present, we are building on our previous findings to build a model that uses consumer spending categories to help predict whether the respective consumer is delinquent. Subsequently, we will cast our predictions of the probability a consumer is delinquent and scale them into an understandable credit risk score. 