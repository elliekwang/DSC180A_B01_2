# Categorizing Banking Transaction Memos
**Contributors:** Jasmine Hong, Heidi Tam, David Tsukamoto, Ellie Wang <br>
**Creation Date:** October 8, 2025 <br>
**Last Updated:** November 1, 2025 <br>

## Overview
Currently, one of the most widely used metrics for evaluating an individual's likelihood to pay back a loan is through the credit score, such as FICO or VantageScore. However, these metrics have their own limitations. Elderly people, for instance, may have not made purchases in the recent past, which can lower their credit score and make it more difficult for them to make large purchases, even if they previously maintained high credit scores and paid all their bills on time. On the other side of the spectrum, younger people may be reliable individuals but have a low credit score due to limited credit history. This project aims to use natural language processing to better understand the likelihood of people paying off their loans in two parts: <br> <br>

1) From October to December 2025, we use the text from banking transaction memos (about 2 million records) and build a strong and reliable model, aiming to classify the spending category each transaction memo falls under. Some example categories include education, food and beverages, and general merchandise. 

2) From January to March 2026, our mission is to use natural language processing to develop a more advanced machine learning model that provides a reliable score that estimates credit risk. 

## Running the Project
1) Navigate into the respective folder and run the following command in your command line or terminal: <br>
```git clone https://github.com/elliekwang/DSC180A_B01_2.git```

2) Install the necessary dependencies by running this line:
```pip install -r requirements.txt```

3) Place the ```ucsd-inflows.pqt``` and ```ucsd-outflows.pqt``` files in the ```data/``` directory. These represent the inflows (money that flows inward to one's bank account) and the outflows (people's spendings), respectively. 

4) Run the entire pipeline with ```python run.py```

## File Structure
[to be completed]

## Conclusion
This project is currently in progress. Please check again towards mid-December for a clear and refined report on our conclusions of Part I of this project. 