# Flipkart-Reviews-Sentiment-Analysis
# Flipkart Reviews Sentiment Analysis using Python & AI

A Machine Learning and NLP-based project that classifies Flipkart product reviews as **positive** or **negative** using a **Decision Tree Classifier**, **WordNet-based sentiment dictionary**, and modern text processing techniques.

## Overview
This project analyzes customer feedback from Flipkart to understand sentiment trends and improve product recommendations. It uses Python, Machine Learning, and NLP to extract, preprocess, classify, and visualize text sentiment.

## Features
- Sentiment classification using **Decision Tree Classifier**
- WordNet-based sentiment dictionary
- Text preprocessing: tokenization, stopword removal, stemming
- TF-IDF feature extraction
- WordCloud & confusion matrix visualizations
- Evaluation using Accuracy, Precision, Recall, F1-score

## Technologies Used
- Python  
- Artificial Intelligence (AI)  
- Machine Learning  
- Natural Language Processing (NLP)

## Methodology
1. **Data Collection** – Flipkart product review dataset  
2. **Preprocessing** – cleaning, tokenizing, filtering irrelevant text  
3. **Feature Extraction** – TF-IDF  
4. **Model Training** – Decision Tree Classifier  
5. **Evaluation** – performance metrics  
6. **Visualization** – WordCloud, Confusion Matrix  

## Sample Code
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

pred = model.predict(X_train)
print("Accuracy:", accuracy_score(y_train, pred))
```

## Conclusion
The model achieved **93.6% accuracy**, proving useful for understanding customer sentiment and supporting data-driven decisions for e-commerce platforms.

## Future Enhancements
- Deep learning (LSTM, Transformers)
- Web deployment using Flask or Django
- Real-time sentiment dashboard
