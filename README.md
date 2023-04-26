# Analysis on Arabic Tweets Towards COVID-19 Pandemic

## Pipeline

## Data preprocessing(train data)

## Data preprocessing(Test data)

## Feature Extraction
we basically extracted 3-features for the word embedding word embeddings from the data:
- Word2Vec.
- Bag of Words.
- TD-IDF.

### Feature Extraction(sentiment model)
1. Word2vec:
    - vector size = 9000
    - min_count = 5
    - window = 4
2. TD-IDF
### Feature Extraction(category model)
1. Word2vec:
    - vector size = 5000
    - min_count = 5
    - window = 4
2. TD-IDF

## Data Balancing
The given train data was biased and unbalanced. So, we used SMOTE to solve this problem.

**Before Balancing**: 
- Class=2, n=5207 (79.411%)
- Class=1, n=954 (14.549%) 
- Class=0, n=396 (6.039%) 

**After Balancing**: 
- Class=2, n=5207 (33.333%) 
- Class=1, n=5207 (33.333%) 
- Class=0, n=5207 (33.333%)


### Data Balancing(Sentiment)

### Data Balancing(Category)


## Model training(ML)
1. RandomForestClassifier
- The random forest is a classification algorithm consisting of many decision trees. 
- It uses bagging and features randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree.
2. XGBClassifier
- XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.
- Random Forest is the best choice over XGB as it is easy to tune and works well even if there are lots of missing data and more noise. 
- Overfitting will not happen easily. With accurate results, XGBoost is hard to work with if there is lots of noise.
3. GaussianNB
- Naive Bayes is a simple classification algorithm based on applying Bayes’ theorem with strong (naive) independence assumptions between the features.
4. SVM
- Support Vector Machine is a supervised machine learning algorithm which can be used for both classification or regression challenges. 
- The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.
5. AutoML
- AutoML is a method to automate the end-to-end process of applying machine learning to real-world problems. 
- It is a combination of automated model selection, hyperparameter tuning, and ensemble building.


## Results

### Score of Stance classification on the test dataset 
| Model        | RandomForest | XGB  | Naive Bayes | SVM  | AutoML |
| ------------ | ------------ | ---- | ----------- | ---- | ------ |
| Accuracy     | 0.72         | 0.71 | 0.35        | 0.56 | 0.73   |
| Macro avg    | 0.51         | 0.49 | 0.33        | 0.44 | 0.49   |
| Weighted avg | 0.75         | 0.74 | 0.42        | 0.63 | 0.75   |

### Score of Category classification on the test dataset 
| Model        | RandomForest | XGB  | Naive Bayes | SVM  | AutoML |
| ------------ | ------------ | ---- | ----------- | ---- | ------ |
| Accuracy     | 0.48         | 0.50 | 0.14        | 0.27 | 0.49   |
| Macro avg    | 0.25         | 0.24 | 0.11        | 0.22 | 0.25   |
| Weighted avg | 0.52         | 0.51 | 0.12        | 0.31 | 0.52   |
