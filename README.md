
#                                                     __Crime Classification Model using Pyspark__


```python
from IPython.display import Image
Image('spark_ml.png')
```




![png](output_1_0.png)




## 1. __Scope__
* We are interesting in a system that could classify crime discription into different categories. We want to create a system that could automatically assign a described crime to category which could  help law enforcements to assign right officers to crime or could automatically assign officers to crime based on the classification.  
* We are using dataset from Kaggle on San Francisco Crime. Our responsibilty is to train a model based on 39 pre-defined categories, test the model accuracy and  deploy it into production. Given a new crime description, the system should assign it to one of 39 categories.

* To solve this problem, we will use a variety of feature extraction techniques along with different supervised machine learning algorithms in Pyspark. 

* This is multi-class text classification problem.

## __2. Setup Spark and load other libraries__


```python
import pyspark
spark = pyspark.sql.SparkSession.builder.appName("clipper-pyspark").getOrCreate()

sc = spark.sparkContext
```


```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
np.random.seed(60)
```

## __3. Data Extraction__


```sh
%%sh
#Let see the first 5 rows
head -5 train.csv
```

    Dates,Category,Descript,DayOfWeek,PdDistrict,Resolution,Address,X,Y
    2015-05-13 23:53:00,WARRANTS,WARRANT ARREST,Wednesday,NORTHERN,"ARREST, BOOKED",OAK ST / LAGUNA ST,-122.425891675136,37.7745985956747
    2015-05-13 23:53:00,OTHER OFFENSES,TRAFFIC VIOLATION ARREST,Wednesday,NORTHERN,"ARREST, BOOKED",OAK ST / LAGUNA ST,-122.425891675136,37.7745985956747
    2015-05-13 23:33:00,OTHER OFFENSES,TRAFFIC VIOLATION ARREST,Wednesday,NORTHERN,"ARREST, BOOKED",VANNESS AV / GREENWICH ST,-122.42436302145,37.8004143219856
    2015-05-13 23:30:00,LARCENY/THEFT,GRAND THEFT FROM LOCKED AUTO,Wednesday,NORTHERN,NONE,1500 Block of LOMBARD ST,-122.42699532676599,37.80087263276921



```python
#Read the data into spark datafrome
from pyspark.sql.functions import col, lower
df = spark.read.format('csv')\
          .option('header','true')\
          .option('inferSchema', 'true')\
          .option('timestamp', 'true')\
          .load('train.csv')

data = df.select(lower(col('Category')),lower(col('Descript')))\
        .withColumnRenamed('lower(Category)','Category')\
        .withColumnRenamed('lower(Descript)', 'Description')
data.cache()
print('Dataframe Structure')
print('----------------------------------')
print(data.printSchema())
print(' ')
print('Dataframe preview')
print(data.show(5))
print(' ')
print('----------------------------------')
print('Total number of rows', df.count())
```

    Dataframe Structure
    ----------------------------------
    root
     |-- Category: string (nullable = true)
     |-- Description: string (nullable = true)
    
    None
     
    Dataframe preview
    +--------------+--------------------+
    |      Category|         Description|
    +--------------+--------------------+
    |      warrants|      warrant arrest|
    |other offenses|traffic violation...|
    |other offenses|traffic violation...|
    | larceny/theft|grand theft from ...|
    | larceny/theft|grand theft from ...|
    +--------------+--------------------+
    only showing top 5 rows
    
    None
     
    ----------------------------------
    Total number of rows 878049


**Explanation**: __To familiar ourselves with the dataset, we need to see the top list of the crime categories and descriptions__.


```python
def top_n_list(df,var, N):
    '''
    This function determine the top N numbers of the list
    '''
    print("Total number of unique value of"+' '+var+''+':'+' '+str(df.select(var).distinct().count()))
    print(' ')
    print('Top'+' '+str(N)+' '+'Crime'+' '+var)
    df.groupBy(var).count().withColumnRenamed('count','totalValue')\
    .orderBy(col('totalValue').desc()).show(N)
    
    
top_n_list(data, 'Category',10)
print(' ')
print(' ')
top_n_list(data,'Description',10)
```

    Total number of unique value of Category: 39
     
    Top 10 Crime Category
    +--------------+----------+
    |      Category|totalValue|
    +--------------+----------+
    | larceny/theft|    174900|
    |other offenses|    126182|
    |  non-criminal|     92304|
    |       assault|     76876|
    | drug/narcotic|     53971|
    | vehicle theft|     53781|
    |     vandalism|     44725|
    |      warrants|     42214|
    |      burglary|     36755|
    |suspicious occ|     31414|
    +--------------+----------+
    only showing top 10 rows
    
     
     
    Total number of unique value of Description: 879
     
    Top 10 Crime Description
    +--------------------+----------+
    |         Description|totalValue|
    +--------------------+----------+
    |grand theft from ...|     60022|
    |       lost property|     31729|
    |             battery|     27441|
    |   stolen automobile|     26897|
    |drivers license, ...|     26839|
    |      warrant arrest|     23754|
    |suspicious occurr...|     21891|
    |aided case, menta...|     21497|
    |petty theft from ...|     19771|
    |malicious mischie...|     17789|
    +--------------------+----------+
    only showing top 10 rows
    


**Explanation**: __Category feature will be our label (multi-class). How many classes?__


```python
data.select('Category').distinct().count()
```




    39



## __4. Partition the dataset into Training and Test dataset__


```python
training, test = data.randomSplit([0.7,0.3], seed=60)
#trainingSet.cache()
print("Training Dataset Count:", training.count())
print("Test Dataset Count:", test.count())
```

    Training Dataset Count: 615417
    Test Dataset Count: 262632


## __5. Define Structure to build Pipeline__
__The process of cleaning the dataset involves:__  
* __Define tokenization function using RegexTokenizer__: RegexTokenizer allows more advanced tokenization based on regular expression (regex) matching. By default, the parameter “pattern” (regex, default: “\s+”) is used as delimiters to split the input text. Alternatively, users can set parameter “gaps” to false indicating the regex “pattern” denotes “tokens” rather than splitting gaps, and find all matching occurrences as the tokenization result.  

* __Define stop remover function using StopWordsRemover__: StopWordsRemover takes as input a sequence of strings (e.g. the output of a Tokenizer) and drops all the stop words from the input sequences. The list of stopwords is specified by the stopWords parameter.  

* __Define bag of words function for Descript variable using CountVectorizer__: CountVectorizer can be used as an estimator to extract the vocabulary, and generates a CountVectorizerModel. The model produces sparse representations for the documents over the vocabulary, which can then be passed to other algorithms like LDA. During the fitting process, CountVectorizer will select the top vocabSize words ordered by term frequency across the corpus. An optional parameter minDF also affects the fitting process by specifying the minimum number (or fraction if < 1.0) of documents a term must appear in to be included in the vocabulary.  

* __Define function to Encode the values of category variable using StringIndexer__: StringIndexer encodes a string column of labels to a column of label indices. The indices are in (0, numLabels), ordered by label frequencies, so the most frequent label gets index 0. In our case, the label colum(Category) will be encoded to label indices, from 0 to 38; the most frequent label (LARCENY/THEFT) will be indexed as 0.

* __Define a pipeline to call these functions__: ML Pipelines provide a uniform set of high-level APIs built on top of DataFrames that help users create and tune practical machine learning pipelines.        


```python
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, OneHotEncoder, StringIndexer, VectorAssembler, HashingTF, IDF, Word2Vec
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, NaiveBayes 

#----------------Define tokenizer with regextokenizer()------------------
regex_tokenizer = RegexTokenizer(pattern='\\W')\
                  .setInputCol("Description")\
                  .setOutputCol("tokens")

#----------------Define stopwords with stopwordsremover()---------------------
extra_stopwords = ['http','amp','rt','t','c','the']
stopwords_remover = StopWordsRemover()\
                    .setInputCol('tokens')\
                    .setOutputCol('filtered_words')\
                    .setStopWords(extra_stopwords)
                    

#----------Define bags of words using countVectorizer()---------------------------
count_vectors = CountVectorizer(vocabSize=10000, minDF=5)\
               .setInputCol("filtered_words")\
               .setOutputCol("features")


#-----------Using TF-IDF to vectorise features instead of countVectoriser-----------------
hashingTf = HashingTF(numFeatures=10000)\
            .setInputCol("filtered_words")\
            .setOutputCol("raw_features")
            
#Use minDocFreq to remove sparse terms
idf = IDF(minDocFreq=5)\
        .setInputCol("raw_features")\
        .setOutputCol("features")

#---------------Define bag of words using Word2Vec---------------------------
word2Vec = Word2Vec(vectorSize=1000, minCount=0)\
           .setInputCol("filtered_words")\
           .setOutputCol("features")

#-----------Encode the Category variable into label using StringIndexer-----------
label_string_idx = StringIndexer()\
                  .setInputCol("Category")\
                  .setOutputCol("label")

#-----------Define classifier structure for logistic Regression--------------
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)

#---------Define classifier structure for Naive Bayes----------
nb = NaiveBayes(smoothing=1)

def metrics_ev(labels, metrics):
    '''
    List of all performance metrics
    '''
    # Confusion matrix
    print("---------Confusion matrix-----------------")
    print(metrics.confusionMatrix)
    print(' ')    
    # Overall statistics
    print('----------Overall statistics-----------')
    print("Precision = %s" %  metrics.precision())
    print("Recall = %s" %  metrics.recall())
    print("F1 Score = %s" % metrics.fMeasure())
    print(' ')
    # Statistics by class
    print('----------Statistics by class----------')
    for label in sorted(labels):
       print("Class %s precision = %s" % (label, metrics.precision(label)))
       print("Class %s recall = %s" % (label, metrics.recall(label)))
       print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))
    print(' ')
    # Weighted stats
    print('----------Weighted stats----------------')
    print("Weighted recall = %s" % metrics.weightedRecall)
    print("Weighted precision = %s" % metrics.weightedPrecision)
    print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
    print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
    print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)
    
```

## __6. Build Multi-Classification__
__The stages involve to perform multi-classification include:__
1. Model training and evaluation
   1. Build baseling model
      1. Logistic regression using CountVectorizer features
   2. Build secondary models
      1. Naive Bayes
      2. Logistic regression and Naive Bayes using TF-IDF features
      3. Logistic regression and Naive Bayes using word2Vec
    
 ### __(i) Baseline Model__ 
Baseline model should be quick, low cost and simple to setup and produce a decent results. One of the reason to consider baselines because they iterate very quickly, while wasting minimal time. To further undertand why and how to apply baselines, please refer to Emmanuel Ameisen's article: [Always start with a stupid model, no exceptions.](https://blog.insightdatascience.com/always-start-with-a-stupid-model-no-exceptions-3a22314b9aaa)

#### __(a). Apply Logistic Regression with  Count Vector Features__
We will build a model to make predictions and score on the test sets using logistics regression using the dataset we transformed using count vectors. And we will see the top 10 predictions from the highest probability from our model, accuracy and other metrics to evaluate our model.  

Note: Fit regex_tokenizer,stopwords_remover,count_vectors,label_string_idx, and lr functions into pipeline.  


```python
pipeline_cv_lr = Pipeline().setStages([regex_tokenizer,stopwords_remover,count_vectors,label_string_idx, lr])
model_cv_lr = pipeline_cv_lr.fit(training)
predictions_cv_lr = model_cv_lr.transform(test)
```


```python
print('-----------------------------Check Top 5 predictions----------------------------------')
print(' ')
predictions_cv_lr.select('Description','Category',"probability","label","prediction")\
                                        .orderBy("probability", ascending=False)\
                                        .show(n=5, truncate=30)
```

    -----------------------------Check Top 5 predictions----------------------------------
     
    +------------------------------+-------------+------------------------------+-----+----------+
    |                   Description|     Category|                   probability|label|prediction|
    +------------------------------+-------------+------------------------------+-----+----------+
    |theft, bicycle, <$50, no se...|larceny/theft|[0.8726782249097988,0.02162...|  0.0|       0.0|
    |theft, bicycle, <$50, no se...|larceny/theft|[0.8726782249097988,0.02162...|  0.0|       0.0|
    |theft, bicycle, <$50, no se...|larceny/theft|[0.8726782249097988,0.02162...|  0.0|       0.0|
    |theft, bicycle, <$50, no se...|larceny/theft|[0.8726782249097988,0.02162...|  0.0|       0.0|
    |theft, bicycle, <$50, no se...|larceny/theft|[0.8726782249097988,0.02162...|  0.0|       0.0|
    +------------------------------+-------------+------------------------------+-----+----------+
    only showing top 5 rows
    



```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator 
evaluator_cv_lr = MulticlassClassificationEvaluator().setPredictionCol("prediction").evaluate(predictions_cv_lr)
print(' ')
print('------------------------------Accuracy----------------------------------')
print(' ')
print('                       accuracy:{}:'.format(evaluator_cv_lr))
```

     
    ------------------------------Accuracy----------------------------------
     
                           accuracy:0.9721844116763713:


 ### __(ii). Secondary Models__
 #### __(a). Apply Naive Bayes with Count Vector Features__
Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes’ theorem with strong (naive) independence assumptions between the features. The spark.ml implementation currently supports both multinomial naive Bayes and Bernoulli naive Bayes.   

 Fit regex_tokenizer,stopwords_remover,count_vectors,label_string_idx, and nb functions into pipeline.


```python
### Secondary model using NaiveBayes
pipeline_cv_nb = Pipeline().setStages([regex_tokenizer,stopwords_remover,count_vectors,label_string_idx, nb])
model_cv_nb = pipeline_cv_nb.fit(training)
predictions_cv_nb = model_cv_nb.transform(test)
```


```python
evaluator_cv_nb = MulticlassClassificationEvaluator().setPredictionCol("prediction").evaluate(predictions_cv_nb)
print(' ')
print('--------------------------Accuracy-----------------------------')
print(' ')
print('                      accuracy:{}:'.format(evaluator_cv_nb))
```

     
    --------------------------Accuracy-----------------------------
     
                          accuracy:0.9933012222188159:


#### __(b). Apply Logistic Regression Using TF-IDF Features__ 
Term frequency-inverse document frequency (TF-IDF) is a feature vectorization method widely used in text mining to reflect the importance of a term to a document in the corpus. Denote a term by t, a document by _d_, and the corpus by _D_. Term frequency TF(t,d) is the number of times that term t appears in document _d_, while document frequency _DF(t,D)_ is the number of documents that contains term _t_. If we only use term frequency to measure the importance, it is very easy to over-emphasize terms that appear very often but carry little information about the document, e.g. “a”, “the”, and “of”. If a term appears very often across the corpus, it means it doesn’t carry special information about a particular document. Inverse document frequency is a numerical measure of how much information a term provides:   $$IDF(t,D) = log {|D|+1DF \over (t,D)+1} $$, where |D| is the total number of documents in the corpus. Since logarithm is used, if a term appears in all documents, its IDF value becomes 0. Note that a smoothing term is applied to avoid dividing by zero for terms outside the corpus. The TF-IDF measure is simply the product of TF and IDF:
$$TFIDF(t,d,D) = {TF(t,d)⋅IDF(t,D)} $$.  

There are several variants on the definition of term frequency and document frequency. In MLlib, we separate TF and IDF to make them flexible.  

Note: Fit regex_tokenizer, stopwords_remover,hashingTF, idf,label_string_idx, and nb functions into pipeline.


```python
pipeline_idf_lr = Pipeline().setStages([regex_tokenizer,stopwords_remover,hashingTf, idf, label_string_idx, lr])
model_idf_lr = pipeline_idf_lr.fit(training)
predictions_idf_lr = model_idf_lr.transform(test)
```


```python
print('-----------------------------Check Top 5 predictions----------------------------------')
print(' ')
predictions_idf_lr.select('Description','Category',"probability","label","prediction")\
                                        .orderBy("probability", ascending=False)\
                                        .show(n=5, truncate=30)
```

    -----------------------------Check Top 5 predictions----------------------------------
     
    +------------------------------+-------------+------------------------------+-----+----------+
    |                   Description|     Category|                   probability|label|prediction|
    +------------------------------+-------------+------------------------------+-----+----------+
    |theft, bicycle, <$50, no se...|larceny/theft|[0.8745035002793186,0.02115...|  0.0|       0.0|
    |theft, bicycle, <$50, no se...|larceny/theft|[0.8745035002793186,0.02115...|  0.0|       0.0|
    |theft, bicycle, <$50, no se...|larceny/theft|[0.8745035002793186,0.02115...|  0.0|       0.0|
    |theft, bicycle, <$50, no se...|larceny/theft|[0.8745035002793186,0.02115...|  0.0|       0.0|
    |theft, bicycle, <$50, no se...|larceny/theft|[0.8745035002793186,0.02115...|  0.0|       0.0|
    +------------------------------+-------------+------------------------------+-----+----------+
    only showing top 5 rows
    



```python
evaluator_idf_lr = MulticlassClassificationEvaluator().setPredictionCol("prediction").evaluate(predictions_idf_lr)
print(' ')
print('-------------------------------Accuracy---------------------------------')
print(' ')
print('                        accuracy:{}:'.format(evaluator_idf_lr))
```

     
    -------------------------------Accuracy---------------------------------
     
                            accuracy:0.9723359770202158:


#### __(c). Apply Naive Bayes with TF-IDF Features__


```python
pipeline_idf_nb = Pipeline().setStages([regex_tokenizer,stopwords_remover,hashingTf, idf, label_string_idx, nb])
model_idf_nb = pipeline_idf_nb.fit(training)
predictions_idf_nb = model_idf_nb.transform(test)
```


```python
evaluator_idf_nb = MulticlassClassificationEvaluator().setPredictionCol("prediction").evaluate(predictions_idf_nb)
print(' ')
print('-----------------------------Accuracy-----------------------------')
print(' ')
print('                          accuracy:{}:'.format(evaluator_idf_nb))
```

     
    -----------------------------Accuracy-----------------------------
     
                              accuracy:0.9950758205262961:


#### __(e). Apply Logistic Regression Using Word2Vec features__ 
Word2Vec is an Estimator which takes sequences of words representing documents and trains a Word2VecModel. The model maps each word to a unique fixed-size vector. The Word2VecModel transforms each document into a vector using the average of all words in the document; this vector can then be used as features for prediction, document similarity calculations, etc. 


```python
pipeline_wv_lr = Pipeline().setStages([regex_tokenizer,stopwords_remover, word2Vec, label_string_idx, lr])
model_wv_lr = pipeline_wv_lr.fit(training)
predictions_wv_lr = model_wv_lr.transform(test)
```


```python
evaluator_wv_lr = MulticlassClassificationEvaluator().setPredictionCol("prediction").evaluate(predictions_wv_lr)
print('--------------------------Accuracy------------')
print(' ')
print('                  accuracy:{}:'.format(evaluator_wv_lr))
```

    --------------------------Accuracy------------
     
                      accuracy:0.9073464410736654:


#### __(f). Apply Naive Bayes Using Word2Vec features__


```python
#pipeline_wv_nb = Pipeline().setStages([regex_tokenizer,stopwords_remover, word2Vec, label_string_idx, nb])
#model_wv_nb = pipeline_wv_nb.fit(training)
#predictions_wv_nb = model_wv_nb.transform(test)
```


```python
#evaluator_wv_nb = MulticlassClassificationEvaluator().setPredictionCol("prediction").evaluate(predictions_wv_nb)
#print('--------Accuracy------------')
#print(' ')
#print('accuracy:{}%:'.format(round(evaluator_wv_nb *100),2))
```

## 7. __Results:__
__The table below has accuracy of the models generated by different extraction techniques.__

|                    | Logistic Regression | Naive Bayes |
| -------------------|:-------------------:|------------:|
| Count Vectoriser   |  97.2%              |   99.3%     |
| TF-IDF             |  97.2%              |   99.5%     |
| Word2Vec           |  90.7%              |             |

**Explanation**: __As you can see, TF-IDF proves to be best vectoriser for this dataset, while Naive Bayes proves to be better algorithm for text analysis than Logistic regression.__

## __8. Deploy the Model__
We will use Flask. To know more about Flask, check [Full Stack Python.](https://www.fullstackpython.com/flask.html)


```python
Image('flask.jpg')
```




![jpeg](output_41_0.jpeg)




```python
from flask import Flask, request, jsonify
from pyspark.ml import PipelineModel
```


```python
app = Flask(__name__)
```


```python
# Load the Model
MODEL=pyspark.ml.PipelineModel("spark-naive-bayes-model")
```


```python
HTTP_BAD_REQUEST = 400
```


```python
@app.route('/predict')
def predict():
    Description = request.args.get('Description', default=None, type=str)
    
    # Reject request that have bad or missing values.
    if Description is None:
        # Provide the caller with feedback on why the record is unscorable.
        message = ('Record cannot be scored because of '
                   'missing or unacceptable values. '
                   'All values must be present and of type string.')
        response = jsonify(status='error',
                           error_message=message)
        # Sets the status code to 400
        response.status_code = HTTP_BAD_REQUEST
        return response
    
    features = [[Description]]
    predictions = MODEL.transform(features)
    label_pred = predictions.select("Description","Category","probability","prediction")
    return jsonify(status='complete', label=label_pred)
```


```python
if __name__ == '__main__':
    app.run(debug=True)
```


```python
import requests
#response = requests.get('http://127.0.0.1:5000/predict?Description=arson')
#response.text
```
