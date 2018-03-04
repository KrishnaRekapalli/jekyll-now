
Extracting information from text is a very important part of many data scence applications. Applications like sentiment analysis, topic classification need to convert the text to some numerical form so that the ML/DL algorithms can train and learn. Most of the data on thw web is in the text form! 

Simplest thing we can can do when given a set of text documents is to count the term frequency. We can unserstand it better with the help of an example. In train text we have three documnets which we eill use to form a vocabulary and then create vector represenations of the documents in the test text. We will use the amazing scikit learn library of python in this tutorial. 


```python
trainText = ["Particle physics is a study of matter" , 
              "What makes matter behave in a way is a very interesting question",
              "Centuries of research later we are still facing some big questions"]

testText = ["matter is made of elementary particles", 
            "the answers to interesting questions in physics is not always an easy find"]
```


```python

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
Cvectorizer = CountVectorizer(stop_words='english')

Cvectorizer.fit_transform(trainText)
print(Cvectorizer)

print(Cvectorizer.vocabulary_)

```

```python

CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words='english',
        strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)
{'particle': 8, 'physics': 9, 'study': 13, 'matter': 7, 'makes': 6, 'behave': 0, 'way': 14, 'interesting': 4, 'question': 10, 'centuries': 2, 'research': 12, 'later': 5, 'facing': 3, 'big': 1, 'questions': 11}
In [58]:

```




| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |


