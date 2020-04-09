import pandas as pd
import numpy as np

def evaluate_recall(y, y_test, k=1):
    num_examples = float(len(y))    
    num_correct = 0 
    for predictions, label in zip(y, y_test):   
        print predictions,label
        if label in predictions[:k]:    
            num_correct += 1  
    return num_correct/num_examples
    
def predict_random(context, utterances):
    return np.random.choice(len(utterances), 10, replace=False)
 
# Evaluate Rndom predictor
test_df = pd.read_csv("/home/gliu/file_database/george_konno_github/chatBot/data/test.csv")
y_random = [predict_random(test_df.Context[x], test_df.iloc[x,1:].values) for x in range(len(test_df))]
 
y_test = np.zeros(len(y_random))
for n in [1, 2, 5, 10]:
    print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y_random, y_test, n)))
    
from sklearn.feature_extraction.text import TfidfVectorizer
class TFIDFPredictor:
    def __init__(self): 
        self.vectorizer = TfidfVectorizer() 
 
    def train(self, data):
        self.vectorizer.fit(np.append(data.Context.values,
                                data.Utterance.values))
    def predict(self, context, utterances):
        # Convert context and utterances into tfidf vector
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)
 
        # The dot product measures the similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
 
        # Sort by top results and return the indices in descending order
        return np.argsort(result, axis=0)[::-1]
test_df = pd.read_csv("/home/gliu/file_database/george_konno_github/myPuiblicGitProjects/chatBot/data/test.csv")
train_df = pd.read_csv("/home/gliu/file_database/george_konno_github/myPuiblicGitProjects/chatBot/data/train.csv")   
pred = TFIDFPredictor()
pred.train(train_df)

y = [pred.predict(test_df.Context[x], test_df.iloc[x,1:].values) for x in range(len(test_df))]
y_random = [predict_random(test_df.Context[x], test_df.iloc[x,1:].values) for x in range(len(test_df))]
 
y_test = np.zeros(len(y))
for n in [1, 2, 5, 10]:
    print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y, y_test, n)))
#print "ASK:"
#print test_df.Context[244]
#y = pred.predict(test_df.Context[3],test_df.iloc[3,1:].values)
##print y
##print np.argmax(y)
#print "REPLY:"
#print test_df.iloc[244,np.argmin(y)+1]