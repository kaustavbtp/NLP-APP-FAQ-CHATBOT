import os
from fastapi import FastAPI, Request
from pydantic import BaseModel 
import json
import nltk
import pandas as pd
import string
from nltk.corpus import stopwords
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')

class FAQBot:
    
    def getDataFrame(self, path, sheetName):
        xlsFile = pd.ExcelFile(path)
        dataframe = pd.read_excel(xlsFile, sheetName)
        dataframe = dataframe[['Question', 'Answer']]
        dataframe = dataframe.dropna().reset_index(drop=True)
        return dataframe

    def cleanText(self, text):
        word_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        return [lemmatizer.lemmatize(word.lower().translate(remove_punct_dict)) for word in word_tokenizer.tokenize(text)]
    
    def processQuestion(self, user_question, dataframe):
        questions = dataframe["Question"].tolist()
        questions.append(user_question)
        TfidfVec = TfidfVectorizer(tokenizer=self.cleanText, stop_words='english')
        tfidf = TfidfVec.fit_transform(questions)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]
        print("IDX:", idx)
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]

        del questions[-1]

        if req_tfidf == 0:
            return "Sorry, I didn't get you"
        else:
            return dataframe.Answer[idx]

myObj = FAQBot()
app = FastAPI()

class QuestionRequest(BaseModel):
    nlp: dict

@app.post('/kba')
async def checkSAPFAQ(request: QuestionRequest):
    # data = request.dict()
    data = request.model_dump()
    if data:
        dataframe = myObj.getDataFrame("FAQs.xlsx", 'Catalogs')
        user_question = data['nlp']['source']
        answer = myObj.processQuestion(user_question, dataframe)
        resp = [{
            'type': 'text',
            'content': answer
        }]
        return {'replies': resp}
    else:
        return {'replies': [{
            'type': 'text',
            'content': 'Something went wrong, Please try again'
        }]}

@app.post('/errors') 
async def err(request: Request): 
    data = await request.json()
    print(data)
    return {'replies': [{
            'type': 'text',
            'content': 'Something went wrong, Please try again'
        }]}

if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)