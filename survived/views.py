from django.shortcuts import render
from django.http import HttpResponse

import pickle
import numpy as np
import pandas as pd

# Create your views here.


def index(request):
    return render(request, 'survived/index.html')


def predict(request):
    # Load the model
    with open('data/model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    if model is None:
        return HttpResponse('there is no model')

    # Load the dataset
    df = pd.read_csv('data/train.csv')

    # Clean the dataset

    df['Sex'] = df['Sex'].map(dict(male=1.0, female=0.0))
    df.drop(columns=['PassengerId', 'Survived'], inplace=True)
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df = df.select_dtypes(exclude='object')

    # If method is POST
    if request.method == "POST":

        # Svaing request POST
        name = request.POST['name']
         
        # Create a new dataframe
        X = pd.DataFrame(
            dict(
            pclass = float(request.POST['pclass']),
            sex = 1.0 if request.POST['sex'] == 'male' else 0.0,
            age = float(request.POST['age']),
            sibsp = request.POST['sibsp'],
            parch = request.POST['parch'],
            fare = request.POST['fare'],
            C = 0.0,
            Q = 0.0,
            S = 1.0,
            Master = 1.0 if request.POST['title'] == 'Master' else 0.0,
            Miss = 1.0 if request.POST['title'] == 'Miss' else 0.0,
            Mr = 1.0 if request.POST['title'] == 'Mr' else 0.0,
            Mrs = 1.0 if request.POST['title'] == 'Mrs' else 0.0,
            Other = 1.0 if request.POST['title'] == 'Other' else 0.0
            ), index=[0]
        )
        
        pred = model.predict(X)
        return render(request, 'survived/predict.html', {
            "predict": pred[0],
            "name": name
        })

    
    


