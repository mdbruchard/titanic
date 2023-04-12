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
        return HttpResponse('theres no model')

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
        sex = request.POST['sex']
        
        # If request post is male turn into 1 else 0
        if sex == 'male':
            sex = 1.0
        else:
            sex = 0.0
            
        # Create a new dataframe
        X = pd.DataFrame(
            dict(
            pclass = float(request.POST['pclass']),
            age = float(request.POST['age']),
            sex = sex,
            sibsp = float(request.POST['sibsp']),
            parch = float(request.POST['parch']),
            fare = float(request.POST['fare']),
            ), index=[0]
        )
        
        # for each colum in the dataset
        for col in df.columns:
            
            # save the max and min
            max_range = max(df[col])
            min_range = min(df[col])

            # col in lowercase
            col = col.lower()
           
            # normalize the new dataframe
            X[col] = (X[col] - min_range) / (max_range - min_range)
        
        # Turn into array and predcting 
        arr = np.array(X)
        print(arr)
        y_pred = model.predict(arr)
        pred = y_pred.tolist()

        print(pred)

        return render(request, 'survived/predict.html', {
            "predict": pred[0],
            "name": name
        })

    
    


