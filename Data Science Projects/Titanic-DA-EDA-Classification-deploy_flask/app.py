from flask import Flask,render_template
import numpy as np
from requests import request
from sklearn.datasets import load_iris
import pickle
from flask import request

app=Flask(__name__,template_folder='templates')
loaded_model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")

#4route prediction
@app.route('/predict',methods=['POST'])
def predict():
    Pclass=request.form['Pclass']
    Sex=request.form['Sex']
    Age=request.form['Age']
    Family_Members=request.form['Family Members']
    Embarked=request.form['Embarked']
    Fare_Category=request.form['Fare Category']

    #take input from user form put in arry
    form_arry=np.array([[Pclass,Sex,Age,Family_Members,Embarked,Fare_Category]])

    #predict on arry of user input form
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    prediction=loaded_model.predict(np.array(form_arry))
    classes = ["deceased","survived"]
    result = classes[int(prediction)]

    return render_template("result.html",result=result)    

if __name__ == '__main__':
    app.run(debug=True)    