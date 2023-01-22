"""

Step 3 - Create the application that predicts heart disease percentage in the population of a town
based on the number of bikers and smokers. 

"""

#import necessary libraries
import numpy as np
from flask import Flask, request, render_template #type:ignore
import pickle

#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file)
model = pickle.load(open('models/model.pkl', 'rb'))

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')

#use the methods argument of the route() decorator to handle different HTTP methods.
@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]  
    prediction = model.predict(features)  
    
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Prediction Horsepower {}'.format(output))

if __name__ == "__main__":
    app.run()