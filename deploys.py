from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Loading the model
model = joblib.load('irisfsk.pkl')

# Labels for prediction results (I changed them to match zero-based indexing)
labels = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

# Defining the Home page
@app.route('/')
def home():
    return render_template('home.html')

# Defining the Prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Taking the inputs and saving them to a variable
            SepalLength = float(request.form['SepalLength'])
            SepalWidth  = float(request.form['SepalWidth'])
            PetalLength = float(request.form['PetalLength'])
            PetalWidth  = float(request.form['PetalWidth'])
            
            # Converting the inputs into a numpy array
            pred_args = np.array([SepalLength, SepalWidth, PetalLength, PetalWidth]).reshape(1, -1)
            
            # Predicting the Label
            model_prediction = model.predict(pred_args)[0]
            
            # Convert model_prediction to a regular Python int
            model_prediction = int(model_prediction)

            # Mapping the prediction to the flower name using the 'labels' dictionary
            flower_name = labels[model_prediction]

            return render_template('predict.html', prediction=flower_name)

        except Exception as e:
            # Handle errors like invalid input
            return f"Error: {str(e)}. Please ensure all inputs are valid numerical values."

    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)

