from flask import Flask,render_template,request
import numpy as np
import pickle
import joblib
app=Flask(__name__)
model = joblib.load('xgb2.pkl')

@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/predict',methods=['POST'])
def predict():
    
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    print(final_features)
    prediction = model.predict(final_features)

    output = prediction[0]
    if output==0:
        output="You are eligible for car policy"
    else:
        output="Your are not eligible for car ploicy "
    return render_template('index1.html',res=output)
if __name__ == "__main__":
    app.run(host="localhost", port=5000)




