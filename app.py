from flask import Flask,request,render_template
import numpy as np
import sklearn
import pickle


app=Flask(__name__)
# Open the model file in read-binary mode and load it
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Open the scaling file in read-binary mode and load it
with open('scaling.pkl', 'rb') as scaling_file:
    scaling = pickle.load(scaling_file)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    N=request.form["nitrogen"]
    P=request.form["phosphorous"]
    K=request.form["potassium"]
    temp=request.form["temperature"]
    hum=request.form["humidity"]
    ph=request.form["soil-ph"]
    rain=request.form["rainfall"]

    feature=[N,P,K,temp,hum,ph,rain]
    feature1=np.array(feature).reshape(1,-1)
    t_feature=scaling.transform(feature1)

    prediction=model.predict(t_feature)

    pred_dict={
   1: 'rice',
   2: 'maize',
   3: 'chickpea',
   4: 'kidneybeans',
   5: 'pigeonpeas',
   6: 'mothbeans',
   7: 'mungbean',
   8: 'blackgram',
   9: 'lentil',
   10: 'pomegranate',
   11: 'banana',
   12: 'mango',
   13: 'grapes',
   14: 'watermelon',
   15: 'muskmelon',
   16: 'apple',
   17: 'orange',
   18: 'papaya',
   19: 'coconut',
   20: 'cotton',
   21: 'jute',
   22: 'coffee'
    }
    if prediction[0] in pred_dict:
        crop=pred_dict[prediction[0]]
        result="Suitable Crop: {}".format(crop)
    else:
        result="There is no crop"
    return render_template("index.html",result=result)

if __name__=="__main__":
    app.run(debug=True)


