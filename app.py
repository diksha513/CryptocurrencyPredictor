import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from crypto_prediction import Crypto_prediction
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    """
    Bitcoin.
    Ethereum.   
    Ripple XRP.
    Litecoin.
    NEO.x
    """
    dict_ = {'BTC':'Bitcoin','ETC':'Ethereum','XRP':'XRP','TRX':'Tron','LTC':'Litecoin'}
    output = []
    # prediction_text_dict = {'q':{'prediction': 1, 'aa':2} , 'w':{'prediction': 2, 'aa':78768}, 'e':{'prediction': 3, 'aa':48646}, 'r':{'prediction': 4, 'aa':532}}
    prediction_text_dict = {}

    for k,v in dict_.items():
        obj = Crypto_prediction(k)
        obj.get_prediction()
        d = pickle.load(open('model.pkl','rb'))

        if type(d) == str:
            prediction_text_dict[k] = {'name':dict_[k] +"(⚠)" ,'prediction': 0, 'today_s data': 0}
            continue

        model = d['regressor']  
        target_label = d['target_label']
        last_closing_pt = d['last_closing_pt']
        val = model.predict(target_label)
        prediction_text_dict[k] = {'name':dict_[k],'prediction': val[0][0], 'today_s data': last_closing_pt[0]}
    print("------------------")
    print(prediction_text_dict)
    print("------------------")
    return render_template('predict.html',prediction_text=prediction_text_dict)


if __name__ == "__main__":
    app.run(debug=True)
