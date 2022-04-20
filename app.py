from flask import Flask , request 
from flask_cors import CORS , cross_origin 
import joblib 
import numpy as np 
import re 

app = Flask(__name__)
CORS(app)

@app.route('/')
def hellowworld():
    return "hellow world";

@app.route('/area',methods=['GET'])
@cross_origin()
def are():
    w = float(request.values.get('w'))
    h = float(request.values.get('h'))
    return str(w*h)

@app.route('/iris',methods=['POST'])
@cross_origin()
def predict_species():
    model = joblib.load('D:/API_FastAPI/iris_model.pkl')
    re = request.values['param']
    input = np.array(re.split(','), dtype=np.float32).reshape(1,-1)
    predict_target = model.predict(input)
    if predict_target == 0:
        return "setosa"
    elif predict_target == 1:
        return "versicolor"
    else:
        return 'virginica'
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)