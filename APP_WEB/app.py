from flask import Flask, jsonify, render_template, request, json
from flask_cors import CORS
import subprocess
import os
import pandas as pd
import base64
import shutil
from pathlib import Path
import requests
from APP_FLASK.Predictor import Predictor

app = Flask(__name__)
predictor = Predictor()
CORS(app)

curr_path = os.getcwd()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = ROOT_DIR.replace('\\', '/')
#CODE_DIR = ROOT_DIR + '/APPWEB'

@app.route('/execute/<file>', methods=['GET'])
def predict_LSTM(file):
    symbol = file
    os.chdir(ROOT_DIR.split("APP_WEB")[0])
    print('asdasd: '+ ROOT_DIR.split("APP_WEB")[0])
    try:
        # Execute your Python file using subprocess module
        # subprocess.run(['python', './predict_stock.py'], check=True)
        # returnValue = subprocess.check_output(
        # ['python', './predict_stock.py'])

        createFile(symbol)
        returnValue = subprocess.check_output(
                "python predict_stock.py")

        json_str = json.dumps(
            {'price': returnValue.decode('utf-8').replace("\\", "").replace("\r", "").replace("\n", "").split("tensor")[1].split("(")[1].split(")")[0]})
        formated = json.loads(json_str)
        # return jsonify(json_str)
        return formated
    except Exception as e:
        return f'Error executing Python file: {str(e)}'


@app.route('/', methods=['GET'])
def index1():
    return render_template('chart.html')


def createFile(content):
    #symbolconfig.json {'symbol': 'GOOGLE'}
    FILE = 'symbolconfig.json'
    path = './configs/' + FILE
    
    # Create and write data to text file
    with open(path, 'w') as fp:
        fp.write('{"symbol":"' + content + '"}')

@app.route('/chart', methods=['GET','POST'])
def my_route():
    symbol = request.args.get('symbol', default = '*', type = str)
    windowsize = request.args.get('windowsize', default = 7, type = int) 
    outputsize = request.args.get('outputsize', default = 7, type = int)


    print(symbol,windowsize,outputsize) 
    response = app.response_class(
		response=json.dumps({'symbol': symbol, 'windowsize': windowsize, 'outputsize': outputsize}),
		status=200,
		mimetype='application/json'
	)
    return response

### 
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symbol = data['symbol']
    model_type = data['model_type']
    window_size = data['window_size']
    output_size = data['output_size']
    
    prediction = predictor.predict(symbol, model_type, window_size, output_size)
    
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(debug=True)


