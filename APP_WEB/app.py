from flask import Flask, jsonify, render_template, request, json
import subprocess
import os
import pandas as pd
import base64
import shutil
from pathlib import Path
import requests

app = Flask(__name__)

curr_path = os.getcwd()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = ROOT_DIR.replace('\\', '/')
#CODE_DIR = ROOT_DIR + '/comprehensive-stockmarket-tool'

@app.route('/execute')
def predict_LSTM():
    os.chdir(ROOT_DIR)
    try:
        print(os.getcwd())
        # Execute your Python file using subprocess module
        # subprocess.run(['python', './predict_stock.py'], check=True)
        returnValue = subprocess.check_output(
        ['python', './predict_stock.py'])

        json_str = json.dumps(
            {'price': returnValue.decode('utf-8').replace("\\", "").replace("\r", "").replace("\n", "").split("tensor")[1].split("(")[1].split(")")[0]})
        formated = json.loads(json_str)
        # return jsonify(json_str)
        return formated
    except Exception as e:
        return f'Error executing Python file: {str(e)}'

if __name__ == '__main__':
    app.run()

