from lines_extraction import *
from models_multi import *
from utils import pred, reshape
from filter import Filter
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, jsonify, make_response, abort
from flask import request
from werkzeug.utils import secure_filename
from scipy import ndimage
import cv2
import sys
import codecs
import time
import datetime
import numpy as np 
import os
import io
import matplotlib.pyplot as plt

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

M = CRNN(output_dim=226)
M.model.load_weights('crnn_multi.h5')

def predict(im, threshold):
    section = divider(im, True, threshold)
    result = ""
    for sec in section:
        batch = [reshape(sec[line[0]:line[1]+1, :]) for line in extract_lines(sec) \
                                                   if reshape(sec[line[0]:line[1]+1, :]) is not None]
        pad_lines = pad_sequences(batch, padding='post', value=255.0)
        result += pred(pad_lines, M, None, print_screen= False, return_text = True) + " "

    return result



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods =['POST'])
def upload():
    file = request.files['file']
    threshold = request.files['threshold']
    in_memory_file = io.BytesIO()
    threshold.save(in_memory_file)
    threshold = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    threshold = int(''.join([unichr(x) for x in threshold]))
    if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return jsonify({'text': predict(img, threshold)}), 200

if __name__ == "__main__":
    app.run(port=1234)

# tasks = [
#     {
#         'id': 1,
#         'title': u'Buy groceries',
#         'description': u'Milk, Cheese, Pizza, Fruit, Tylenol', 
#         'done': False
#     },
#     {
#         'id': 2,
#         'title': u'Learn Python',
#         'description': u'Need to find a good Python tutorial on the web', 
#         'done': False
#     }
# ]

# def make_public_task(task):
#     new_task = {}
#     for field in task:
#         if field == 'id':
#             new_task['uri'] = url_for('get_task', task_id=task['id'], _external=True)
#         else:
#             new_task[field] = task[field]
#     return new_task

# @app.route('/todo/api/v1.0/tasks', methods = ['GET'])
# def get_tasks():
#     return jsonify({'tasks': [make_public_task(task) for task in tasks]})

# @app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
# def get_task(task_id):
#     task = [task for task in tasks if task['id'] == task_id]
#     if len(task) == 0:
#         abort(404)
#     return jsonify({'task': task[0]})

# @app.errorhandler(404)
# def not_found(error):
#     return make_response(jsonify({'error': 'Not found'}), 404)

# @app.route('/todo/api/v1.0/tasks', methods=['POST'])
# def create_task():
#     if not request.json or not 'title' in request.json:
#         abort(400)
#     task = {
#         'id': tasks[-1]['id'] + 1,
#         'title': request.json['title'],
#         'description': request.json.get('description', ""),
#         'done': False
#     }
#     tasks.append(task)
#     return jsonify({'task': task}), 201

# @app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['PUT'])
# def update_task(task_id):
#     task = [task for task in tasks if task['id'] == task_id]
#     if len(task) == 0:
#         abort(404)
#     if not request.json:
#         abort(400)
#     if 'title' in request.json and type(request.json['title']) != unicode:
#         abort(400)
#     if 'description' in request.json and type(request.json['description']) is not unicode:
#         abort(400)
#     if 'done' in request.json and type(request.json['done']) is not bool:
#         abort(400)
#     task[0]['title'] = request.json.get('title', task[0]['title'])
#     task[0]['description'] = request.json.get('description', task[0]['description'])
#     task[0]['done'] = request.json.get('done', task[0]['done'])
#     return jsonify({'task': task[0]})

# @app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['DELETE'])
# def delete_task(task_id):
#     task = [task for task in tasks if task['id'] == task_id]
#     if len(task) == 0:
#         abort(404)
#     tasks.remove(task[0])
#     return jsonify({'result': True})

# if __name__ == "__main__":
#     app.run(port=1234, debug=True)