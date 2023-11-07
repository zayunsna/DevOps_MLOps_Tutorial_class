from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

@app.route('/hello')
def hello():
    return 'Hello World!'

@app.route('/echo', methods=['POST'])
def echo():
    parms = request.get_json()

    if not parms:
        return jsonify({'error' : 'No data provided'}), 400
    
    return jsonify(parms)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'data' not in request.files:
        return jsonify({'error':'File not Founds'}), 400
    data = request.files['data']
    
    image = Image.open(data)
    width = image.width
    height = image.height

    return jsonify({'width':width, 'height':height})


if __name__ == '__main__':
    app.run()