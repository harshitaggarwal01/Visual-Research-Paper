import os
from flask import Flask, request, render_template
import requests

app = Flask(__name__)

@app.route('/handle_form', methods=['POST'])
def handle_form():
    print("Posted file: {}".format(request.files['file']))
    file = request.files['file']
    files = {"input": file.read()}
    r = requests.post("https://cloud.science-miner.com/grobid/api/processFulltextDocument", files=files)

    if r.ok:
        print(r.status_code)
        return r.text
        # return "File uploaded!"
    else:
        return "Error uploading file!"

@app.route("/")
def index():
    return render_template("index.html");   


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)