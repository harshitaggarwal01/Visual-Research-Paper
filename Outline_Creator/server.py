from flask import Flask, request, render_template
import requests
import os

counter = 0
app = Flask(__name__)

@app.route('/handle_form', methods=['POST'])
def handle_form():
    print("Posted file: {}".format(request.files['file']))
    file = request.files['file']
    files = {"input": file.read()}
    r = requests.post("https://cloud.science-miner.com/grobid/api/processFulltextDocument", files=files)

    if r.ok:
        print(r.status_code)
        xml_str = (r.text)
        save_path_file = f'source{counter}.xml'
        with open(save_path_file, "w", encoding="utf-8") as f: 
            f.write(xml_str)  
        return f"XML file created: <{save_path_file}>"
    else:
        return "Error uploading file!"

@app.route("/")
def index():
    return render_template("index.html");   


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)