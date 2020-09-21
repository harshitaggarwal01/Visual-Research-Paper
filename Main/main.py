import os
from flask import Flask, request, render_template
import requests
from summarizer import Summarizer
import lxml.etree as ET
from xml.dom import minidom
import re
import pprint
from pyvis.network import Network
from operator import itemgetter 

####SUMMARIZER
def bert_summarize(document):
    model=Summarizer()

    # doc=document.read()

    summary=model(document,min_length=10,ratio=0.5)

    return summary


####XML TO TEXT
def xml2text(file_path):
    tree = ET.parse(file_path)   # import xml from flask
    root = tree.getroot()

    headings_seq = []
    headings_name = []
    headings_para = []

    headings_seq.append({'n' : '0 '})
    headings_name.append("Abstract")

    for item in root.findall("./{http://www.tei-c.org/ns/1.0}text/{http://www.tei-c.org/ns/1.0}body/{http://www.tei-c.org/ns/1.0}div/{http://www.tei-c.org/ns/1.0}head"):
        headings_seq.append(item.attrib)
        headings_name.append(item.text)

    mydoc = minidom.parse(file_path)
    divs = mydoc.getElementsByTagName("div")
    for div in divs: #<div>
        if(div.parentNode.nodeName == 'abstract'):
            for elem in div.childNodes:
                if(elem.nodeName == 'p'):
                    headings_para.append({"Abstract": elem.firstChild.data})

        if(div.parentNode.nodeName == "body"):
            try:
                x = 'works'
            except AttributeError:
                continue
            else:
                for elem in div.childNodes:
                    if(not elem.nodeName =='formula'):
                        if(elem.nodeName == 'head'):
                            section = elem.firstChild.data
                            continue
                        if(elem.nodeName == 'p'):
                            headings_para.append({section : bert_summarize(elem.firstChild.data)})
    # pp = pprint.PrettyPrinter(width=41, depth = 5, compact=True)
    # pp.pprint(headings_seq)
    # pp.pprint(headings_name)
    # pp.pprint(headings_para)
    create_MindMap(headings_name, headings_para)



#### MINDMAP GENERATOR

def create_MindMap(headings_name, headings_para): #list of headings and list of dicts
    net = Network("The Visual Research Paper")
    root = "Your Visual Research Paper"
    #root_id = 1
    net.add_node(root, shape = "circle", value = 500000, title = root, color = 'ffb3bf', fill = "pink")
    #print(net.nodes)
    #id = 2
    nodes_list = []
    for i in headings_name:
        nodes = []
        #print(i)
        for item in headings_para:
            for key in item:
                if(i == key):
                    nodes.append(item[key])
        nodes_list.append(nodes)

    for node in headings_name:
        net.add_node(node, shape = "ellipse", value = 30000, title = node, label = node, color = 'b37d8b', fill = "red")
        net.add_edge(root, node, weight=.97, length = 3000, arrow = True)
        for x in nodes_list:
            for item in x:
                net.add_node(item, shape = "textbox", value = (len(node)*1000), title = item, color = 'ffecf1')
                net.add_edge(node, item, weight=.87, length = 5000, arrow = True)

print(net.nodes)
net.toggle_physics(True)
net.show_buttons(filter_=['physics'])
net.show("mindmap.html")

####FLASK

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
        xml2text(save_path_file)
    else:
        return "Error uploading file!"

@app.route("/")
def index():
    return render_template("index.html");   


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)

