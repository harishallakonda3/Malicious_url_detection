from flask import Flask, render_template, request
from url import check # import the function that detects malicious URLs from another Python file

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/url.py', methods=['GET','POST'])
def detect():
    url = request.form['url'] # get the URL entered by the user in the HTML form
    result = check(url) # call the function that detects whether a URL is malicious or not
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=False)
