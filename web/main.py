from flask import Flask, render_template, request


app = Flask(__name__)
app.secret_key = 'development_key'


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    pass


if __name__ == '__main__':
    app.debug = True
    app.run(port=8080, host='127.0.0.1')
