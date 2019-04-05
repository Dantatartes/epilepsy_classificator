from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired


app = Flask(__name__)
app.secret_key = 'development_key'


class Form(FlaskForm):
    file = FileField("", validators=[DataRequired()])
    submit = SubmitField('Classify')


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = Form()
    if request.method == "POST":
        file = form.file.data
        name = file.filename
        print(name)
        file.save(f"static/save/{name}")
    return render_template("block.html", form=form)


if __name__ == '__main__':
    app.debug = True
    app.run(port=8080, host='127.0.0.1')
