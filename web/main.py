from flask import Flask, render_template, request, redirect
from flask_wtf import FlaskForm
from wtforms import MultipleFileField, SubmitField
from wtforms.validators import DataRequired


app = Flask(__name__)
app.secret_key = 'development_key'


def check_pkl_files(file):
    if file.filename.split(".")[-1] == "pkl":
        return True
    return False


class Form(FlaskForm):
    file = MultipleFileField("File", validators=[DataRequired()])
    submit = SubmitField('Classify')


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = Form()
    if request.method == "POST":
        files = form.file.data
        for file in files:
            if check_pkl_files(file) is False:
                return redirect("/index")
        for file in files:
            file.save(f"static/save/{file.filename}")
    return render_template("block.html", form=form)


if __name__ == '__main__':
    app.debug = True
    app.run(port=8080, host='127.0.0.1')
