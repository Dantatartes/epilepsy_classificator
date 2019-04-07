from flask import Flask, render_template, request, redirect, send_from_directory, session
from flask_wtf import FlaskForm
from wtforms import MultipleFileField, SubmitField
from wtforms.validators import DataRequired
from web.permission import check
import os

app = Flask(__name__, static_folder='static')
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
    if "flag" in session:
        os.remove(f"{os.getcwd()}/submission.pkl")
    session.pop("flag", 0)
    form = Form()
    if request.method == "POST":
        files = form.file.data
        for file in files:
            if check_pkl_files(file) is False:
                return redirect("/index")
        session["flag"] = True
        for file in files:
            file.save(f"static/{file.filename}")

        return redirect("/send_file")
    return render_template("block.html", form=form)


@app.route("/send_file", methods=['GET', 'POST'])
@check
def send_file():
    return send_from_directory(app.static_folder, 'submission.pkl', as_attachment=True)


if __name__ == '__main__':
    app.debug = False
    app.run(port=8080, host='127.0.0.1')
