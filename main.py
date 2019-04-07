from flask import Flask, render_template, request, redirect, send_from_directory, session
from flask_wtf import FlaskForm
from wtforms import MultipleFileField, SubmitField
from wtforms.validators import DataRequired
from permission import check

from sklearn.neighbors import KNeighborsClassifier
from model import *
import pandas as pd


app = Flask(__name__, static_folder='static')
app.secret_key = 'development_key'


def check_pkl_files(file):
    if file.filename.split(".")[-1] == "pkl":
        return True
    return False


class Form(FlaskForm):
    file = MultipleFileField("File", validators=[DataRequired()])
    submit = SubmitField('Classify')

def process(files_list):
    model = Model(data_dir='./pp2', model=KNeighborsClassifier(n_neighbors=5), pytorch_=False, )
    model.train([f'seiz_{i}.pkl' for i in range(1, 1881)])
    prediction = model.predict_test(files_list, have_label=False)
    files_list = " ".join(files_list).replace(".pkl", "")
    files_list = files_list.split()
    final_df = pd.DataFrame({'id': files_list, 'label': prediction}, index=None)
    final_df.to_csv('./static/submission.csv', index=False)


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    session.pop("flag", 0)
    form = Form()
    if request.method == "POST":
        files = form.file.data
        for file in files:
            if check_pkl_files(file) is False:
                return redirect("/index")
        session["flag"] = True
        for file in files:
            file.save(f"pp2/{file.filename}")

        filenames = [x.filename for x in files]
        process(filenames)

        return redirect("/send_file")
    return render_template("block.html", form=form)


@app.route("/send_file", methods=['GET', 'POST'])
@check
def send_file():
    return send_from_directory(app.static_folder, 'submission.csv', as_attachment=True)


if __name__ == '__main__':
    app.debug = False
    app.run(port=8080, host='127.0.0.1')
