import os

from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    required_fields = (
        'gender',
        'ethnicity',
        'parental_level_of_education',
        'lunch',
        'test_preparation_course',
        'reading_score',
        'writing_score',
    )
    if not all(request.form.get(field) for field in required_fields):
        return render_template(
            'home.html', error='Please complete every field before predicting.'
        ), 400

    try:
        reading_score = float(request.form['reading_score'])
        writing_score = float(request.form['writing_score'])
        if not all(0 <= score <= 100 for score in (reading_score, writing_score)):
            raise ValueError

        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=reading_score,
            writing_score=writing_score,
        )
        pred_df=data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    except (TypeError, ValueError):
        return render_template(
            'home.html', error='Scores must be numbers between 0 and 100.'
        ), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=os.getenv('FLASK_DEBUG') == '1')
