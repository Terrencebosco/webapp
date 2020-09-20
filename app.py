import flask
import pickle
import pandas as pd
import category_encoders as ce
from joblib import load


column_lables = ['mileage',
 'ext_color',
 'city_mpg',
 'high_mpg',
 'engine',
 'car_year',
 'make',
 'size',
 'car_type']

app = flask.Flask(__name__, template_folder='templates')
model = load('model/pipeline.pkl')

@app.route('/')
def main():
    return(flask.render_template('main.html'))
                                    
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in flask.request.form.values()]
    final_features = pd.DataFrame([int_features], columns=column_lables)
    prediction = model.predict(final_features)
    
    output = prediction[0]
    return flask.render_template('main.html', prediction_text=output)
    
if __name__ == '__main__':
    app.run()