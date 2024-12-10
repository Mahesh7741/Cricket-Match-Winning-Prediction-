from flask import Flask, render_template, request, jsonify
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from backend.model import IPLScorePredictor

app = Flask(__name__)

# Initialize the predictor
predictor = IPLScorePredictor('data/ipl_data.csv')

@app.route('/')
def index():
    # Get unique values for dropdowns
    unique_values = predictor.get_unique_values()
    return render_template('index.html', 
                           venues=unique_values['venues'],
                           bat_teams=unique_values['bat_teams'],
                           bowl_teams=unique_values['bowl_teams'],
                           batsmen=unique_values['batsmen'],
                           bowlers=unique_values['bowlers'])

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    venue = request.form['venue']
    bat_team = request.form['bat_team']
    bowl_team = request.form['bowl_team']
    batsman = request.form['batsman']
    bowler = request.form['bowler']
    
    # Make prediction
    predicted_score = predictor.predict_score(venue, bat_team, bowl_team, batsman, bowler)
    
    return jsonify({'predicted_score': predicted_score})

if __name__ == '__main__':
    app.run(debug=True)
