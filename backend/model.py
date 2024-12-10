import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import pandas as pd

class IPLScorePredictor:
    def __init__(self, data_path):
        # Load and preprocess data
        self.ipl = pd.read_csv(data_path)
        self.prepare_data()
        
    def prepare_data(self):
        # Drop unnecessary columns
        df = self.ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 
                             'wickets_last_5', 'mid', 'striker', 'non-striker'], axis=1)
        
        # Separate features and target
        X = df.drop(['total'], axis=1)
        y = df['total']
        
        # Encode categorical features
        self.venue_encoder = LabelEncoder()
        self.batting_team_encoder = LabelEncoder()
        self.bowling_team_encoder = LabelEncoder()
        self.batsman_encoder = LabelEncoder()
        self.bowler_encoder = LabelEncoder()
        
        X['venue'] = self.venue_encoder.fit_transform(X['venue'])
        X['bat_team'] = self.batting_team_encoder.fit_transform(X['bat_team'])
        X['bowl_team'] = self.bowling_team_encoder.fit_transform(X['bowl_team'])
        X['batsman'] = self.batsman_encoder.fit_transform(X['batsman'])
        X['bowler'] = self.bowler_encoder.fit_transform(X['bowler'])
        
        # Scale the data
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Build and train model
        self.model = self.build_model(X_scaled.shape[1])
        
    def build_model(self, input_shape):
        model = keras.Sequential([
            keras.layers.Input(shape=(input_shape,)),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        
        huber_loss = tf.keras.losses.Huber(delta=1.0)
        model.compile(optimizer='adam', loss=huber_loss)
        
        return model
    
    def predict_score(self, venue, bat_team, bowl_team, batsman, bowler):
        # Encode input values
        encoded_venue = self.venue_encoder.transform([venue])[0]
        encoded_bat_team = self.batting_team_encoder.transform([bat_team])[0]
        encoded_bowl_team = self.bowling_team_encoder.transform([bowl_team])[0]
        encoded_batsman = self.batsman_encoder.transform([batsman])[0]
        encoded_bowler = self.bowler_encoder.transform([bowler])[0]
        
        # Prepare input data
        input_data = np.array([encoded_venue, encoded_bat_team, encoded_bowl_team, 
                                encoded_batsman, encoded_bowler]).reshape(1, -1)
        input_data = self.scaler.transform(input_data)
        
        # Make prediction
        predicted_score = self.model.predict(input_data)
        return int(predicted_score[0, 0])
    
    def get_unique_values(self):
        return {
            'venues': self.ipl['venue'].unique().tolist(),
            'bat_teams': self.ipl['bat_team'].unique().tolist(),
            'bowl_teams': self.ipl['bowl_team'].unique().tolist(),
            'batsmen': self.ipl['batsman'].unique().tolist(),
            'bowlers': self.ipl['bowler'].unique().tolist()
        }