import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from src.data_generator import generate_marine_data
from src.quantum_layer import quantum_feature_mapping
from src.lstm_model import build_quantum_hybrid_model
import os

def create_sequences(data, seq_len):
    xs, ys, q_feats = [], [], []
    # Index 3,4,5 are the environmental features
    features = data[:, 3:6] 
    target = data[:, 6]
    
    for i in range(len(data) - seq_len):
        window = features[i:(i + seq_len)]
        xs.append(window)
        ys.append(target[i + seq_len])
        # Average window for quantum input
        q_feats.append(np.mean(window, axis=0))
        
    return np.array(xs), np.array(ys), np.array(q_feats)

def run_training():
    df = generate_marine_data()
    scaler = MinMaxScaler()
    scaled_data = df.copy()
    cols_to_scale = ['current_speed', 'wind_speed', 'sea_temp', 'waste_kg_per_km2']
    scaled_data[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    seq_len = 7
    X, y, Q = create_sequences(scaled_data.values, seq_len)
    
    # Quantum mapping
    print("Computing Quantum Features...")
    Q_transformed = quantum_feature_mapping(Q)
    
    # Train/Test Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    Q_train, Q_test = Q_transformed[:split], Q_transformed[split:]
    
    model = build_quantum_hybrid_model(seq_len, 3)
    model.fit([X_train, Q_train], y_train, epochs=50, batch_size=16, verbose=1)
    
    # Predictions
    preds = model.predict([X_test, Q_test])
    
    # Rescale back for results
    # Dummy array to inverse transform
    dummy = np.zeros((len(preds), 4))
    dummy[:, 3] = preds.flatten()
    final_preds = scaler.inverse_transform(dummy)[:, 3]
    
    dummy_y = np.zeros((len(y_test), 4))
    dummy_y[:, 3] = y_test
    final_y = scaler.inverse_transform(dummy_y)[:, 3]
    
    # Save Results
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    model.save('models/quatide_model.h5')
    
    res_df = pd.DataFrame({'True': final_y, 'Pred': final_preds})
    res_df.to_csv('results/test_results.csv', index=False)
    
    plt.figure(figsize=(12,6))
    plt.plot(final_y, label='Actual Waste')
    plt.plot(final_preds, label='Hybrid Prediction')
    plt.legend()
    plt.title("Quatide Performance")
    plt.savefig('results/prediction_chart.png')
    print("Training complete. Results saved in /results.")

if __name__ == "__main__":
    run_training()

