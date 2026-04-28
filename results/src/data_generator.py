import numpy as np
import pandas as pd
import os

def generate_marine_data(n_days=365, output_path="data/marine_waste_data.csv"):
    """
    Generates synthetic marine environmental data for the Eastern Mediterranean.
    """
    np.random.seed(42)
    os.makedirs("data", exist_ok=True)

    # Coordinates: East Med (31.5-32.5 N, 34.0-35.5 E)
    lats = np.random.uniform(31.5, 32.5, n_days)
    lons = np.random.uniform(34.0, 35.5, n_days)
    
    # Features with seasonal patterns
    time = np.linspace(0, 4 * np.pi, n_days)
    current_speed = 0.5 + 0.3 * np.sin(time) + np.random.normal(0, 0.05, n_days)
    wind_speed = 10 + 5 * np.cos(time) + np.random.normal(0, 2, n_days)
    sea_temp = 20 + 8 * np.sin(time/2) + np.random.normal(0, 0.5, n_days)

    # Target: Waste accumulation (influenced by wind and currents)
    # Base waste + current influence + wind influence + noise
    waste = (current_speed * 50) + (wind_speed * 2) + (sea_temp * 0.5) + np.random.normal(10, 5, n_days)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=n_days),
        'latitude': lats,
        'longitude': lons,
        'current_speed': current_speed,
        'wind_speed': wind_speed,
        'sea_temp': sea_temp,
        'waste_kg_per_km2': waste
    })

    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    return df

def load_real_data(filepath):
    """Loads CSV data for the Quatide pipeline."""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    generate_marine_data()
  
