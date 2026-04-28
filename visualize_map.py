import folium
import pandas as pd

def generate_map():
    # Load test results and original data to get coordinates
    results = pd.read_csv('results/test_results.csv')
    original = pd.read_csv('data/marine_waste_data.csv')
    
    # The test set is the last N rows of the original data
    test_coords = original.tail(len(results)).reset_index()
    
    m = folium.Map(location=[31.5, 34.5], zoom_start=8)
    
    for i in range(len(results)):
        val = results.iloc[i]['Pred']
        lat = test_coords.iloc[i]['latitude']
        lon = test_coords.iloc[i]['longitude']
        
        color = 'red' if val > 80 else 'green'
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=val/10,
            popup=f"Predicted Waste: {val:.1f} kg/km²",
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)
        
    m.save('results/map.html')
    print("Map generated at results/map.html")

if __name__ == "__main__":
    generate_map()

