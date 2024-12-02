import numpy as np
from pykrige.uk import UniversalKriging
import pandas as pd
from scipy.spatial import cKDTree
import time

# Load and preprocess the dataset
data = pd.read_csv('E:\Intern-2024\kriging\sample_data_1year.csv')
station_data = data[['Station_ID', 'Longitude', 'Latitude', 'Elevation']]
precipitation_data = data.drop(columns=['Station_ID', 'Longitude', 'Latitude', 'Elevation'])

# Coordinates and elevations
longitudes = station_data['Longitude'].values
latitudes = station_data['Latitude'].values
elevations = station_data['Elevation'].values
station_ids = station_data['Station_ID'].values

# Load best monthly variogram model parameters (from previous analysis)
best_variogram_params = {
    'January': {'model': 'exponential', 'len_scale': 0.9315950163011789, 'variance': 0.1358851882835086, 'nugget': 0.013588518828350861},
    'February': {'model': 'exponential', 'len_scale': 0.9315950163011789, 'variance': 0.26746631204399635, 'nugget': 0.026746631204399636},
    'March': {'model': 'spherical', 'len_scale': 0.9315950163011789, 'variance': 0.4115012762195685, 'nugget': 0.041150127621956856},
    'April': {'model': 'exponential', 'len_scale': 0.9315950163011789, 'variance': 1.121783498580198, 'nugget': 0.1121783498580198},
    'May': {'model': 'spherical', 'len_scale': 0.9315950163011789, 'variance': 4.928337610912069, 'nugget': 0.49283376109120697},
    'June': {'model': 'spherical', 'len_scale': 0.9315950163011789, 'variance': 22.490436705027843, 'nugget': 2.2490436705027843},
    'July': {'model': 'spherical', 'len_scale': 0.9315950163011789, 'variance': 49.15443790095582, 'nugget': 4.9154437900955825},
    'August': {'model': 'spherical', 'len_scale': 0.9315950163011789, 'variance': 42.43829771848843, 'nugget': 4.243829771848843},
    'September': {'model': 'spherical', 'len_scale': 0.9315950163011789, 'variance': 18.524979187876454, 'nugget': 1.8524979187876456},
    'October': {'model': 'spherical', 'len_scale': 0.9315950163011789, 'variance': 1.0319094197193248, 'nugget': 0.10319094197193249},
    'November': {'model': 'spherical', 'len_scale': 0.9315950163011789, 'variance': 0.027317645488109204, 'nugget': 0.0027317645488109206},
    'December': {'model': 'exponential', 'len_scale': 0.9315950163011789, 'variance': 0.030786674050434185, 'nugget': 0.0030786674050434187},
}

# Function to perform kriging prediction
def kriging_prediction(loo_longitudes, loo_latitudes, loo_values, valid_longitudes, valid_latitudes, valid_elevations, month):
    params = best_variogram_params[month]
    model_name = params['model']
    len_scale = params['len_scale']
    variance = params['variance']
    nugget = params['nugget']
    
    kriging_model = UniversalKriging(
        loo_longitudes,
        loo_latitudes,
        loo_values,
        drift_terms=['specified'],
        variogram_model=model_name,
        variogram_parameters={'sill': variance, 'range': len_scale, 'nugget': nugget},
        specified_drift=[loo_elevations]
    )
    
    pred_value, _ = kriging_model.execute(
        'points',
        [valid_longitudes],
        [valid_latitudes],
        specified_drift_arrays=[np.array([valid_elevations])]
    )
    return pred_value[0]

# Initialize overall and station-specific error tracking
overall_errors = []
station_metrics = []

# Start processing
start_time = time.time()

# Loop through each day
for day in precipitation_data.columns:
    day_data = precipitation_data[day]

    # Extract the month abbreviation (e.g., 'Jan' from 'Jan_01_2018')
    month_abbr = day.split('_')[0]  # 'Jan' from 'Jan_01_2018'
    
    # Map the month abbreviation to the full month name (e.g., 'Jan' -> 'January')
    month_name = pd.to_datetime(month_abbr, format='%b').strftime('%B')  # 'January' from 'Jan'

    # Filter valid data
    valid_indices = ~day_data.isna()
    valid_longitudes = longitudes[valid_indices]
    valid_latitudes = latitudes[valid_indices]
    valid_elevations = elevations[valid_indices]
    valid_values = day_data[valid_indices].values
    valid_station_ids = station_ids[valid_indices]

    # KD-tree for fallback
    tree = cKDTree(np.vstack((valid_longitudes, valid_latitudes)).T)

    # Leave-One-Out Cross-Validation
    for i in range(len(valid_values)):
        # Leave out the i-th data point
        loo_longitudes = np.delete(valid_longitudes, i)
        loo_latitudes = np.delete(valid_latitudes, i)
        loo_elevations = np.delete(valid_elevations, i)
        loo_values = np.delete(valid_values, i)

        try:
            print(f"kriging started for {day}")
            pred_value = kriging_prediction(
                loo_longitudes, loo_latitudes, loo_values,
                valid_longitudes[i], valid_latitudes[i], valid_elevations[i], month_name
            )
        except Exception:
            # Nearest neighbor fallback
            distances, indices = tree.query([valid_longitudes[i], valid_latitudes[i]], k=6)
            neighbor_values = valid_values[indices]
            pred_value = np.mean(neighbor_values)

        # Calculate and store overall errors
        overall_errors.append(abs(pred_value - valid_values[i]))

        # Store station-specific metrics
        station_metrics.append({
            'Station_ID': valid_station_ids[i],
            'Prediction': pred_value,
            'Actual': valid_values[i],
            'Error': abs(pred_value - valid_values[i])
        })

# Compute overall error metrics
overall_mae = np.mean(overall_errors)
overall_rmse = np.sqrt(np.mean(np.square(overall_errors)))
print(f"Overall MAE: {overall_mae:.4f}, Overall RMSE: {overall_rmse:.4f}")

# Compute station-wise metrics
station_metrics_df = pd.DataFrame(station_metrics)
station_summary = station_metrics_df.groupby('Station_ID').agg(
    MAE=('Error', 'mean'),
    RMSE=('Error', lambda x: np.sqrt(np.mean(np.square(x))))
).reset_index()

# Save station-wise metrics to CSV
output_path = 'E:\\Intern-2024\\kriging\\ppt_1year_station_error_metrics.csv'
station_summary.to_csv(output_path, index=False)

# Print runtime
end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")
print(f"Station-wise error metrics saved to {output_path}")
