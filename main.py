import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
# constants
VELOCITY_AIR = 23 # m/s
WING_GAUST_DURATION = 3 # seconds

# csv keywords
TIMESTAMP = "timestamp"
ROTARY_WING_MODE = "vtol_in_rw_mode"
TRANSISTION_MODE = "vtol_in_trans_mode"
ARMED = "armed"
VELOCITY_NORTH = "vel_n_m_s"
VELOCITY_EAST = "vel_e_m_s"
LONGITUDE = "lon"
LATITUDE = "lat"


# DATA_FILES_PATH = (
#     "case_flight_logs/case_circular_flight_actuator_armed_0.csv",
#     "case_flight_logs/case_circular_flight_distance_sensor_0.csv",
#     "case_flight_logs/case_circular_flight_vehicle_gps_position_0.csv",
#     "case_flight_logs/case_circular_flight_vtol_vehicle_status_0.csv",
# )

DATA_FILES_PATH = (
    "case_flight_logs/case_realistic_flight_actuator_armed_0.csv",
    "case_flight_logs/case_realistic_flight_distance_sensor_0.csv",
    "case_flight_logs/case_realistic_flight_vehicle_gps_position_0.csv",
    "case_flight_logs/case_realistic_flight_vtol_vehicle_status_0.csv",
)


datasets = list(map(pd.read_csv, DATA_FILES_PATH))

# Find a time interval where all sensors have been activated
time_start, time_end, n = 0, np.inf, 0
for dataset in datasets:
    timestamps = np.array(dataset[TIMESTAMP])
    time_start = max(time_start, timestamps[0])
    time_end = min(time_end, timestamps[-1])
    n = max(n, len(dataset[TIMESTAMP]))

time = np.linspace(time_start, time_end, n)

def make_dataset_uniform_in_time(time, dataset):
    column_names = dataset.columns
    column_data = np.array([np.interp(time, dataset[TIMESTAMP], column).astype(column.dtype) for column_name, column in dataset.items()]).T
    dataframe = pd.DataFrame(data=column_data, columns=dataset.columns)
    dataframe[TIMESTAMP] = time
    return dataframe


def slice_array(x, filter):
    """
    All indicies where filter is false will be deleted from the array. The array
    will than be subdivided wherever data is deleted. The original input array will 
    not be affected.
    """
    filter = np.array(filter, dtype=np.int8)

    filter_difference = np.diff(filter)
    filter_difference = np.concatenate((np.array([filter[0]]), filter_difference))

    start_indicies = np.where(filter_difference == 1)[0]
    end_indicies = np.where(filter_difference == -1)[0]

    if len(end_indicies) < len(start_indicies):
        if start_indicies[-1] == len(x) -1:
            start_indicies = start_indicies[:-1]
        else:
            end_indicies = np.append(end_indicies, np.array([len(x)-1]))

    slices = []

    for start_index, end_index in zip(start_indicies, end_indicies):
        slices.append(x[start_index:end_index])

    return slices


def largest_wind_gaust(dataset):
    delta_t = dataset[TIMESTAMP][1] - dataset[TIMESTAMP][0]
    delta_t_seconds = delta_t/1_000_000.
    n = int(delta_t_seconds/WING_GAUST_DURATION)

    magnitude = lambda vector: np.sqrt(vector[:,0]**2 + vector[:,1]**2)

    velocity_ground = np.array([dataset[VELOCITY_EAST], dataset[VELOCITY_NORTH]])
    velocity_air = velocity_ground/magnitude(velocity_ground) * VELOCITY_AIR

    wind = velocity_ground - velocity_air
    wind_magnitude = magnitude(wind)


    fixed_wing_mode = ((dataset[ROTARY_WING_MODE] == 0) & (dataset[TRANSISTION_MODE] == 0) & (dataset[ARMED] == 1)).astype(dtype=np.int64)
    time_slices = slice_array(dataset[TIMESTAMP], fixed_wing_mode)
    wind_magnitude_slices = slice_array(dataset[TIMESTAMP], fixed_wing_mode)

    kernel = np.ones(n)/n

    largest_wind_gaust = 0
    largest_wind_gaust_time = None

    for time_slice, wind_magnitude_slice in zip(time_slices, wind_magnitude_slices):
        if len(wind_magnitude_slice) > n:
            wind_gaust = np.convolve(magnitude(wind), kernel, mode="valid")
            largest_wind_gaust_index = np.argmax(wind_gaust)
            if wind_gaust[largest_wind_gaust_index] > largest_wind_gaust:
                largest_wind_gaust = wind_gaust[largest_wind_gaust_index]
                largest_wind_gaust_time = time_slice[largest_wind_gaust_index]
    
    return largest_wind_gaust_time, largest_wind_gaust




dataset = pd.concat(list(map(lambda dataset:make_dataset_uniform_in_time(time, dataset), datasets)), axis=1)

fixed_wing_mode = ((dataset[ROTARY_WING_MODE] == 0) & (dataset[TRANSISTION_MODE] == 0) & (dataset[ARMED] == 1)).astype(dtype=np.int64)
dataset_sliced = pd.concat(slice_array(dataset, fixed_wing_mode))

magnitude = lambda vector: np.sqrt(vector[:,0]**2 + vector[:,1]**2)

velocity_ground = np.array([dataset_sliced[VELOCITY_EAST], dataset_sliced[VELOCITY_NORTH]]).T
velocity_air = (velocity_ground.T*(1/magnitude(velocity_ground))).T * VELOCITY_AIR

wind = velocity_ground - velocity_air



X = np.linspace(np.min(dataset[LONGITUDE]), np.max(dataset[LONGITUDE]), 100)
Y = np.linspace(np.min(dataset[LATITUDE]), np.max(dataset[LATITUDE]), 100)

mesh_x, mesh_y = np.meshgrid(X, Y)



mesh = np.dstack([mesh_x, mesh_y])

interpolated_wind_smoothed = RBFInterpolator(np.array([dataset_sliced[LONGITUDE], dataset_sliced[LATITUDE]]).T, wind, smoothing=3.)
wind_mesh_smoothed = np.array([interpolated_wind_smoothed(row) for row in mesh])

interpolated_wind = RBFInterpolator(np.array([dataset_sliced[LONGITUDE], dataset_sliced[LATITUDE]]).T, wind)
wind_mesh = np.array([interpolated_wind(row) for row in mesh])

standard_deviation_wind = np.sqrt((wind_mesh - wind_mesh_smoothed)**2)

