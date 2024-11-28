import pandas as pd
import numpy as np
from memory_profiler import profile 

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

DATA_FILES_PATH = (
    "case_flight_logs/case_circular_flight_actuator_armed_0.csv",
    "case_flight_logs/case_circular_flight_distance_sensor_0.csv",
    "case_flight_logs/case_circular_flight_vehicle_gps_position_0.csv",
    "case_flight_logs/case_circular_flight_vtol_vehicle_status_0.csv",
)


datasets = tuple(map(pd.read_csv, DATA_FILES_PATH))

# Find a time interval where all sensors have been activated
time_start, time_end, n = 0, np.inf, 0
for dataset in datasets:
    timestamps = np.array(dataset[TIMESTAMP])
    time_start = max(time_start, timestamps[0])
    time_end = min(time_end, timestamps[-1])
    n = max(n, len(dataset[TIMESTAMP]))

time = np.linspace(time_start, time_end, n)

dataset_actuators, dataset_distance_sensor, dataset_gps, dataset_vehicle_status = datasets

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

    if len(end_indicies) < start_indicies:
        if start_indicies[-1] == len(x) -1:
            start_indicies = start_indicies[:-1]
        else:
            end_indicies = np.append(end_indicies, np.array([len(x)-1]))

    slices = []

    for start_index, end_index in zip(start_indicies, end_indicies):
        slices.append(x[start_index:end_index])

    return slices


def wind_gaust(time, wind):
    delta_t = time[1] - time[0]
    delta_t_seconds = delta_t/1000.
    n = int(delta_t_seconds/WING_GAUST_DURATION)

    kernel = np.ones(n)/n
    wind_gaust = np.convolve(wind, kernel, mode="valid")

    largest_wind_gaust_index = np.argmax(wind_gaust)

    return wind_gaust[largest_wind_gaust_index], time[largest_wind_gaust_index]



rotary_wing_mode = np.rint(np.interp(time, dataset_vehicle_status[TIMESTAMP], dataset_vehicle_status[ROTARY_WING_MODE]))
transition_mode = np.rint(np.interp(time, dataset_vehicle_status[TIMESTAMP], dataset_vehicle_status[TRANSISTION_MODE]))
armed = np.rint(np.interp(time, dataset_actuators[TIMESTAMP], dataset_actuators[ARMED]))
fixed_wing_mode = np.array((rotary_wing_mode == 0) & (transition_mode == 0) & (armed == 1), dtype=np.int64)


velocity_north = np.interp(time, dataset_gps[TIMESTAMP], dataset_gps[VELOCITY_NORTH])
velocity_east = np.interp(time, dataset_gps[TIMESTAMP], dataset_gps[VELOCITY_EAST])
velocity_ground = np.sqrt(velocity_east**2 + velocity_east**2)

wind_magnitude = np.abs(velocity_ground-VELOCITY_AIR)
wind_magnitude_slices = slice_array(wind_magnitude, fixed_wing_mode)
time_slices = slice_array(time, fixed_wing_mode)


largest_wind_gaust_time = None
largest_wind_gaust_magnitude = 0

for wind_gaust_time_slice, wind_magnitude_slice in time_slices, wind_magnitude_slices:
    wind_gaust_magnitude, wind_gaust_time = wind_gaust(wind_gaust_time_slice, wind_gaust_magnitude)

    if wind_gaust_magnitude > largest_wind_gaust_magnitude:
        largest_wind_gaust_magnitude = wind_gaust_magnitude
        largest_wind_gaust_time = wind_gaust_time



#https://simulationbased.com/2020/06/02/smoothing-data-by-rolling-average-with-numpy/
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html