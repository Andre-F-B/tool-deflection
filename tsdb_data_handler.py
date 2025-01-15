import os
import time
import pickle
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.integrate import cumulative_trapezoid
import pandas as pd
import numpy as np
import pyvista as pv
user = os.getlogin()

def read_data(directory, **kwargs):
    table = kwargs.get('table', ["cnc", "drive", "iepe", "prog", "tool", "wcs", "auxiliary"])

    try:
        if os.listdir(directory)[0].endswith("parquet"):
            try:
                if "cnc" in table:
                    cnc = pd.read_parquet(os.path.join(directory, "cnc.parquet"))
                    tsdb_data = cnc
            except:
                print("importing cnc data failed")
            try:
                if "drive" in table:
                    drive = pd.read_parquet(os.path.join(directory, "drive.parquet"))
                    tsdb_data = pd.merge_asof(tsdb_data, drive, on="Timestamp")
            except:
                print("importing drive data failed")
            try:
                if "prog" in table:
                    prog = pd.read_parquet(os.path.join(directory, "prog.parquet"))
                    tsdb_data = pd.merge_asof(tsdb_data, prog, on="Timestamp")
            except:
                print("importing prog data failed")
            try:
                if "tool" in table:
                    tool = pd.read_parquet(os.path.join(directory, "tool.parquet"))
                    tsdb_data = pd.merge_asof(tsdb_data, tool, on="Timestamp")
            except:
                print("importing tool data failed")
            try:
                if "wcs" in table:
                    wcs = pd.read_parquet(os.path.join(directory, "wcs.parquet"))
                    tsdb_data = pd.merge_asof(tsdb_data, wcs, on="Timestamp")
            except:
                print("importing wcs data failed")
            try:
                if "iepe" in table:
                    iepe = pd.read_parquet(os.path.join(directory, "iepe.parquet"))
                    if len(str(iepe.Timestamp.iloc[0])) == 13:
                        iepe = repair_timestamps(iepe).reset_index(drop=True)
                    iepe.ACC = iepe.ACC / np.exp2(24) * 100 #transform unit to g
                    iepe = vibration_velocity(iepe)

                    acc_vel_rms = iepe[["Timestamp", "acc_vel_rms"]][0::20]
                    acc_vel_rms["Timestamp"] = np.array(acc_vel_rms["Timestamp"]/1000, dtype='int64')
                    tsdb_data = pd.merge_asof(tsdb_data, acc_vel_rms, on="Timestamp")

                else:
                    iepe = pd.DataFrame([])
            except:
                iepe = pd.DataFrame([])
                print("importing iepe data failed")

            tsdb_data = tsdb_data.dropna()
            tsdb_data.reset_index(drop=True, inplace=True)

    except:
        print("No data could be found")
        return pd.DataFrame([]), pd.DataFrame([])

    try:
        if "cnc" and "wcs" in table:
            tsdb_data = generate_tcp(tsdb_data)
    except:
        print("TCP variables could not be generated")

    try:
        if "drive" and "wcs" in table:
            tsdb_data = convert_units(tsdb_data)
            tsdb_data = kraftmodell(tsdb_data)
    except:
        print("Force variables could not be generated")

    print("Data has been imported from: " + directory)
    return tsdb_data.reset_index(drop=True), iepe.reset_index(drop=True)

def generate_tcp(data):
    # Perform the coordinate transformation

    x = data['XActPos'] - (data['X']) * 10000
    y = data['YActPos'] - (data['Y']) * 10000
    z = data['ZActPos'] - (data['Z']) * 10000

    # x = data['XActPos'] - (data['X'] + data['RCSX']) * 10000
    # y = data['YActPos'] - (data['Y'] + data['RCSY']) * 10000
    # z = data['ZActPos'] - (data['Z'] + data['RCSZ']) * 10000

    data['TCPX'] = x * np.cos(np.deg2rad(data['gamma'])) - y * np.sin(np.deg2rad(data['gamma']))
    data['TCPY'] = x * np.sin(np.deg2rad(data['gamma'])) + y * np.cos(np.deg2rad(data['gamma']))
    data['TCPZ'] = z

    return data


def cnc_to_vnck(data):
    vnck = pd.DataFrame(
        {
            'Timestamp': data.Timestamp,
            'S1Actrev': data.S1Actrev,
            'Actfeed': data.Actfeed,
            'XActPos': data.TCPX,
            'YActPos': data.TCPY,
            'ZActPos': data.TCPZ,
            'ToolLength': data.Length,
            'ToolDiameter': data.Diameter,
            'ToolID': data.Number.astype('int')
        }
    )

    return vnck

def plot_tcp(data, show_wkp=False, scalar=None, cmap_lim=[], title="", save=""):
    if scalar is None:
        scalar = data["S1ActTrq"]

    plotter = pv.Plotter()

    # tcp = pv.PolyData(data[["XActPos", "YActPos", "ZActPos"]].values / 10000)
    tcp = pv.PolyData(data[["TCPX", "TCPY", "TCPZ"]].values/10000)

    # Create PyVista colormap
    cmap = "viridis"
    if len(cmap_lim) == 0:
        cmap_lim = [scalar.min(), scalar.max()]

    # Add 3D point cloud
    plotter.show_grid()
    plotter.add_points(tcp, scalars=scalar, cmap=cmap, clim=cmap_lim)

    if show_wkp:
        if os.path.isfile('wkp.stl'):
            wkp = pv.read('wkp.stl')
            # wkp = wkp.translate([data['MaxEdgeX'].values[-1],
            #                      data['MinEdgeY'].values[-1],
            #                      data['MaxEdgeZ'].values[-1]])
            wkp = wkp.translate([data['RCSX'].values[-1],
                                 data['RCSY'].values[-1],
                                 data['RCSZ'].values[-1]])
            plotter.add_mesh(wkp, opacity=0.5)
        else:
            wkp = generate_wkp(data)
            # wkp = pv.Box(bounds=(wkp['init_x'] - data.RCSX.values[-1], wkp['end_x'] - data.RCSX.values[-1],
            #                      wkp['init_y'] - data.RCSY.values[-1], wkp['end_y'] - data.RCSY.values[-1],
            #                      wkp['init_z'] - data.RCSZ.values[-1], wkp['end_z'] - data.RCSZ.values[-1]))
            wkp = pv.Box(bounds=(wkp['init_x'], wkp['end_x'],
                                 wkp['init_y'], wkp['end_y'],
                                 wkp['init_z'], wkp['end_z']))

            plotter.add_mesh(wkp, opacity=0.5)

    # points = np.array([[0, 0, 0]])
    # point_cloud = pv.PolyData(points)
    # plotter.add_mesh(point_cloud, color='red', point_size=10.0, render_points_as_spheres=True)

    plotter.add_text(title, position='upper_edge', font_size=14, color='black')
    plotter.view_xy()
    plotter.camera.zoom(0.8)
    if len(save) > 0:
        plotter.save_graphic(f"{save}.svg")
    plotter.show(full_screen=False)


def generate_wkp(data):
    wkp = {
        "init_x": int(data['MinEdgeX'].values[-1]),
        "init_y": int(data['MinEdgeY'].values[-1]),
        "init_z": int(data['MinEdgeZ'].values[-1]),
        "end_x": int(data['MaxEdgeX'].values[-1]),
        "end_y": int(data['MaxEdgeY'].values[-1]),
        "end_z": int(data['MaxEdgeZ'].values[-1])
    }

    return wkp


def group_by(df, column="Operation", outputtype=1):
    """
    :param df: dataframe
    :param column: column name used for grouping
    :param outputtype: 1: list, 2: dict
    :return: grouped dataframes
    """

    df['group'] = df[column].ne(df[column].shift()).cumsum()
    df = df.groupby('group')

    if outputtype == 1:
        dfs = []

    if outputtype == 2:
        dfs = {}

    for name, data_ in df:
        if "ewind" in data_['Name'].values[0]:
            data_ = data_.replace([data_['category'].values[0]], 'Tap')

        if outputtype == 1:
            dfs.append({'group':data_[column].unique()[0], 'data':data_})

        if outputtype == 2:
            dfs[data_[column].unique()[0]] = data_

    return dfs


def repair_timestamps(df):
    sample_time = 50
    num_samples = len(df)
    df.Timestamp *= 1000
    start_timestamp = df.Timestamp.iloc[0]
    new_timestamps = np.arange(start_timestamp, start_timestamp + num_samples * sample_time, sample_time)
    df['Timestamp'] = new_timestamps
    return df


def downsample_rms(df):
    group_size = 20
    # Extract the 'ACC' series
    acc_series = df['ACC']

    # Ensure the length of the series is a multiple of group_size
    trimmed_length = (len(acc_series) // group_size) * group_size
    trimmed_series = acc_series.values[:trimmed_length]

    # Reshape the series to a 2D array where each row contains 'group_size' values
    reshaped_series = trimmed_series.reshape(-1, group_size)

    # Calculate the RMS for each group of consecutive values
    rms_values = np.sqrt(np.mean(reshaped_series ** 2, axis=1))

    # Select the corresponding timestamps
    timestamps = df['Timestamp'].values[group_size - 1:trimmed_length:group_size]

    # Create a new DataFrame for the RMS values with the corresponding timestamps
    rms_df = pd.DataFrame({'Timestamp': timestamps, 'ACC_rms': rms_values})
    rms_df.Timestamp = (rms_df.Timestamp/1000).astype('int64')
    return rms_df


def vibration_velocity(df):
    acc_rate = 20000
    f_acc_HP = 40
    f_acc_TP = 5000
    order_acc_filter = 6
    window_size_rms = 20

    acc = df['ACC'].to_numpy()

    butter_filt = signal.butter(order_acc_filter, (f_acc_HP, f_acc_TP), btype='bandpass', fs=acc_rate,
                           output='sos')  # ordnung des filters, untere- und obere- eckfrequenz
    acc_filtered = signal.sosfilt(butter_filt, acc)  # second order sections, digitaler filter

    acc_filtered = acc_filtered * 9806.65 #transform from g into mm/s^2

    acc_vel = cumulative_trapezoid(acc_filtered, dx=1/acc_rate, initial=0)
    df['acc_vel'] = acc_vel
    df['acc_vel_rms'] = (df['acc_vel']**2).rolling(window_size_rms).mean().apply(np.sqrt)
        
    return df

def lp_filter(y, fs=1000, cutoff=50, order=4):
    # Butterworth low-pass filter design
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize the frequency

    # Get the filter coefficients
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y_filt = signal.filtfilt(b, a, y)

    return y_filt

def convert_units(df):
    df["XActTrq"] = df["XActTrq"] / 1000 * 5.03
    df["XTrqColMon"] = df["XTrqColMon"] / 1000 * 5.03
    df["YActTrq"] = df["YActTrq"] / 1000 * 5.52
    df["YTrqColMon"] = df["YTrqColMon"] / 1000 * 5.52
    df["Y2ActTrq"] = df["Y2ActTrq"] / 1000 * 5.52
    df["Y2TrqColMon"] = df["Y2TrqColMon"] / 1000 * 5.52
    df["ZActTrq"] = df["ZActTrq"] / 1000 * 1.69
    df["ZTrqColMon"] = df["ZTrqColMon"] / 1000 * 1.69
    df["S1ActTrq"] = df["S1ActTrq"] / 1000 * 3.8

    df["S1Actrev"] = df["S1Actrev"] * (0.001/360) * 60
    df["S1Cmdrev"] = df["S1Cmdrev"] * (0.001/360) * 60
    df["S1Currrev"] = df["S1Currrev"] * (0.001/360) * 60

    return df


def kraftmodell(df):
    # transfer torque values to force values with unit N
    df["Fx"] = df["XTrqColMon"] * (2 * np.pi / 7E-3 * 0.92)
    df["Fy"] = df["YTrqColMon"] * (2 * np.pi / 7E-3 * 0.92)
    df["Fy2"] = df["Y2TrqColMon"] * (2 * np.pi / 7E-3 * 0.92)
    df["Fz"] = df["ZTrqColMon"] * (2 * np.pi / 4.17E-3 * 0.92)
    df["Fs"] = df["S1ActTrq"] / (df.Diameter / 2) * 1000

    return df


def sp_friction_model(df, inertia=True):
    sp_friction_model = pickle.load(
        open(rf"C:\Users\{user}\iCloudDrive\Documents_PTW\01_AICoM\00_Python\models\sp_friction_model", 'rb'))

    S1ActTrq_friction = sp_friction_model.predict(df["S1Actrev"].values.reshape(-1, 1))

    S1Actacc = df["S1Actrev"].diff()

    S1ActTrq_inertia = S1Actacc * 0.047

    if inertia:
        return S1ActTrq_friction + S1ActTrq_inertia
    else:
        return S1ActTrq_friction


def group_by_category(df, columnname):
    groups = df.groupby(columnname)

    dfs = []
    i = 0
    for name, group in groups:
        i += 1
        group.reset_index(inplace=True)
        dfs.append(group)
        print(f"Group {i}/{len(groups)}: {name}")

    return dfs


def group_by_segment(df, columnname="Timestamp", threshold=1):
    df = df.copy()
    timestamp_diff = df[columnname].diff()
    new_segment = abs(timestamp_diff) > threshold
    # df.loc[:, 'segment'] = new_segment.cumsum()
    df['segment'] = new_segment.cumsum()
    groups = df.groupby('segment')

    dfs = []
    i = 0
    for name, group in groups:
        i += 1
        group.reset_index(inplace=True)
        dfs.append(group)
        print(f"Group {i}/{len(groups)}: {name}")

    return dfs

# Create sequences
def create_sequences(X, Y, T, n_steps):
    seq_x, seq_y = [], []
    seq_t = []
    for i in range(len(X)):
        end_ix = i + n_steps
        if end_ix > len(X)-1:
            break
        seq_x.append(X[i:end_ix, :])
        seq_y.append(Y[end_ix, :])
        seq_t.append(T[end_ix, :])

    return np.array(seq_x), np.array(seq_y), np.array(seq_t).astype('uint16')