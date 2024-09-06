import os
from collections import OrderedDict
import numpy as np
import shutil
import matplotlib.pyplot as plt
import re
import pandas as pd
from scipy.interpolate import interp1d


# Function to check if a string starts with a number
def starts_with_number(input_string):
    pattern = r'^-?\d'  # Regular expression pattern for a leading digit (including negative sign)
    return re.match(pattern, input_string) is not None


def unique(data):
    data[data == "-0.000"] = "0.000"
    data_list = [tuple(row) for row in data]
    unique_data_list = list(OrderedDict.fromkeys(data_list))
    unique_data = np.array(unique_data_list, dtype=object)
    return unique_data


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    folder_path = './path/slot_sim'
    target_path = './path/processed'

    theta90_file_names = [filename for filename in os.listdir(folder_path) if "Theta90" in filename]
    phi0_file_names = [filename.replace("Theta90", "Phi0") for filename in theta90_file_names]
    phi90_file_names = [filename.replace("Theta90", "Phi90") for filename in theta90_file_names]

    for (theta90_file, phi0_file) in list(zip(theta90_file_names, phi0_file_names)):
        theta90_list = unique(np.array(pd.read_csv(os.path.join(folder_path, theta90_file), header=None))[1:])
        phi0_list = unique(np.array(pd.read_csv(os.path.join(folder_path, phi0_file), header=None))[1:])

        for i in range(max(len(theta90_list), len(phi0_list))):
            if i < len(theta90_list):
                theta90_list[i, 0], theta90_list[i, 1] = float(theta90_list[i, 0]), float(
                    theta90_list[i, 1].split('/')[0])
            if i < len(phi0_list):
                phi0_list[i, 0], phi0_list[i, 1] = float(phi0_list[i, 0]), float(phi0_list[i, 1].split('/')[0])

        f = interp1d(theta90_list[:, 0].astype(float), theta90_list[:, 1].astype(float), kind='cubic')
        theta90_angle = np.linspace(0, 360, 361)
        theta90_rho = f(theta90_angle)
        theta90_list = np.column_stack((theta90_angle, theta90_rho))

        f = interp1d(phi0_list[:, 0].astype(float), phi0_list[:, 1].astype(float), kind='cubic')
        phi0_angle = np.linspace(-180, 180, 361)
        phi0_rho = f(phi0_angle)
        phi0_list = np.column_stack((phi0_angle, phi0_rho))

        np.savetxt(os.path.join(target_path, 'az', theta90_file), theta90_list, delimiter=',', header='')
        np.savetxt(os.path.join(target_path, 'vt', phi0_file), phi0_list, delimiter=',', header='')


    # for file_name in file_names:
    #     if file_name.endswith('.MSI_'):
    #         file_path = os.path.join(folder_path, file_name)
    #         destination_file = os.path.join(destination_path, file_name)
    #         if os.path.exists(destination_file):
    #             os.remove(destination_file)
    #         shutil.move(file_path, destination_path)

    # for file_name in file_names:
    #     file_path = os.path.join(folder_path, file_name)
    #     with open(file_path, 'r') as f:
    #         file_str = f.read()
    #         contents = file_str.split('\n')
    #     contents = [float(x) for x in contents if x is not ""]
    #     len_gain = int(len(contents) / 2)
    #     if len(contents) % 2 != 0:
    #         continue
    #     if len_gain == 0:
    #         continue
    #     az_gain = np.array(contents[:len_gain])
    #     vt_gain = np.array(contents[len_gain:])
    #     # gain = gain - np.min(gain)
    #     theta = np.linspace(0, 360 - 360 / len_gain, len_gain)  # * 2 * np.pi / 360
    #     first_name = file_name.split('.')[0]
    #     stacked_az = np.column_stack((theta, az_gain))
    #     stacked_vt = np.column_stack((theta, vt_gain))
    #     np.savetxt(os.path.join(save_path, 'az', first_name + '.csv'), stacked_az, delimiter=',', header='')
    #     np.savetxt(os.path.join(save_path, 'vt', first_name + '.csv'), stacked_vt, delimiter=',', header='')

    # for file_name in file_names:
    #     file_path = os.path.join(folder_path, file_name)
    #     phi, az_gain, theta, vt_gain = [], [], [], []
    #     az, vt = False, False
    #     if file_name in ['_TIA-RAKARG_16_863-870_MHZ_8DBI (1).ADF_', '_BN_5_ANTENNA.ADF_', '_TARANA_BN_3GHZ.ADF_',
    #                      '_TIA-RAKARG_13_863___870_MHZ_5.8DBI.ADF_', '_TIA-RAKARG_16_863-870_MHZ_8DBI.ADF_',
    #                      '_TIA-RAKARG_19_902___930_MHZ_5.0DBI.ADF_', '_TIA-RAKARJ_15_863___870_MHZ_2.8DBI.ADF_']:
    #         continue
    #     print(file_name)
    #     with open(file_path, 'r', encoding='cp1252') as f:
    #         for line in f:
    #             if starts_with_number(line) and len(az_gain) == 0:
    #                 az = True
    #             elif not starts_with_number(line):
    #                 az, vt = False, False
    #             elif starts_with_number(line) and len(az_gain) != 0 and not az:
    #                 vt = True
    #
    #             if az:
    #                 line = line.replace('\n', '')
    #                 info = re.split(r'[ ,\t]+', line)
    #                 info = [s for s in info if s]
    #                 angle, amp = info[0], info[1]
    #                 phi.append(float(angle))
    #                 az_gain.append(float(amp))
    #             elif vt:
    #                 line = line.replace('\n', '')
    #                 info = re.split(r'[ ,\t]+', line)
    #                 info = [s for s in info if s]
    #                 angle, amp = info[0], info[1]
    #                 theta.append(float(angle))
    #                 vt_gain.append(float(amp))
    #
    #     stacked_az = np.column_stack((phi, az_gain))
    #     stacked_vt = np.column_stack((theta, vt_gain))
    #     first_name = file_name.split('.')[0]
    #     np.savetxt(os.path.join(save_path, 'az', first_name + '.csv'), stacked_az, delimiter=',', header='')
    #     np.savetxt(os.path.join(save_path, 'vt', first_name + '.csv'), stacked_vt, delimiter=',', header='')
