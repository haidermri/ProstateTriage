import pandas as pd
import os
import glob
import pickle
import numpy as np
import SimpleITK as sitk
import yaml

def process_data(data_l):

    array_list = []
    for i in range(len(data_l)):
        array = np.zeros(13)
        ptid_starter = data_l[i][0].split('_')
        if ptid_starter[1].startswith('PICAI'):
            ptid_starter[1] = ptid_starter[1].removeprefix('PICAI')
        array[0] = int(''.join(ptid_starter[1:3])) # ptid
        array[1] = np.nan # Unused
        array[2] = int(data_l[i][2]) # Positive
        if data_l[i][3] == 'unknown': # age
            array[3] = np.nan
        else:
            array[3] = float(data_l[i][3])
        if data_l[i][4] == '': # psa
            array[4] = np.nan
        else:
            array[4] = float(data_l[i][4])
        if data_l[i][5] == '': # PSAd
            array[5] = np.nan
        else:
            array[5] = float(data_l[i][5])
        array[6] = np.nan # Unused
        array[7] = np.nan # Unused
        array[8] = np.nan # Unused
        if data_l[i][9] == 'unknown' or data_l[i][9] == '': # PosISUP
            array[9] = np.nan
        else:
            array[9] = float(data_l[i][9])
        if data_l[i][10] == 'nan' or data_l[i][10] == '': # ISUP
            array[10] = np.nan
        else:
            array[10] = float(data_l[i][10])
        if data_l[i][11] == 'nan' or data_l[i][11] == '': # PosPIRADS
            array[11] = np.nan
        else:
            array[11] = float(data_l[i][11])
        if data_l[i][12] == 'nan' or data_l[i][12] == '': # PIRADS
            array[12] = np.nan
        else:
            array[12] = float(data_l[i][12])
        array_list.append(array)
    np_array = np.stack(array_list)
    return np_array

def process_dataframe(path, spreadsheet): # if t1 then t1_* else tall_* edit 2024-JUL-10
    df = pd.read_excel(spreadsheet)
    labels = df["Label"].unique()


    for label in labels:
        label_df = df[df["Label"] == label]
        axt2_list = []
        b1600_list = []
        adc_list = []
        wg_list = []
        pz_list = []
        tz_list = []
        label_list = []
        data_list = []


        for index, row in label_df.iterrows():
            folder_path = os.path.join(path, row["folder"])
            axt2_file = os.path.join(folder_path, "axt2.nii.gz")
            b1600_file = os.path.join(folder_path, "b1600.nii.gz")
            adc_file = os.path.join(folder_path, "adc.nii.gz")

            if not os.path.exists(axt2_file) or not os.path.exists(b1600_file) or not os.path.exists(adc_file):
                raise ValueError("One or more required files are missing for patient: ", row["folder"])


            wg_file = glob.glob(os.path.join(folder_path, "p_axt2_*.nii.gz"))[0]
            label_file = glob.glob(os.path.join(folder_path, "t1_axt2_*.nii.gz"))[0]
            pz_file = glob.glob(os.path.join(folder_path, "pz_axt2_*.nii.gz"))[0]
            tz_file = glob.glob(os.path.join(folder_path, "tz_axt2_*.nii.gz"))[0]

            axt2_list.append(axt2_file)
            b1600_list.append(b1600_file)
            adc_list.append(adc_file)

            wg_list.append(wg_file)
            label_list.append(label_file)
            pz_list.append(pz_file)
            tz_list.append(tz_file)

            # Calculate PSAd from PSA and wg_file by loading the image and calculating the volume
            wg_img = sitk.ReadImage(wg_file)
            wg_array = sitk.GetArrayFromImage(wg_img)
            wg_spacing = wg_img.GetSpacing()
            wg_volume = np.sum(wg_array) * np.prod(wg_spacing) # in mm^3, need to convert to cc
            wg_volume = wg_volume / 1000
            psa = row["psa"]
            psad = psa / wg_volume

            # Write to spreadsheet
            df.at[index, "PSAd"] = psad

            positive = row["PosISUP"]
            if (positive == 'unknown' or positive == ''):
                if label == 'test':
                    raise ValueError("Positive ISUP is unknown for test set patient: ", row["folder"])
                else:
                    positive = row['PosPIRADS'] # Only use PIRADS in lieu of ISUP if ISUP is unknown for training

            data = [
                row["folder"],
                1, # Unused
                positive,
                row["age"],
                row["psa"],
                row["PSAd"],
                1, # Unused
                1, # Unused
                1, # Unused
                row["PosISUP"],
                row["ISUP"],
                row["PosPIRADS"],
                row["PIRADS"],
            ]
            data_list.append(data)
            
            # Data now in format [pt_id, Unused, Positive, age, psa, PSAd, Unused, Unused, Unused, PosISUP, ISUP, PosPIRADS, PIRADS]
            # Indices we care about: pt_id at 0, Positive at 2, PosISUP at 9, ISUP at 10, PosPIRADS at 11, PIRADS at 12

        data_list = process_data(data_list)

        pickle_file = f"{label}_set.pickle"
        print('Length of axt2_list: ', len(axt2_list))
        if len(axt2_list) != len(b1600_list) or len(axt2_list) != len(adc_list) or len(axt2_list) != len(wg_list) \
            or len(axt2_list) != len(label_list) or len(axt2_list) != len(data_list) or len(axt2_list) != len(pz_list) \
                or len(axt2_list) != len(tz_list):
            raise ValueError("Lengths of lists do not match.")
        with open(pickle_file, "wb") as f:
            pickle.dump((axt2_list, b1600_list, adc_list, wg_list, pz_list, tz_list, label_list, data_list), f)

    # Write modified df
    df.to_excel("prostate_triage.xlsx", index=False)

# Example usage:
if __name__ == "__main__":
    running_yaml = 'preprocess_paramdic.yaml'
    with open(running_yaml, 'r') as stream:
        paramdic = yaml.full_load(stream)
    
    process_dataframe(paramdic['coreg_dir'],paramdic['spreadsheet'])
