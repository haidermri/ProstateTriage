import pandas as pd
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
from monai.networks.nets import UNet
from monai.transforms.transform import MapTransform
from argparse import ArgumentParser
import socket
import torch
import glob
from monai import data, transforms
from report_guided_annotation import extract_lesion_candidates

def PCBG_calc(psa,age): # PCBG calculator assuming race = non-black, priorbiopsy=unsure, dre=unsure, famhx=unsure
    #https://github.com/ClevelandClinicQHS/riskcalc-website/blob/main/PBCG/R_code_PBCG_risk_calculator.R
    race = 0
    data = np.array([1,np.log2(psa),age,race])
    low = np.array([-2.23794923 , 0.05343098 , 0.01553627 , 0.69593716])
    high = np.array([-6.13292904 , 0.62979529 , 0.06002002 , 0.43816016])
    S1 = np.dot(low,data)
    S2 = np.dot(high,data)
    risk_no = 1/(1+np.exp(S1)+np.exp(S2))*100
    risk_low = np.exp(S1)/(1+np.exp(S1)+np.exp(S2))*100
    # risk_no = np.round(risk_no)
    # risk_low = np.round(risk_low)
    risk_high = 100.0-risk_no-risk_low
    return risk_no, risk_low, risk_high
    # return risk_low + risk_high # chance for PCa

def PCPT_calc(psa,age):
    #https://github.com/ClevelandClinicQHS/riskcalc-website/blob/main/PCPTRC/riskcalc.R
    psa = float(psa)
    age = float(age)
    race = 0
    data = np.array([1,np.log2(psa),age,race])
    low = np.array([-2.81814489, 0.24044370, 0.01370219, 0.12000825])
    high = np.array([-6.84249970, 0.70043815, 0.04574460, 1.01699029])
    S1 = np.dot(low,data)
    S2 = np.dot(high,data)
    risk_no = 1/(1+np.exp(S1)+np.exp(S2))*100
    risk_low = np.exp(S1)/(1+np.exp(S1)+np.exp(S2))*100
    risk_high = 100.0-risk_no-risk_low
    return risk_no, risk_low, risk_high

parser = ArgumentParser()
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--spreadsheet", default="../Preprocess/prostate_triage.xlsx", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--data_dir", default="/home/user/Documents/coreg", type=str, help="dataset directory")
parser.add_argument('--checkpoint_epoch', default=-1, type=int, help='epoch number to load model')

def main(args):
    experiment_df = pd.read_excel(args.spreadsheet)

    # Raise error if psa or age are blank or "unknown"
    if experiment_df["psa"].isnull().sum() > 0:
        raise ValueError("There are", experiment_df["psa"].isnull().sum(), "cases with blank PSA values")
    if experiment_df["age"].isnull().sum() > 0:
        raise ValueError("There are", experiment_df["age"].isnull().sum(), "cases with blank age values")

    # Create a new dataframe "triage"
    triage_list = []

    args.modelname = args.logdir

    root_dir = args.data_dir

    startfeatures = 32
    channels = (startfeatures, startfeatures*2, startfeatures*4, startfeatures*8, startfeatures*16)
    model = UNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=2,
            channels=channels,
            strides=(2,2,2,2),
            num_res_units=2,
            dropout=0.1,
        )
    model.cuda(0)

    if args.checkpoint_epoch != -1:
        checkpoint_dir = os.path.join(args.logdir, str(args.checkpoint_epoch))
        model_name = "model_" + str(args.checkpoint_epoch) + ".pt"
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        checkpoint = torch.load(checkpoint_path)
    else: # Attempt to load the "model_last.pt" at args.logdir
        checkpoint_path = os.path.join(args.logdir, "model_last.pt")
        checkpoint = torch.load(checkpoint_path)
    model_dict = checkpoint["state_dict"]

    model.load_state_dict(model_dict)
    model.cuda(0)
    model.eval()

    transform_list = [
        transforms.LoadImaged(keys=["axt2", "highb", "adc", "wg","pz","tz"]),
        transforms.EnsureTyped(keys=["axt2", "highb", "adc", "wg","pz","tz"],data_type='tensor'),
        transforms.EnsureChannelFirstd(keys=["axt2", "highb", "adc", "wg","pz","tz"],channel_dim='no_channel'),
        transforms.Orientationd(keys=["axt2", "highb", "adc", "wg","pz","tz"], axcodes="RAS"),
        transforms.Spacingd(
            keys=["axt2", "highb", "adc", "wg","pz","tz"], pixdim=(0.5,0.5,3.0), mode=("bilinear", "bilinear", "bilinear", "nearest","nearest","nearest")
        ),
        transforms.ScaleIntensityRangePercentilesd(
                    keys=["axt2","highb","adc"], lower=0.0, upper=98.0, b_min=0.0, b_max=1.0, clip=True, channel_wise=True
            ),
    ]

    transform_list += [
        transforms.MaskIntensityd(keys=["axt2", "highb", "adc"], mask_key="wg"),
    ]

    transform_list += [
        transforms.ConcatItemsd(keys=["axt2", "highb", "adc"], name="image"),
        transforms.EnsureTyped(keys=["image", "wg","pz","tz"],data_type='tensor'),
    ]
    transforms_all = transforms.Compose(transform_list)

    # Iterate over each label in the Label column
    for label in experiment_df["Label"].unique():
        # Get all rows with the current label
        print("Processing label", label)
        label_rows = experiment_df[experiment_df["Label"] == label]

        np_dict = []

        count = 0
        
        # Iterate over each row
        for _, row in tqdm(label_rows.iterrows(), total=len(label_rows)):
            count += 1
            if count > 50:
                break
            pt_folder = os.sep.join([root_dir,row['folder']])

            p_seg_glob = glob.glob(pt_folder + "/p_axt2_*.nii.gz")
            if len(p_seg_glob) != 1:
                raise ValueError("Expected exactly one p_axt2_*.nii.gz file in", pt_folder,"Instead got ", p_seg_glob)
            p_seg = p_seg_glob[0]
            tz_glob = glob.glob(pt_folder + "/tz_axt2_*.nii.gz")
            if len(tz_glob) != 1:
                raise ValueError("Expected exactly one tz_axt2_*.nii.gz file in", pt_folder,"Instead got ", tz_glob)
            tz_seg = tz_glob[0]
            pz_glob = glob.glob(pt_folder + "/pz_axt2_*.nii.gz")
            if len(pz_glob) != 1:
                raise ValueError("Expected exactly one pz_axt2_*.nii.gz file in", pt_folder,"Instead got ", pz_glob)
            pz_seg = pz_glob[0]

            # load all relevant files
            load_dict = {
                "axt2": pt_folder + "/axt2.nii.gz",
                "adc": pt_folder + "/adc.nii.gz",
                "highb": pt_folder + "/b1600.nii.gz",
                "wg": p_seg,
                "tz": tz_seg,
                "pz": pz_seg
            }
            data = transforms_all(load_dict)

            
            p_seg_np = data["wg"].squeeze().numpy()
            tz_seg_np = data["tz"].squeeze().numpy()
            pz_seg_np = data["pz"].squeeze().numpy()

            voxel_volume = 0.5 * 0.5 * 3.0 / 1000

            wg_vol = (p_seg_np == 1).sum() * voxel_volume
            pz_vol = (pz_seg_np == 1).sum() * voxel_volume
            tz_vol = (tz_seg_np == 1).sum() * voxel_volume
            
            if row["folder"].startswith("AIPR_2"):
                ptid = int(''.join(row['folder'].split('_')[1:3]))
            elif row["folder"].startswith("AIPR_PICAI"):
                ptid = int(''.join(row['folder'].removeprefix('AIPR_PICAI').split('_')[0:2]))
            
            image_torch = data["image"].unsqueeze(0).cuda(0)

            with torch.no_grad():
                output = model(image_torch)
                output_softmax = torch.nn.functional.softmax(output, dim=1)
                output_np = output_softmax[0,1,:,:,:].cpu().numpy()

            # Note: We're coercing PSA and age to the limits in the paper so the model doesn't break down
            # PSA 2-50, age 40-90
            coerced_psa = np.clip(int(row["psa"]), 2, 50)
            coerced_age = np.clip(int(row["age"]), 40, 90)
            PCBG_no, PCBG_low, PCBG_high = PCBG_calc(row["psa"], row["age"])
            PCPT_no, PCPT_low, PCPT_high = PCPT_calc(row["psa"], row["age"])
            # Note for later: We only care about the high risk value

            max_confidence = extract_lesion_candidates(output_np)[0]
            output_tz = output_np * tz_seg_np
            output_pz = output_np * pz_seg_np
            max_tz_confidence = extract_lesion_candidates(output_tz)[0]
            max_pz_confidence = extract_lesion_candidates(output_pz)[0]
            if not pd.isna(row['PIRADS']):
                pirads = int(row['PIRADS'])
            else:
                pirads = -1
            if not pd.isna(row['ISUP']):
                isup = int(row['ISUP'])
            else:
                isup = -1
            max_confidence = output_np[p_seg_np == 1].max()
            max_tz_confidence = output_np[tz_seg_np == 1].max()
            max_pz_confidence = output_np[pz_seg_np == 1].max()

            # Get the PIRADS and ISUP values
            if not pd.isna(row["PIRADS"]):
                pirads = row["PIRADS"]
            else:
                pirads = -1
            if not pd.isna(row["ISUP"]):
                isup = row["ISUP"]
            else:
                isup = -1

            positive = row['PosISUP'] if not pd.isna(row['PosISUP']) else row['PosPIRADS']

            asdf = [ptid, positive, row['age'], wg_vol, pz_vol, tz_vol, row["psa"], row['PSAd'], max_confidence, max_tz_confidence, max_pz_confidence, PCBG_no, PCBG_low, PCBG_high, PCPT_no, PCPT_low, PCPT_high, pirads, isup]
            np_dict.append(np.array(asdf))

        np_tosave = np.stack(np_dict)
        np.save(f"../Triage/triage_{args.modelname}_{label}.npy", np_tosave)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)