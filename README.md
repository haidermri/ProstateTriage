# Using ML to Triage Prostate MRI

This repository hosts the code for a study assessing ML for prostate MRI workflow triaging

## Description

The inclusion of prostate MRI in the diagnostic workup for prostate cancer is increasing the number of MRI referrals. This increases the demand for expert uroradiologist's time. This study explored the use of deep learning on imaging and non-imaging factors for the purpose of automated removal of negative MRI cases from the radiologist's worklist.

The code from this study includes modification of publicly available code. Comments with direct links to the relevant code before modification have been included in the relevant code files. The UNet training and evaluation code was modified from the MONAI SwinUNETR/BTCV repository found here: https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV

## Getting Started

### Installing

```
conda install --file prostate_triage.yml
conda activate prostate_triage
```

### Executing program

The code for this study is partitioned into three categories:
1. Data preprocessing
2. U-Net training
3. Logistic regression training

The code expects data in NIFTI format (.nii.gz extension) organized in the format \[data_dir]/\[study identifier]/\[files].nii.gz. Each folder should have the following files:
* axt2.nii.gz: The axial T2 image
* b1600.nii.gz: The calculated DWI at b=1600
* adc.nii.gz: The calculated apparent diffusion coefficient map
* p_axt2_ai.nii.gz: The nnUNet-generated whole-gland segmentation
* tz_axt2_ai.nii.gz: The nnUNet-generated transition zone segmentation
* pz_axt2_ai.nii.gz: The nnUNet-generated peripheral zone segmentation
* t1_axt2_he.nii.gz: The human expert-generated index lesion segmentation (if available and lesion present, else blank)

Study identifier is expected to take the following anonymized form:
```
AIPR_[anonymized patient identifier]_[anonymized study date]_[unique study ID]
```

 This code also expects an excel spreadsheet containing the following entries per study:
1. Folder: Study identifier that matches the study identifier directory names
2. Age: Patient age
3. PSA: Patient psa at time of scan
4. ISUP: Biopsy results after the scan (if available, else blank)
5. PIRADS: PI-RADS v2.1 interpretation of the scan (if available, else blank)
6. PSAd: PSA density (may be left blank and will be filled in preprocessing)
7. PosPIRADS: Boolean with value 1 if PI-RADS>=3 else 0
8. PosISUP: Boolean with value 1 if ISUP>=2 else 0
9. Label: Whether this study is in the train, validation, or test sets (\[train, val, test])

An example spreadsheet of proper format can be found at ./Preprocess/prostate_triage_pre.xlsx

#### Data Preprocessing
Begin from the ./Preprocess/ subfolder:
1. Adjust hyperparameters in preprocess_paramdic.yaml to the following:
	* coreg_dir: Location of final processed images
	* data_dir: Location of raw unprocessed data
	* pre_coreg_dir: Temporary location used during processing
	* spreadsheet: Name and location of the above excel spreadsheet with metadata
2. Run process_images.py which will preprocess the dataset for the experiment, including generated a new spreadsheet with updated PSAd values based on the prostate whole-gland segmentation
	```
	python process_images.py
	```
3. Run gen_datadicts.py to generate the data dictionary pickle files needed for UNet training. These should be placed in the UNet/dataset folder for UNet training
	```
	python gen_datadicts.py
	```

#### U-Net Training
Train the UNet with:
```
python main.py --logdir [model name] --data_dir [coreg_dir from Data Preprocessing]
```
Note that this script uses DDP to spread training across all available GPUs. Example flags that can assist with training:
* --nocache: Do not use persistent cache when training
* --max_epochs n: Run training for n epochs (default 300)
* --batch_size b: Run training in mini-batches of size b (default 27)

Evaluate the UNet with:
```
python evaluate.py --logdir [model name] --data_dir [coreg_dir from Data Preprocessing] --model_epoch [model epoch to evaluate]
```
Add the --test flag to evaluate on the test dataset instead of the validation dataset

Finally, generate the necessary outputs for logistic regression training with:
```
python gen_triage.py --logdir [model name] --data_dir [coreg_dir from Data Preprocessing] --checkpoint_epoch [model epoch checkpoint to use for LR]
```
This will add .npy files to the Triage folder for use in training a logistic regression model

#### Logistic Regression Training and Evaluation
Train and evaluate the logistic regression model with:
```
python train_triage.py --logdir [model name]
```
Evaluation will subsequently run on the validation dataset. The following flags can also assist in code usage:
* --eval_only: Run only the evaluation portion of the code
* --test: For evaluation, use the test instead of the validation dataset

## Authors

This study is the work of the following authors: <br />
[Emerson P. Grabke](https://www.linkedin.com/in/egrabke), Carolina A. M. Heming, Amit Hadari, Antonio Finelli, Sangeet Ghai, Katherine Lajkosz, Babak Taati, [Masoom A. Haider](mailto:m.haider@utoronto.ca)

From: The Joint Dept of Medical Imaging, Sinai Health System, University of Toronto, Canada

## Acknowledgements
Portions of this work were modified from the [Medical Open Network for AI](https://github.com/Project-MONAI/MONAI) and the [PI-CAI Grand Challenge](https://github.com/DIAGNijmegen/picai_eval): <br />
M. J. Cardoso et al., ‘MONAI: An open-source framework for deep learning in healthcare’, arXiv [cs.LG], 04-Nov-2022. <br />
Saha A, Bosma JS, Twilt JJ, et al. Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI): an international, paired, non-inferiority, confirmatory study. Lancet Oncol 2024; 25: 879–887

## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. <br />
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 <br />
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.