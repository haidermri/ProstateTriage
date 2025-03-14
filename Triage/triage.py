import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score
import joblib
import os

from report_guided_annotation import extract_lesion_candidates

import argparse

from tqdm import tqdm
import pickle
import socket

import time

parser = argparse.ArgumentParser(description="Prostate Triage LR training script")
parser.add_argument("--logdir", default="test", type=str, help="model foldername")
parser.add_argument("--max_iter", default=int(1e9), type=int, help="max iterations")
parser.add_argument("--eval_only", action="store_true", help="evaluate only")
parser.add_argument("--test", action="store_true", help="run on test set")


def main():
    args = parser.parse_args()
    out_folderbase = f"triage_{args.logdir}"
    out_dir = "./runs/"+out_folderbase
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f"Output directory: {out_dir}")

    model_file = f"{out_dir}/{out_folderbase}.pkl"
    args.modeldir = args.logdir

    if not args.eval_only:
    
        # load data array from npy
        data = np.load(f'triage_{args.modeldir}_train.npy', allow_pickle=True)

        print("Data shape:",data.shape)
        ptid = data[:,0]
        label = data[:,1]
        age = data[:,2]
        wg = data[:,3]
        pz = data[:,4]
        tz = data[:,5]
        psa = data[:,6]
        psad = data[:,7]
        max_confidence = data[:,8]
        max_tz_confidence = data[:,9]
        max_pz_confidence = data[:,10]
        PCBG_no = data[:,11]
        PCBG_low = data[:,12]
        PCBG_high = data[:,13]
        PCPT_no = data[:,14]
        PCPT_low = data[:,15]
        PCPT_high = data[:,16]
        pirads = data[:,17]
        isup = data[:,18]

        X = []
        X.append(age)
        X.append(psad)
        X.append(max_tz_confidence)
        X.append(max_pz_confidence)

        X = np.array(X).T # Stack horizontally, i.e. column-wise
        print("Model input shape:",X.shape)
        y=label

        model = LogisticRegression(n_jobs=-1,verbose=0,max_iter=args.max_iter,multi_class='ovr', penalty='l2', class_weight= 'balanced')
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        }

        model_pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                                        ('model',GridSearchCV(model, param_grid, n_jobs=-1, verbose=1, cv=5, scoring='roc_auc',error_score='raise')),
                                        ])
        
        print("Training model")

        start_time = time.time()
        model_pipeline.fit(X, y)
        end_time = time.time()
        joblib.dump(model_pipeline, model_file)

        print("Done training and saving model to path {}".format(model_file))
        train_time = end_time - start_time
        print("Trained in {} seconds".format(train_time))
    else:
        print("Evaluating only:")
        model_pipeline = joblib.load(model_file)
    
    print("Evaluation")
    # Evaluate model
    if args.test:
        data = np.load(f'triage_{args.modeldir}_test.npy', allow_pickle=True)
    else:
        data = np.load(f'triage_{args.modeldir}_val.npy', allow_pickle=True)

    ptid = data[:,0]
    label = data[:,1]
    age = data[:,2]
    wg = data[:,3]
    pz = data[:,4]
    tz = data[:,5]
    psa = data[:,6]
    psad = data[:,7]
    max_confidence = data[:,8]
    max_tz_confidence = data[:,9]
    max_pz_confidence = data[:,10]
    PCBG_no = data[:,11]
    PCBG_low = data[:,12]
    PCBG_high = data[:,13]
    PCPT_no = data[:,14]
    PCPT_low = data[:,15]
    PCPT_high = data[:,16]
    pirads = data[:,17]
    isup = data[:,18]

    
    X = []
    X.append(age)
    X.append(psad)
    X.append(max_tz_confidence)
    X.append(max_pz_confidence)

    X = np.array(X).T # Stack horizontally, i.e. column-wise
    print("Model input shape:",X.shape)
    y=label

    start_time = time.time()
    y_pred = model_pipeline.predict(X)
    inter_time = time.time()
    y_prob = model_pipeline.predict_proba(X)[:,1]
    end_time = time.time()

    prediction_time = inter_time - start_time
    probability_time = end_time - inter_time
    print(f"Predicted in {prediction_time} seconds, got probabilities in {probability_time} seconds")

    auc = roc_auc_score(y, y_prob)
    f1 = f1_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    ap = average_precision_score(y, y_prob)

    print(f"AUC: {auc}, F1: {f1}, Accuracy: {acc}, AP: {ap}")

    # Save results
    with open(f"{out_dir}/results.txt", "w") as f:
        f.write(f"AUC: {auc}, F1: {f1}, Accuracy: {acc}, AP: {ap}, Train time: {train_time}, Prediction time: {prediction_time}, Probability time: {probability_time}")

    # Save probabilities in excel format
    df = pd.DataFrame({"ptid": ptid, "label": label, "prob": y_prob, "pred": y_pred})
    df.to_excel(f"{out_dir}/probabilities.xlsx")

if __name__ == "__main__":
    main()