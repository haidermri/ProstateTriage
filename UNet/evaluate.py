import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from utils.data_utils import get_loader
from monai.networks.nets import UNet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from functools import partial
from monai.inferers import sliding_window_inference
from collections import OrderedDict
from picai_eval import evaluate
from report_guided_annotation import extract_lesion_candidates
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
import pickle
from monai.transforms import Activations, AsDiscrete, Compose
import monai.transforms as transforms
from monai.data import decollate_batch
import torch.nn as nn

from monai.utils.enums import MetricReduction
from monai.losses import DiceCELoss, FocalLoss, TverskyLoss
from monai.metrics import DiceMetric

import socket
import nibabel as nib
import glob
from picai_eval.analysis_utils import calculate_iou

import pandas as pd
import concurrent.futures
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import (Callable, Dict, Hashable, Iterable, List, Optional, Sized,
                    Tuple, Union)

from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass

from picai_eval.analysis_utils import (calculate_dsc, calculate_iou,
                                       label_structure, parse_detection_map)
from picai_eval.image_utils import (read_label, read_prediction,
                                    resize_image_with_crop_or_pad)
from picai_eval.metrics import Metrics

PathLike = Union[str, Path]

# Cases that are radiologist false negatives without lesion annotation
false_negatives = []
# Convert false_negatives to ints
false_negatives = [int(''.join(x.split('_')[1:3])) for x in false_negatives]

# Mdified from the picai_eval repo: https://github.com/DIAGNijmegen/picai_eval
def evaluate_case_test(
    y_det: "Union[npt.NDArray[np.float32], str, Path]",
    y_true: "Union[npt.NDArray[np.int32], str, Path]",
    min_overlap: float = 0.10,
    overlap_func: "Union[str, Callable[[npt.NDArray[np.float32], npt.NDArray[np.int32]], float]]" = 'IoU',
    case_confidence_func: "Union[str, Callable[[npt.NDArray[np.float32]], float]]" = 'max',
    allow_unmatched_candidates_with_minimal_overlap: bool = True,
    y_det_postprocess_func: "Optional[Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]]" = None,
    y_true_postprocess_func: "Optional[Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]]]" = None,
    case_weight: Optional[float] = None,
    lesion_weight: Optional[float] = None,
    idx: Optional[str] = None,
) -> Tuple[List[Tuple[int, float, float]], float]:
    """
    Gather the list of lesion candidates, and classify in TP/FP/FN.

    Lesion candidates are matched to ground truth lesions, by maximizing the number of candidates
    with sufficient overlap (i.e., matches), and secondly by maximizing the total overlap of all candidates.

    Parameters:
    - y_det: Detection map, which should be a 3D volume containing connected components (in 3D) of the
        same confidence. Each detection map may contain an arbitrary number of connected components,
        with different or equal confidences. Alternatively, y_det may be a filename ending in
        .nii.gz/.mha/.mhd/.npy/.npz, which will be loaded on-the-fly.
    - y_true: Ground truth label, which should be a 3D volume of the same shape as the detection map.
        Alternatively, `y_true` may be the filename ending in .nii.gz/.mha/.mhd/.npy/.npz, which should
        contain binary labels and will be loaded on-the-fly. Use `1` to encode ground truth lesion, and
        `0` to encode background.
    - min_overlap: defines the minimal required overlap (e.g., Intersection over Union or Dice similarity
        coefficient) between a lesion candidate and ground truth lesion, to be counted as a true positive
        detection.
    - overlap_func: function to calculate overlap between a lesion candidate and ground truth mask.
        May be 'IoU' for Intersection over Union, or 'DSC' for Dice similarity coefficient. Alternatively,
        provide a function with signature `func(detection_map, annotation) -> overlap [0, 1]`.
    - allow_unmatched_candidates_with_minimal_overlap: when multiple lesion candidates have sufficient
        overlap with the ground truth lesion mask, this determines whether the lesion that is not selected
        counts as a false positive.
    - y_det_postprocess_func: function to apply to detection map. Can for example be used to extract
        lesion candidates from a softmax prediction volume.
    - y_true_postprocess_func: function to apply to annotation. Can for example be used to select the lesion
        masks from annotations that also contain other structures (such as organ segmentations).

    Returns:
    - a list of tuples with:
        (is_lesion, prediction confidence, overlap)
    - case level confidence score derived from the detection map
    """
    y_list: List[Tuple[int, float, float]] = []
    if isinstance(y_true, (str, Path)):
        y_true = read_label(y_true)
    if isinstance(y_det, (str, Path)):
        y_det = read_prediction(y_det)
    if overlap_func == 'IoU':
        overlap_func = calculate_iou
    elif overlap_func == 'DSC':
        overlap_func = calculate_dsc
    elif isinstance(overlap_func, str):
        raise ValueError(f"Overlap function with name {overlap_func} not recognized. Supported are 'IoU' and 'DSC'")

    # convert dtype to float32
    y_true = y_true.astype('int32')
    y_det = y_det.astype('float32')

    # if specified, apply postprocessing functions
    if y_det_postprocess_func is not None:
        y_det = y_det_postprocess_func(y_det)
    if y_true_postprocess_func is not None:
        y_true = y_true_postprocess_func(y_true)

    # check if detection maps need to be padded
    if y_det.shape[0] < y_true.shape[0]:
        print("Warning: padding prediction to match label!")
        y_det = resize_image_with_crop_or_pad(y_det, y_true.shape)
    if np.min(y_det) < 0:
        raise ValueError("All detection confidences must be positive!")

    # perform connected-components analysis on detection maps
    confidences, indexed_pred = parse_detection_map(y_det) # Gets confidences and segmentations in index-matched lists
    lesion_candidate_ids = np.arange(len(confidences))

    if not y_true.any():
        # benign case, all predictions are FPs
        for lesion_confidence in confidences.values():
            y_list.append((0, lesion_confidence, 0.))
    else:
        # malignant case, collect overlap between each prediction and ground truth lesion
        labeled_gt, num_gt_lesions = ndimage.label(y_true, structure=label_structure)
        gt_lesion_ids = np.arange(num_gt_lesions)
        overlap_matrix = np.zeros((num_gt_lesions, len(confidences)))

        for lesion_id in gt_lesion_ids:
            # for each lesion in ground-truth (GT) label
            gt_lesion_mask = (labeled_gt == (1+lesion_id))

            # calculate overlap between each lesion candidate and the current GT lesion
            for lesion_candidate_id in lesion_candidate_ids:
                # calculate overlap between lesion candidate and GT mask
                lesion_pred_mask = (indexed_pred == (1+lesion_candidate_id))
                overlap_score = overlap_func(lesion_pred_mask, gt_lesion_mask)

                # store overlap
                overlap_matrix[lesion_id, lesion_candidate_id] = overlap_score

        # match lesion candidates to ground truth lesion (for documentation on how this works, please see
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html)
        overlap_matrix[overlap_matrix < min_overlap] = 0  # don't match lesions with insufficient overlap
        overlap_matrix[overlap_matrix > 0] += 1  # prioritize matching over the amount of overlap
        matched_lesion_indices, matched_lesion_candidate_indices = linear_sum_assignment(overlap_matrix, maximize=True)

        # remove indices where overlap is zero
        mask = (overlap_matrix[matched_lesion_indices, matched_lesion_candidate_indices] > 0)
        matched_lesion_indices = matched_lesion_indices[mask]
        matched_lesion_candidate_indices = matched_lesion_candidate_indices[mask]

        # all lesion candidates that are matched are TPs
        for lesion_id, lesion_candidate_id in zip(matched_lesion_indices, matched_lesion_candidate_indices):
            lesion_confidence = confidences[lesion_candidate_id]
            overlap = overlap_matrix[lesion_id, lesion_candidate_id]
            overlap -= 1  # return overlap to [0, 1]

            assert overlap > min_overlap, "Overlap must be greater than min_overlap!"

            y_list.append((1, lesion_confidence, overlap))

        # all ground truth lesions that are not matched are FNs
        unmatched_gt_lesions = set(gt_lesion_ids) - set(matched_lesion_indices)
        y_list += [(1, 0., 0.) for _ in unmatched_gt_lesions]

        # all lesion candidates with insufficient overlap/not matched to a gt lesion are FPs
        if allow_unmatched_candidates_with_minimal_overlap:
            candidates_sufficient_overlap = lesion_candidate_ids[(overlap_matrix > 0).any(axis=0)]
            unmatched_candidates = set(lesion_candidate_ids) - set(candidates_sufficient_overlap)
        else:
            unmatched_candidates = set(lesion_candidate_ids) - set(matched_lesion_candidate_indices)
        y_list += [(0, confidences[lesion_candidate_id], 0.) for lesion_candidate_id in unmatched_candidates]

    # determine case-level confidence score
    if case_confidence_func == 'max':
        # take highest lesion confidence as case-level confidence
        case_confidence = np.max(y_det)
    elif case_confidence_func == 'bayesian':
        # if c_i is the probability the i-th lesion is csPCa, then the case-level
        # probability to have one or multiple csPCa lesion is 1 - Π_i{ 1 - c_i}
        case_confidence = 1 - np.prod([(1 - c) for c in confidences.values()])
    else:
        # apply user-defines case-level confidence score function
        case_confidence = case_confidence_func(y_det)

    return y_list, case_confidence, case_weight, lesion_weight, idx

# Evaluate all cases
def evaluate_test(
    y_det: "Iterable[Union[npt.NDArray[np.float64], str, Path]]",
    y_true: "Iterable[Union[npt.NDArray[np.float64], str, Path]]",
    case_weighting: "Optional[Iterable[float]]" = None,
    lesion_weighting: "Optional[Iterable[float]]" = None,
    subject_list: Optional[Iterable[Hashable]] = None,
    min_overlap: float = 0.10,
    overlap_func: "Union[str, Callable[[npt.NDArray[np.float32], npt.NDArray[np.int32]], float]]" = 'IoU',
    case_confidence_func: "Union[str, Callable[[npt.NDArray[np.float32]], float]]" = 'max',
    allow_unmatched_candidates_with_minimal_overlap: bool = True,
    y_det_postprocess_func: "Optional[Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]]" = None,
    y_true_postprocess_func: "Optional[Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]]]" = None,
    num_parallel_calls: int = 3,
    verbose: int = 0,
) -> Metrics:
    """
    Evaluate 3D detection performance.

    Parameters:
    - y_det: iterable of all detection_map volumes to evaluate. Each detection map should a 3D volume
        containing connected components (in 3D) of the same confidence. Each detection map may contain
        an arbitrary number of connected components, with different or equal confidences.
        Alternatively, y_det may contain filenames ending in .nii.gz/.mha/.mhd/.npy/.npz, which will
        be loaded on-the-fly.
    - y_true: iterable of all ground truth labels. Each label should be a 3D volume of the same shape
        as the corresponding detection map. Alternatively, `y_true` may contain filenames ending in
        .nii.gz/.mha/.mhd/.npy/.npz, which should contain binary labels and will be loaded on-the-fly.
        Use `1` to encode ground truth lesion, and `0` to encode background.
    - sample_weight: case-level sample weight. These weights will also be applied to the lesion-level
        evaluation, with same weight for all lesion candidates of the same case.
    - subject_list: list of sample identifiers, to give recognizable names to the evaluation results.
    - min_overlap: defines the minimal required Intersection over Union (IoU) or Dice similarity
        coefficient (DSC) between a lesion candidate and ground truth lesion, to be counted as a true
        positive detection.
    - overlap_func: function to calculate overlap between a lesion candidate and ground truth mask.
        May be 'IoU' for Intersection over Union, or 'DSC' for Dice similarity coefficient. Alternatively,
        provide a function with signature `func(detection_map, annotation) -> overlap [0, 1]`.
    - case_confidence_func: function to derive case-level confidence from detection map. Default: max.
    - allow_unmatched_candidates_with_minimal_overlap: when multiple lesion candidates have sufficient
        overlap with the ground truth lesion mask, this determines whether the lesion that is not selected
        counts as a false positive.
    - y_det_postprocess_func: function to apply to detection map. Can for example be used to extract
        lesion candidates from a softmax prediction volume.
    - y_true_postprocess_func: function to apply to annotation. Can for example be used to select the lesion
        masks from annotations that also contain other structures (such as organ segmentations).
    - num_parallel_calls: number of threads to use for evaluation. Set to 1 to disable parallelization.
    - verbose: (optional) controll amount of printed information.

    Returns:
    - Metrics
    """
    if case_weighting is None:
        case_weighting = itertools.repeat(1)
    if lesion_weighting is None:
        lesion_weighting = itertools.repeat(1)
    if subject_list is None:
        # generate indices to keep track of each case during multiprocessing
        subject_list = itertools.count()

    # initialize placeholders
    case_target: Dict[Hashable, int] = {}
    case_weight: Dict[Hashable, float] = {}
    case_pred: Dict[Hashable, float] = {}
    lesion_results: Dict[Hashable, List[Tuple[int, float, float]]] = {}
    lesion_weight: Dict[Hashable, List[float]] = {}

    # construct case evaluation kwargs
    evaluate_case_kwargs = dict(
        min_overlap=min_overlap,
        overlap_func=overlap_func,
        case_confidence_func=case_confidence_func,
        allow_unmatched_candidates_with_minimal_overlap=allow_unmatched_candidates_with_minimal_overlap,
        y_det_postprocess_func=y_det_postprocess_func,
        y_true_postprocess_func=y_true_postprocess_func,
    )

    with ThreadPoolExecutor(max_workers=num_parallel_calls) as pool:
        if num_parallel_calls >= 2:
            # process the cases in parallel
            futures = {
                pool.submit(
                    evaluate_case_test,
                    y_det=y_det_case,
                    y_true=y_true_case,
                    case_weight=casew,
                    lesion_weight=lesionw,
                    idx=idx,
                    **evaluate_case_kwargs
                ): idx
                for (y_det_case, y_true_case, casew, lesionw, idx) in zip(y_det, y_true, case_weighting, lesion_weighting, subject_list)
            }

            iterator = concurrent.futures.as_completed(futures)
        else:
            # process the cases sequentially
            def func(y_det_case, y_true_case, case_weight, lesion_weight, idx):
                return evaluate_case_test(
                    y_det=y_det_case,
                    y_true=y_true_case,
                    case_weight=case_weight,
                    lesion_weight=lesion_weight,
                    idx=idx,
                    **evaluate_case_kwargs
                )

            iterator = map(func, y_det, y_true, case_weighting, lesion_weighting, subject_list)

        if verbose:
            total: Optional[int] = None
            if isinstance(subject_list, Sized):
                total = len(subject_list)
            iterator = tqdm(iterator, desc='Evaluating', total=total)

        for result in iterator:
            if isinstance(result, tuple):
                # single-threaded evaluation
                lesion_results_case, case_confidence, casew, lesionw, idx = result
            elif isinstance(result, concurrent.futures.Future):
                # multi-threaded evaluation
                lesion_results_case, case_confidence, casew, lesionw, idx = result.result()
            else:
                raise TypeError(f'Unexpected result type: {type(result)}')

            # aggregate results
            case_weight[idx] = casew
            case_pred[idx] = case_confidence
            if lesionw == 0: # exception case for false negatives, should only happen in test set
                # print("Detected false negative case at idx", idx)
                lesion_results_case = [(1.0, case_confidence, 1.0)]
                case_target[idx] = np.max([a[0] for a in lesion_results_case])
                # Thus in the false negative case, case_target should be set to 1
                # Case confidence then should just be the max confidence of the lesion
            elif len(lesion_results_case):
                case_target[idx] = np.max([a[0] for a in lesion_results_case])
            else:
                case_target[idx] = 0

            # accumulate outputs
            lesion_results[idx] = lesion_results_case
            lesion_weight[idx] = [lesionw] * len(lesion_results_case)

    # collect results in a Metrics object
    metrics = Metrics(
        lesion_results=lesion_results,
        case_target=case_target,
        case_pred=case_pred,
        case_weight=case_weight,
        lesion_weight=lesion_weight,
    )

    return metrics

def probe_directory(model_dir, directory, loader, model, args, acc_func, acc2_func, loss_func, post_label, post_pred):
    with torch.no_grad():
        inf_size = [args.roi_x, args.roi_y, args.roi_z]
        if inf_size != [256,256,32]:
            model_inferer = partial(
                sliding_window_inference,
                roi_size=inf_size,
                sw_batch_size=args.sw_batch_size,
                predictor=model,
                overlap=args.infer_overlap,
            )
        else:
            model_inferer = None
        subfolders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
        
        print("Select a folder:")
        for i, subfolder in enumerate(subfolders):
            print(f"{i+1}. {subfolder}")

        if args.model_epoch != -1:
            selected_folder = os.path.join(model_dir, args.logdir)
            output_folder = os.path.join(directory, args.logdir+"_output")
            epochs = int(str(args.model_epoch).strip())
        else:
            folder_index = int(input("Enter the folder number: ")) - 1
            selected_folder = os.path.join(model_dir, subfolders[folder_index])
            output_folder = os.path.join(directory, subfolders[folder_index]+"_output")
            epoch_input = input("Enter epoch number(s) (comma-separated) or leave blank: ")
            epochs = epoch_input.strip().split(",") if ',' in epoch_input else (int(epoch_input.strip()) if epoch_input else None)

        args.logdir = output_folder

        if args.add_to_dirname:
            args.logdir += args.add_to_dirname

        if args.test:
            args.logdir = output_folder + "_test"

            
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
        
        if isinstance(epochs, int):
            df_list = []

            epoch = str(epochs)
            model_path = os.path.join(selected_folder, epoch, f"model_{epochs}.pt")
            checkpoint = torch.load(model_path, map_location="cuda:"+str(args.gpu))
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                new_state_dict[k.replace("backbone.", "")] = v
            model.load_state_dict(new_state_dict)
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"]
            else:
                start_epoch = 0
            if "best_acc" in checkpoint:
                best_acc = checkpoint["best_acc"]
            else:
                best_acc = 0
            print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(model_path, start_epoch, best_acc))
            model.requires_grad_(False) # We don't want to train the models
            
            # Run predictions on dataloader
            rownum = 0
            imnum = 0
            print("Length of loader: ", len(loader.dataset))
            outputs = np.zeros((len(loader.dataset),256,256,32))
            targets = np.zeros((len(loader.dataset),256,256,32))
            lesion_weight = np.zeros((len(loader.dataset),))
            
            for idx, batch_data in tqdm(enumerate(loader)):
                for k in range(batch_data['label'].shape[0]):
                    if int(batch_data['data'][k,0].item()) in false_negatives:
                        lesion_weight[idx*args.batch_size+k] = 0
                    else:
                        lesion_weight[idx*args.batch_size+k] = 1
                data, target = batch_data["image"].cuda(args.gpu), batch_data["label"].cuda(args.gpu)
                with autocast(enabled=args.amp):
                    if model_inferer is not None:
                        logits = model_inferer(data)
                    else:
                        logits = model(data)

                for k in range(logits.shape[0]):
                    # print("PTID: ", batch_data['data'][k,0].item())
                    
                    if not os.path.exists(args.logdir+'/'+epoch):
                        os.makedirs(args.logdir+'/'+epoch)

                    accuracy = acc_func(logits[k,:,:,:,:].unsqueeze(0), target[k,:,:,:,:].unsqueeze(0))
                    accuracy2 = acc2_func(logits[k,:,:,:,:].unsqueeze(0), target[k,:,:,:,:].unsqueeze(0))

                    loss = loss_func(logits[k,:,:,:,:].unsqueeze(0), target[k,:,:,:,:].unsqueeze(0))
                    loss = loss.detach().cpu().numpy()

                    
                    if args.out_channels == 1:
                        sample_logits = torch.sigmoid(logits[k,:,:,:,:])[0,:,:,:]
                    else:
                        sample_logits = torch.softmax(logits[k,:,:,:,:],dim=0)[1,:,:,:]

                    sample_logits = sample_logits.cpu().detach().numpy()
                    
                    df_list.append({'ID': batch_data['data'][k,0].item(), 'accuracy': accuracy.item(), 'accuracy2': accuracy2.item(), 'loss': loss,'anything_segmented': np.any((sample_logits > 0.5)),'highest_confidence': np.max(sample_logits),'gt_positive':batch_data['data'][k,2].item()})
                    
                    sample_label = target[k,0,:,:,:].cpu().detach().numpy()
                    
                    sample_logits_thresh = sample_logits > 0.5
                    
                    outputs[idx*args.batch_size+k,:,:,:] = sample_logits
                    targets[idx*args.batch_size+k,:,:,:] = sample_label
                imnum += data.shape[0]
            if idx*args.batch_size+k+1 != len(loader.dataset):
                print("Warning: Not all images were processed, instead got ", idx*args.batch_size+k+1)

            metrics = evaluate_test(
                            y_det=outputs,
                            y_true=targets,
                            y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0],
                            num_parallel_calls=args.valworkers,
                            lesion_weighting = lesion_weight,
                        )

            AP = metrics.AP
            auroc = metrics.auroc
            picai_score = metrics.score
            print("AP: {}".format(AP),
                    "auroc: {}".format(auroc),
                    "picai_score: {}".format(picai_score))
            
            # Precision-Recall (PR) curve
            precision = metrics.precision
            recall = metrics.recall

            # Receiver Operating Characteristic (ROC) curve
            tpr = metrics.case_TPR
            fpr = metrics.case_FPR
            
            prdisp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=AP)
            rocdisp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auroc) # ROC
            prdisp_fig = prdisp.plot()
            rocdisp_fig = rocdisp.plot()
            print('saving figure at: ', args.logdir+'/'+str(epoch)+'/'+str(epoch)+'_precision_recall_curve.png')
            prdisp_fig.figure_.savefig(args.logdir+'/'+str(epoch)+'/'+str(epoch)+'_precision_recall_curve.png')
            rocdisp_fig.figure_.savefig(args.logdir+'/'+str(epoch)+'/'+str(epoch)+'_roc_curve.png')
            # Free-Response Receiver Operating Characteristic (FROC) curve
            try:
                sensitivity = metrics.lesion_TPR
                fp_per_case = metrics.lesion_FPR
                frocdisp = RocCurveDisplay(fpr=fp_per_case, tpr=sensitivity) # FROC
                frocdisp_fig = frocdisp.plot()
                frocdisp_fig.figure_.savefig(args.logdir+'/'+str(epoch)+'/'+str(epoch)+'_froc_curve.png')
            except:
                print('No lesions detected')
            plt.close('all')

            # pickle metrics
            with open(args.logdir+'/'+str(epoch)+'/'+str(epoch)+'_metrics.pkl', 'wb') as f:
                pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)

            df = pd.DataFrame(df_list)  
            df.to_excel(args.logdir+'/'+str(epoch)+'/experiment_metrics.xlsx', index=False)

parser = argparse.ArgumentParser(description="Prostate triage UNet segmentation evaluation pipeline")
parser.add_argument("--checkpoint", action="store_true", help="resume training from previous checkpoint")
parser.add_argument("--checkpoint_epoch", default=-1, type=int, help="epoch of the checkpoint to load")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--append_logdir", default="", type=str, help="append to logdir")
parser.add_argument("--data_dir", default="/home/user/Documents/coreg", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=300, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=27, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-3, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=5, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=5, type=int, help="number of workers")
parser.add_argument("--valworkers", default=1, type=int, help="number of val workers")
parser.add_argument('--prefetch_factor', default=2, type=int, help='prefetch factor for dataloader')

parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=3, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=0.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=0.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=3.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=256, type=float, help="roi in x direction")
parser.add_argument("--roi_y", default=256, type=float, help="roi in y direction")
parser.add_argument("--roi_z", default=32, type=float, help="roi in z direction")
parser.add_argument('--min_percentile', default=0.0, type=float, help='min percentile for intensity clipping')
parser.add_argument('--max_percentile', default=98.0, type=float, help='max percentile for intensity clipping')

parser.add_argument('--randflip_prob', default=0.5, type=float, help='probability of random flip')
parser.add_argument('--randrotate_prob', default=0.5, type=float, help='probability of random rotate')
parser.add_argument('--randaffine_prob', default=0.5, type=float, help='probability of random affine')
parser.add_argument('--randzoom_prob', default=0.5, type=float, help='probability of random zoom')
parser.add_argument('--randnoise_prob', default=0.2, type=float, help='probability of random noise')
parser.add_argument('--randscale_prob', default=0.2, type=float, help='probability of random scale')
parser.add_argument('--randshift_prob', default=0.2, type=float, help='probability of random shift')

parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="cosine_anneal", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument('--startfeatures', default=32, type=int, help='number of UNet starting features')
parser.add_argument('--alpha', default=0.75, type=float, help='alpha for Focal Loss or CE')

parser.add_argument('--nocache', action='store_true', help='do not use cache')
parser.add_argument('--cache_dir', default='/home/user/Documents/cache', type=str, help='cache directory')

parser.add_argument('--max_grad_norm', default=3.0, type=float, help='max gradient norm for gradient clipping')

parser.add_argument('--add_to_dirname', default='', type=str, help='add to dirname')

parser.add_argument('--test', action='store_true', help='run on test set')
parser.add_argument('--model_epoch', default=-1, type=int, help='epoch to evaluate')

parser.add_argument('--cache_add', type=str, help='add to cache_dir')


if __name__ == '__main__':
    print("Initializing...")
    args = parser.parse_args()

    args.nocache = True

    args.save_model_every = args.val_every

    if not (args.roi_x == 256 and args.roi_y == 256 and args.roi_z == 32):
        raise ValueError("Only 256x256x32 supported")

    args.noamp = True
    args.amp = not args.noamp
    args.distributed = False
    args.gpu = 0
    args.nocompile = True
    args.postrain = False

    directory = './metrics'
    model_dir = '.'
    args.model_dir = model_dir
    
    channels = (args.startfeatures, args.startfeatures*2, args.startfeatures*4, args.startfeatures*8, args.startfeatures*16)
    model = UNet(
        spatial_dims=3,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        channels=channels,
        strides=(2,2,2,2),
        num_res_units=2,
        dropout=0.1,
    )

    model.cuda(args.gpu)
    model.eval()

    if args.out_channels == 2:
        acc_func = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
        acc2_func = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, ignore_empty=False, get_not_nans=True)
    else:
        acc_func = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
        acc2_func = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, ignore_empty=False, get_not_nans=True)
    
    reduction = "mean"
    
    if args.out_channels == 2:
        dice_loss = FocalLoss(to_onehot_y=True, use_softmax=True, alpha=args.alpha, gamma=2.0, reduction=reduction)
        post_label = AsDiscrete(to_onehot=args.out_channels)
        post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    else:
        dice_loss = FocalLoss(use_softmax=False, alpha=args.alpha, gamma=2.0, reduction=reduction)
        post_label = AsDiscrete(threshold=0.5)
        post_pred = AsDiscrete(threshold=0.5)
    
    loss_func = dice_loss

    loader_list = get_loader(args)
    model.eval()
    if args.test:
        loader = loader_list[2]
    else:
        loader = loader_list[1]


    probe_directory(model_dir, directory, loader, model, args, acc_func, acc2_func, loss_func, post_label, post_pred)