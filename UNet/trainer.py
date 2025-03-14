import os
import shutil
import time

import neptune

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
from report_guided_annotation import extract_lesion_candidates
from picai_eval import evaluate
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from monai.data import decollate_batch
import monai.transforms as transforms
import pickle
import matplotlib.pyplot as plt

import nibabel as nib


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)


        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)  
            loss = loss_func(logits, target)
        
        if args.amp and scaler is not None:
            # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # unscale the gradients of optimizer's assigned params in-place
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None,loss_func=None, scaler=None):
    model.eval()
    run_acc = AverageMeter()
    run_loss = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
                data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)

                if loss_func is not None:
                    loss = loss_func(logits, target)
                    run_loss.update(loss.item(), n=logits.shape[0])
            if not logits.is_cuda:
                target = target.cpu()

            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)
            

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                individual_acc = acc.detach().cpu().numpy()
                run_acc.update(individual_acc, n=not_nans.detach().cpu().numpy())

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
        if loss_func is not None:
            avg_loss = run_loss.avg

    return np.mean(run_acc.avg) , avg_loss


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def save_checkpoint_dir(model, epoch, args, logdir, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(logdir, filename)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    if model_inferer is not None:
        print("WARNING: Using model_inferer")
    if args.logdir is not None and args.rank == 0:
        run = neptune.init_run(project="ProstateTriage",
                                api_token="Token",
                                source_files=["**/*.py", '*.py'], 
                                mode='offline', 
                                )  # your credentials
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0

    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args,
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0:
            run['train/loss'].append(train_loss)
            run['train/epoch'].append(epoch)
            if scheduler is not None:
                lr = scheduler.get_last_lr()[0]
                run['train/lr'].append(lr)
            elif optimizer is not None:
                run['train/lr'].append(optimizer.param_groups[-1]['lr'])
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc, val_avg_loss = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
                loss_func=loss_func, 
                scaler=scaler,
            )

            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
               print(
                   "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                   "acc",
                   val_avg_acc,
                   "time {:.2f}s".format(time.time() - epoch_time),
               )
               run['validation/dice'].append(val_avg_acc)
               run['validation/epoch'].append(epoch)
               run['validation/loss'].append(val_avg_loss)
               if val_avg_acc > val_acc_max:
                   print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                   val_acc_max = val_avg_acc
                   b_new_best = True

            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_last.pt", optimizer=optimizer, scheduler=scheduler)
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_last.pt"), os.path.join(args.logdir, "model_best.pt"))
                    if hasattr(model, "metrics"):
                        print('Saving metrics')
                        with open(os.path.join(args.logdir,'metrics.pickle'), 'wb') as handle:
                            pickle.dump(model.metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        if args.rank == 0 and args.logdir is not None and args.save_model_every != 0 and (epoch + 1) % args.save_model_every == 0:
            print("Saving model at epoch {}".format(epoch))
            save_checkpoint_dir(model, epoch, args, os.path.join(args.logdir,str(epoch)), filename="model_{}.pt".format(epoch), optimizer=optimizer, scheduler=scheduler)
            if hasattr(model, "metrics"):
                with open(os.path.join(args.logdir,str(epoch),'metrics_{}.pickle'.format(epoch)), 'wb') as handle:
                    pickle.dump(model.metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
