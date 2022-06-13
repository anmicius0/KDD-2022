# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Training and Validation
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
import os
import sys
import time
from typing import Callable

import numpy as np
from paddle.io import DataLoader

from common import EarlyStopping
from common import Experiment
from common import adjust_learning_rate
from common import traverse_wind_farm
from prepare import prep_env


def val(experiment, data_loader, criterion):
    # type: (Experiment, DataLoader, Callable) -> np.array
    """
    Desc:
        Validation function
    Args:
        experiment:
        data_loader:
        criterion:
    Returns:
        The validation loss
    """
    validation_loss = []
    for i, (batch_x, batch_y) in enumerate(data_loader):
        sample, true = experiment.process_one_batch(batch_x, batch_y)
        loss = criterion(sample, true)
        validation_loss.append(loss.item())
    validation_loss = np.average(validation_loss)
    return validation_loss


def train_and_val(experiment, model_folder, is_debug=False):
    # type: (Experiment, str, bool) -> float
    """
    Desc:
        Training and validation
    Args:
        experiment:
        model_folder: folder name of the model
        is_debug:
    Returns:
        None
    """
    args = experiment.get_args()
    model = experiment.get_model()
    train_data, train_loader = experiment.get_data(flag='train')
    val_data, val_loader = experiment.get_data(flag='val')

    path_to_model = os.path.join(args["checkpoints"], model_folder)
    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)

    early_stopping = EarlyStopping(patience=args["patience"], verbose=True)
    model_optim = experiment.get_optimizer()
    criterion = Experiment.get_criterion()

    epoch_start_time = time.time()
    latest_loss = 0
    for epoch in range(args["train_epochs"]):
        iter_count = 0
        train_loss = []
        model.train()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            sample, truth = experiment.process_one_batch(batch_x, batch_y)
            loss = criterion(sample, truth)
            train_loss.append(loss.item())
            loss.backward()
            model_optim.minimize(loss)
            model_optim.step()
        val_loss = val(experiment, val_loader, criterion)
        latest_loss = val_loss

        if is_debug:
            epoch_end_time = time.time()
            print("Elapsed time for epoch-{}: {}".format(epoch, epoch_end_time - epoch_start_time))
            epoch_start_time = epoch_end_time

        # Early Stopping if needed
        early_stopping(val_loss, model, path_to_model, args["turbine_id"])
        if early_stopping.early_stop:
            break
        adjust_learning_rate(model_optim, epoch + 1, args)

    return latest_loss


if __name__ == "__main__":
    settings = prep_env()
    #
    # Set up the initial environment
    # Current settings for the model
    partition = int(sys.argv[1])
    traverse_wind_farm(train_and_val, settings, 'cross_day', partition)
