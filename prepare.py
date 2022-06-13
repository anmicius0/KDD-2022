# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Prepare the experimental settings
"""

import paddle


def prep_env():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        "task": "MS",
        "lstm_layer": 2,
        "dropout": 0.05,
        "train_epochs": 10,
        "batch_size": 32,
        "patience": 3,
        "lr": 1e-4,
        "lr_adjust": "type1",
        "train_size": 235,
        "val_size": 10,

        # For split
        "per_split": 35,

        # Colab paths
        "filename": "avg_pn_tstp.csv",
        "data_path": "./data",
        "checkpoints": "./checkpoints",

        # Don't touch these
        "test_size": 0,
        "num_workers": 0,
        "is_debug": False,
        "target": "Patv",
        "pred_file": "predict.py",
        "framework": "paddlepaddle",
        "capacity": 134,
        "input_len": 144,
        "output_len": 288,
        "start_col": 3,
        "in_var": 10,
        "out_var": 1,
        "day_len": 144,
        "total_size": 245,
        "gpu": 0,
        "turbine_id": 0,
    }

    ###
    # Prepare the GPUs
    if paddle.device.is_compiled_with_cuda():
        settings["use_gpu"] = True
        paddle.device.set_device('gpu:{}'.format(settings["gpu"]))
    else:
        settings["use_gpu"] = False
        paddle.device.set_device('cpu')

    print("The experimental settings are: \n{}\n".format(str(settings)))
    return settings
