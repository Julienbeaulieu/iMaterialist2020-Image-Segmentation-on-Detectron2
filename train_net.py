"""
iMaterialist 2020 training script. 

This script runs a trainer where we pass in custom dataset mapper which contains all the 
attributes of each instance. 

We register the data dictionnaries, load the configs, and run the trainer
"""


import pandas as pd
import logging
from environs import Env
from pathlib import Path
import pickle

import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.utils.logger import setup_logger

from imaterialist.data.dataset_mapper import iMatDatasetMapper
from imaterialist.config import add_imaterialist_config
from imaterialist.data.datasets.coco import register_datadict
from imaterialist.modeling import build_model

from imaterialist.modeling import roi_heads

# Get environment variables 
env = Env()
env.read_env()

# Set path to the data
path_data_interim = Path(env("path_interim"))

class FashionTrainer(DefaultTrainer):
    'A customized version of DefaultTrainer. We add a custom mapping to the dataloader'
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=iMatDatasetMapper(cfg))
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=iMatDatasetMapper(cfg))

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

def setup(args):
    """
    Setup all the custom and default configs before training
    """
    cfg = get_cfg()
    add_imaterialist_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(args.config_file)
    
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "imaterialist" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="imaterialist")
    return cfg

def main(args):
    """
    load dataframes
    register detectron2 datadictionnaries
    setup config
    initialize the trainer
    run trainer to train the model
    """
    # load dataframe
    # fixme: this number needs to update or dynamic
    # datadic_train = pd.read_feather(path_data_interim / 'imaterialist_train_multihot_n=400.feather')
    # datadic_val = pd.read_feather(path_data_interim / 'imaterailist_test_multihot_n=100.feather')

    datadict_train = pickle.load(open(path_data_interim / 'imaterialist_train_multihot_n=400.p', 'rb'))
    datadict_val = pickle.load(open(path_data_interim / 'imaterialist_test_multihot_n=100.p', 'rb'))

    register_datadict(datadict_train, "sample_fashion_train")
    register_datadict(datadict_val, "sample_fashion_test")

    cfg = setup(args)

    trainer = FashionTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    args.config_file = "/home/julien/data-science/kaggle/imaterialist/configs/exp06.yaml"
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )