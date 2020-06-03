from detectron2.config import CfgNode as CN
from detectron2 import model_zoo
from detectron2.config import get_cfg
from pathlib import Path
from environs import Env
from detectron2.engine import default_setup
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
import os

env = Env()
env.read_env()

def add_imaterialist_config(cfg: CN):
    """
    Add config for imaterialist2 head
    """

    _C = cfg

    _C.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    _C.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    ##### Input #####
    # Set a smaller image size than default to avoid memory problems

    # Size of the smallest side of the image during training
    # _C.INPUT.MIN_SIZE_TRAIN = (400,)
    # # Maximum size of the side of the image during training
    # _C.INPUT.MAX_SIZE_TRAIN = 600

    # # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    # _C.INPUT.MIN_SIZE_TEST = 400
    # # Maximum size of the side of the image during testing
    # _C.INPUT.MAX_SIZE_TEST = 600
    
    _C.SOLVER.IMS_PER_BATCH = 2
    _C.SOLVER.BASE_LR = 0.0004
    _C.SOLVER.MAX_ITER = 50000
    _C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # default: 512
    _C.MODEL.ROI_HEADS.NUM_CLASSES = 46  # 46 classes in iMaterialist
    _C.MODEL.ROI_HEADS.NUM_ATTRIBUTES = 295
    # this should ALWAYS be left at 1 because it will double or more memory usage if higher.
    _C.DATALOADER.NUM_WORKERS = 1

def initialize_imaterialist_config():
    """
    Cannot directly merge until intialize the imaterialist config properly in the first place.
    :return:
    """
    cfg = get_cfg()
    add_imaterialist_config(cfg)
    return cfg

def setup_prediction(args):
    """
    Setup up the cfg per the prediction requirement.
    Will use weight specified in the environmental variable.
    :param args:
    :return:
    """
    cfg = initialize_imaterialist_config()

    # Merge from pretrained or opts
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg = update_weights_outpath(cfg, env("path_trained_weights"))
    # cfg must have (cfg.DATASETS.TEST[0])

    # Set max input size
    #cfg.INPUT.MAX_SIZE_TEST = 1024

    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "imaterialist" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="imaterialist")
    return cfg

def update_weights_outpath(cfg, weights_path):
    """
    Update these two attributes using environmental variable because the CFG past along was hard coded.
    :param cfg:
    :param weights_path:
    :return:
    """
    # Add the trained weights
    cfg.MODEL.WEIGHTS = weights_path
    cfg.OUTPUT_DIR = env("path_output")

    return cfg
