"""
Run the coco evaluator on "sample_fashion_test" to evaluate performance using 
AP metric

TODO: add command line interface for all dataset and model weight inputs instead of 
having them hard coded.
"""


from pathlib import Path

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from iMaterialist2020.imaterialist.config import add_imaterialist_config
from iMaterialist2020.imaterialist.data.datasets.coco import register_datadict
from iMaterialist2020.imaterialist.data.dataset_mapper import iMatDatasetMapper
from environs import Env

env = Env()
env.read_env()

# Get training dataframe
path_data = Path(env("path_raw"))
path_image = path_data / "train/"
path_output = Path(env("path_output"))
path_eval = Path(env("path_eval"))
path_data_interim = Path(env("path_interim"))
path_model = Path(env("path_model"))

if __name__=="__main__":
    # load dataframe
    # fixme: this number needs to update or dynamic
    datadic_train = pd.read_feather(path_data_interim / 'imaterialist_train_multihot_n=266721.feather')
    datadic_test = pd.read_feather(path_data_interim / 'imaterailist_test_multihot_n=66680.feather')

    register_datadict(datadic_train, "sample_fashion_train")
    register_datadict(datadic_test, "sample_fashion_test")

    # cfg = setup(args)
    cfg = get_cfg()

    # Add Solver etc.
    add_imaterialist_config(cfg)

        # Merge from config file.
    config_file = "/home/dyt811/Git/cvnnig/iMaterialist2020/configs/config.yaml"
    cfg.merge_from_file(config_file)

    # Load the final weight.
    cfg.MODEL.WEIGHTS = str(path_model / "model_0109999.pth")
    cfg.OUTPUT_DIR = str(path_output)

    trainer = DefaultTrainer(cfg)

    # load weights
    trainer.resume_or_load(resume=False)

    # Evaluate performance using AP metric implemented in COCO API
    evaluator = COCOEvaluator("sample_fashion_test", cfg, False, output_dir=str(path_output))
    val_loader = build_detection_test_loader(cfg, "sample_fashion_test", mapper=iMatDatasetMapper(cfg))
    inference_on_dataset(trainer.model, val_loader, evaluator)