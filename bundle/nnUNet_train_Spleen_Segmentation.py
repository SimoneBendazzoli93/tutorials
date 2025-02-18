from monai.config import print_config
import os
import tempfile
from monai.bundle.config_parser import ConfigParser
from monai.apps.nnunet import nnUNetV2Runner
import random
from monai.bundle.nnunet import convert_nnunet_to_monai_bundle
import json
from pathlib import Path

print_config()

os.environ["MONAI_DATA_DIRECTORY"] = "/home/maia-user/MONAI/Data"


def main():

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
    root_dir = tempfile.mkdtemp() if directory is None else directory
    
    dataroot = os.path.join(root_dir, "Task09_Spleen/")

    test_dir = os.path.join(dataroot, "imagesTs/")
    train_dir = os.path.join(dataroot, "imagesTr/")
    label_dir = os.path.join(dataroot, "labelsTr/")
    
    datalist_json = {"testing": [], "training": []}
    
    datalist_json["testing"] = [
    {"image": "./imagesTs/" + file} for file in os.listdir(test_dir) if (".nii.gz" in file) and ("._" not in file)
    ]
    
    datalist_json["training"] = [
    {"image": "./imagesTr/" + file, "label": "./labelsTr/" + file, "fold": 0}
    for file in os.listdir(train_dir)
    if (".nii.gz" in file) and ("._" not in file)
    ]  # Initialize as single fold
    
    random.seed(42)
    random.shuffle(datalist_json["training"])
    
    num_folds = 5
    fold_size = len(datalist_json["training"]) // num_folds
    for i in range(num_folds):
        for j in range(fold_size):
            datalist_json["training"][i * fold_size + j]["fold"] = i
            
    datalist_file = Path(root_dir).joinpath("Task09_Spleen","Task09_Spleen_folds.json")
    with open(datalist_file, "w", encoding="utf-8") as f:
        json.dump(datalist_json, f, ensure_ascii=False, indent=4)
    print(f"Datalist is saved to {datalist_file}")
    
    
    nnunet_root_dir = os.path.join(root_dir, "nnUNet")

    os.makedirs(nnunet_root_dir, exist_ok=True)

    data_src_cfg = os.path.join(nnunet_root_dir, "data_src_cfg.yaml")
    data_src = {
        "modality": "CT",
        "dataset_name_or_id": "09",
        "datalist": os.path.join(root_dir, "Task09_Spleen/msd_task09_spleen_folds.json"),
        "dataroot": os.path.join(root_dir, "Task09_Spleen"),
    }

    ConfigParser.export_config_file(data_src, data_src_cfg)
    
    runner = nnUNetV2Runner(
    input_config=data_src_cfg, trainer_class_name="nnUNetTrainer_10epochs", work_dir=nnunet_root_dir
    )
    
    runner.convert_dataset()
    
    runner.plan_and_process(npfp=2, n_proc=[2, 2, 2])
    
    runner.train_single_model(config="3d_fullres", fold=0)
