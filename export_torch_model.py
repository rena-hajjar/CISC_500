import torch
from typing import Tuple, Union
from batchgenerators.utilities.file_and_folder_operations import load_json, join, subdirs

from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
import logging
import argparse
import os
from os.path import isfile

from nnUNetTrainer import nnUNetTrainer


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--in_channels", type=int, default=1)
    return parser.parse_args()


class nnUNetPredictor:
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm
        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring

        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        else:
            print('perform_everything_on_device=True is only supported for cuda devices! Setting to False.')
            perform_everything_on_device = False

        self.device = device
        self.perform_everything_on_device = perform_everything_on_device

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds


def initialize_from_trained_model_folder(model_training_output_dir: str,
                                         use_folds: Union[Tuple[Union[int, str]], None],
                                         checkpoint_name: str = 'checkpoint_final.pth'):
    args = parseArgs()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if use_folds is None:
        use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

    dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
    plans = load_json(join(model_training_output_dir, 'plans.json'))
    plans_manager = PlansManager(plans)

    parameters = []
    for i, f in enumerate(use_folds):
        f = int(f) if f != 'all' else f
        checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                map_location=torch.device('cpu'), weights_only=False)
        if i == 0:
            trainer_name = checkpoint['trainer_name']
            configuration_name = checkpoint['init_args']['configuration']
        parameters.append(checkpoint['network_weights'])

    configuration_manager = plans_manager.get_configuration(configuration_name)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)

    trainer_class = nnUNetTrainer
    network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        enable_deep_supervision=False
    )

    network.load_state_dict(parameters[0])
    network = network.to("cpu")
    network.eval()

    example_input = torch.rand(1, args.in_channels, 512, 768)
    traced_script_module = torch.jit.trace(network, example_input)
    traced_script_module.save(os.path.join(args.output_dir, "model_traced.pt"))
    logging.info(f"Traced model saved to {os.path.join(args.output_dir, 'model_traced.pt')}.")


if __name__ == "__main__":
    model_training_dir = r"C:\Users\Rena\Documents\YEAR4\CISC500\CISC_500\nnUNet_Data\nnUNet_results\Dataset555\nnUNetTrainer__nnUNetPlans__2d"

    initialize_from_trained_model_folder(model_training_output_dir=model_training_dir, use_folds=None)
