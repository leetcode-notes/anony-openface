"""
This module is the driver function for the main experiment. It accepts details on the configuration and setup for the
tests you want to run, and makes calls to the training files
"""
import os
# import comet_ml
import argparse
import base_model as base_model
import AUnets.config as confg
# from AUnets.models import vgg16
# from Experiments.mainpro import train, train_2, train_3
import corr_driver
import intensity
import evaluate

# from Experiments.VGG.test import test_function


"""
Setup configurations for VGG

'M' denotes positions of max-pool layers
Integers denote position of convolutional layer and the number of output channels for that layer
"""
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}

# ################################################################---------#--------#####################

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--experiment", choices=[1, 3, 4, 6, 8, 9], type=int, required=True,
                    help="the experiment you want to run")
    ap.add_argument("-c", "--config", choices=["A", "B", "C", "D", "E"], required=True,
                    help="model configuration choice")
    ap.add_argument("-t", "--train", choices=["y", "n"], required=True, help="train or test")
    ap.add_argument("--load_model", choices=[False, True], type=bool,
                    help="set to true if you want to warm start from a previous training cycle. By default set to False"
                    )
    ap.add_argument("--find_stat", choices=[False, True], type=bool,
                    help="set to true if you want to find mean and standard deviation of the training dataset")
    ap.add_argument("--loss_weight", type=float,
                    help="value defines suppression weight of CorrNet loss in final loss calculation. ")
    ap.add_argument("--sampling_per", type=float, help="sampling size for the ensemble for experiment 6")

    ap.add_argument('--mode', type=str, default='sample', choices=['train', 'test', 'val', 'sample'])
    ap.add_argument('--OF', type=str, default='None', choices=['None', 'Alone', 'Horizontal', 'Vertical', 'Channels',
                                                               'Conv', 'FC6', 'FC7'])
    ap.add_argument('--DEMO', type=str, default='test_set')
    ap.add_argument('--finetuning', type=str, default='imagenet', choices=['emotionnet', 'imagenet', 'random'])
    ap.add_argument('--pretrained_model', type=str, default='')

    # PATHS
    ap.add_argument('--metadata_path', type=str, default='./data')
    ap.add_argument('--log_path', type=str, default='./snapshot/logs')
    ap.add_argument('--model_save_path', type=str, default='./AUnets/models/fold_0/OF_Horizontal/')
    ap.add_argument('--train_path', type=str, default='./AUnets/models/fold_0/OF_Horizontal/')
    ap.add_argument('--test_path', type=str, default='./AUnets/models/fold_0/OF_Horizontal/')

    # HYPERPARAMETERS
    ap.add_argument('--lr', type=float, default=0.00001)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--image_size', type=int, default=224)
    ap.add_argument('--dataset', type=str, default='BP4D', choices=['BP4D'])
    ap.add_argument('--num_epochs', type=int, default=12)
    ap.add_argument('--num_epochs_decay', type=int, default=13)
    ap.add_argument('--stop_training', type=int, default=2)
    ap.add_argument('--beta1', type=float, default=0.5)
    ap.add_argument('--beta2', type=float, default=0.999)
    ap.add_argument('--GPU', type=str, default='0')
    ap.add_argument('--HYDRA', action='store_true', default=False)
    ap.add_argument('--DELETE', action='store_true', default=False)
    ap.add_argument('--TEST_TXT', action='store_true', default=False)
    ap.add_argument('--TEST_PTH', action='store_true', default=False)
    ap.add_argument('--test_model', type=str, default='')
    ap.add_argument('--use_tensorboard', action='store_true', default=False)
    ap.add_argument('--SHOW_MODEL', action='store_true', default=False)
    ap.add_argument('--results_path', type=str, default='./snapshot/results')
    ap.add_argument('--fold', type=str, default='0')
    ap.add_argument('--mode_data', type=str, default='normal', choices=['normal', 'aligned'])
    ap.add_argument('--AU', type=str, default='1')
    # ap.add_argument('--pretrained_model', type=str, default='')
    ap.add_argument('--log_step', type=int, default=2000)  # tensorboard update
    args = vars(ap.parse_args())
    config = ap.parse_args()
    config = confg.update_config(config)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU

    if args["train"] is "y":  # this block is entered if train is "y"

        if args["experiment"] is 8:  # experiment cnn starts training
            """
            Experiment 4 finds correlations between AUs 
            """
            au_net = base_model.DriverVGG4(cfg[args["config"]])
            corr_driver.train_cnn(au_net, config)

    else:  # this block is activated when train is "n" a.k.a when you want to test
        if args["experiment"] is 1:
            au_net = base_model.DriverVGG4(cfg[args["config"]])
            evaluate.Bosphorous().test_cnn(au_net, config)
        elif args["experiment"] is 3:
            au_net = base_model.DriverVGG4(cfg[args["config"]])
            evaluate.Bosphorous().train_cnn(au_net, config)
        elif args["experiment"] is 4:
            au_net = base_model.DriverVGG4(cfg[args["config"]])
            evaluate.DISFA().test_cnn(au_net, config)
        elif args["experiment"] is 6:
            au_net = base_model.DriverVGG4(cfg[args["config"]])
            evaluate.DISFA().train_cnn(au_net, config)
        elif args["experiment"] is 9:
            au_net = base_model.DriverVGG4(cfg[args["config"]])
            evaluate.BP4D().test_cnn(au_net, config)
