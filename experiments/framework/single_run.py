import torch
import time
import json
import math
from tqdm import tqdm
from experiments.framework.read_config import load_config,validate_config,clean_config
from src.channel.Uplink_MIMO_Channel import UplinkMimoChannel
from src.Detector.DeepDIC_Det import DeepSIC_proj
from src.channel.Modulation import MODULATIONS
from BONG_torch_version import BONG,BOG
from src.Utils.utils import prepare_experiment_data



def create_model(config):
    """create DeepSIC Detector base on the configuration"""
    model_config = config['model']
    channel_config = config['channel']
    algo_config = config['algorithm']
    model_type = model_config['type'].lower()

    if model_type == 'deepsic':
        detector = DeepSIC_proj(
            symbol_bits = int(torch.log2(torch.tensor(len(MODULATIONS[channel_config['modulation']])))),
            num_users=channel_config['num_users'],
            num_antennas=channel_config['num_antennas'],
            num_layers=model_config['num_layers'],
            hidden_dim=model_config['hidden_dim'],
            cov_type= config['algorithm']['covariance_type'],
            init_cov_scale= model_config['init_param_cov'],
            Pulse=model_config["Pulse"],
            OU= model_config["OU"],
            F = model_config["F"]
        )


    else:
        pass

    return detector


def create_channel(config):
    """create up-link Mimo channel"""
    channel_config = config['channel']
    channel = UplinkMimoChannel(
        path =channel_config['channel_path'],
        modulation_type=channel_config['modulation'],
        num_users=channel_config['num_users'],
        num_antennas=channel_config['num_antennas'],
        apply_non_linearity= not channel_config['linear_channel']
    )

    return channel

def create_online_train_fn(config):
    """create fn for online learning"""
    algo_config = config['algorithm']
    fn_name = 'Update'

    if algo_config['method'].lower() == 'bong':
        fn_name += '_BONG_lin'
    else:
        fn_name += '_BOG_lin'

    if algo_config['covariance_type'].lower() == 'dlr':
        fn_name += '_DLR'
    elif algo_config['covariance_type'].lower() == 'full':
        fn_name += '_full'
    else:
        fn_name += '_diag'

    if config['model']['OU'] == True:
        fn_name += '_OU'
    else:
        fn_name += '_F'
    if algo_config['method'].lower() == 'bog' and  algo_config['reparameterized'] == True:
        fn_name += '_reparm'


    if algo_config['method'].lower() == 'bong':
        online_fn = getattr(BONG, fn_name)
    else:
        online_fn = getattr(BOG,fn_name)

    return online_fn

def evaluate_model(model, test_rx: torch.Tensor, test_labels: torch.Tensor) -> float:
    """
    Test model and return Bit Error Rate (BER).

    Args:
        model: DeepSIC-like model with soft_decode_batch method.
        test_rx (torch.Tensor): Received signals, shape (num_samples, rx_dim)
        test_labels (torch.Tensor): True bits, shape (num_samples, num_users, symbol_bits)

    Returns:
        float: Bit Error Rate (BER)
    """
    predictions = model.soft_decode_batch(test_rx)

    num_users = model.num_users
    symbol_bits = model.symbol_bits
    predictions = predictions.reshape(-1, num_users, symbol_bits)

    predicted_bits = (predictions > 0.5).float()

    total_bits = test_labels.numel()
    bit_errors = (predicted_bits != test_labels).sum().item()
    ber = bit_errors / total_bits

    return ber




def measure_runtime(detector,step_fn,rx,y):
    """measure runtime of the model"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    detector.train_batch(step_fn,rx,y)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    train_time = end_time-start_time

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    detector.soft_decode_batch(rx)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    inference_time = end_time-start_time

    return train_time,inference_time


def run_experiment(config):
    """perform one experiment due to configuration"""
    # create channel
    channel = create_channel(config)
    # create model
    detector = create_model(config)
    # import online learning function
    step_fn = create_online_train_fn(config)

    # make dataset
    sync_frames = config['experiment']['sync_frames']
    track_frames = config['experiment']['track_frames']
    total_frames = sync_frames + track_frames

    sync_dataloader = prepare_experiment_data(
        channel=channel,
        num_samples=config['experiment']['symbols_per_frame'],
        num_frames=sync_frames,
        snr=config['channel']['snr'],
    )

    track_dataloader = prepare_experiment_data(
        channel=channel,
        num_samples=config['experiment']['pilot_per_frame'],
        num_frames=track_frames,
        snr=config['channel']['snr'],
        start_frame=sync_frames
    )

    test_dataloader = prepare_experiment_data(
        channel=channel,
        num_samples=config['experiment']['test_dim'],
        num_frames=total_frames,
        snr=config['channel']['snr'],
    )

    test_dataloader_iter = iter(test_dataloader)

    #measure runtime
    train_rx ,train_label = next(test_dataloader_iter)
    training_time , inference_time = measure_runtime(detector,step_fn,train_rx,train_label)
    # run sync frames+test each 3 frames.

    sync_ber = []
    for train_rx,train_label in  tqdm(sync_dataloader, total=sync_frames, leave=False, desc='Sync frames'):
        detector.train_batch(step_fn,train_rx,train_label)
        test_rx, test_labels = next(test_dataloader_iter)
        ber = evaluate_model(detector,test_rx,test_labels)
        sync_ber.append(ber)

    # run train frames +test each frame.
    track_ber = []
    for train_rx, train_label in tqdm(track_dataloader, total=track_frames, leave=False, desc='track frames'):
        detector.train_batch(step_fn, train_rx, train_label)
        test_rx, test_label = next(test_dataloader_iter)
        ber = evaluate_model(detector,test_rx, test_label)
        track_ber.append(ber)
    # generate results
    print(sync_ber,track_ber)
    return {training_time,inference_time,sync_ber,track_ber},detector



def main():
    """main function to run experiment """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    import argparse

    parser = argparse.ArgumentParser(description='Run experiment from JSON config')
    parser.add_argument('--config_path', type=str, help='Path to experiment config JSON file',
                        default="single_config.json")
    parser.add_argument('--output_dir', type=str, help='Base output directory for results',
                        default=r"C:\Users\owner\OneDrive\Desktop\Msc\Codes\adaptive-deep-receiver-Torch\experiments\data")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config_path)
    validate_config(config)
    config = clean_config(config)
    """config_hash = generate_config_hash(config)"""


    results , model = run_experiment(config)
    #save results
    return



if __name__ =='__main__':
    main()





