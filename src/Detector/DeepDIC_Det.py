from DeepSIC_Block import DeepSIC_Block
from src.Pulse.projection_fn import define_projection_matrix_and_bias
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Detector.base_model import base_model_generator
from src.Utils.CovarianceType import CovarianceType





class DeepSIC_proj():

    """Bayesian DeepSIC where each block is an independent Bayesian neural network.

        Args:
            symbol_bits (int): Number of bits per symbol.
            num_users (int): Number of users.
            num_antennas (int): Number of receive antennas.
            num_layers (int): Number of soft interference cancellation (SIC) layers.
            hidden_dim (int): Size of the hidden layer of each block.
            cov_type (CovarianceType, optional): Type of covariance for the parameters.
            init_cov_scale (float, optional): Initial parameter covariance scale. Defaults to 0.1.
            Pulse (bool, optional): Whether to use projection matrices. Defaults to False.
            OU (bool, optional): Whether to use Ornstein-Uhlenbeck process for online training. Defaults to True.
            F (bool, optional): Whether to use state transition matrix F in online training. Defaults to False.
        """

    def __init__(
            self,
            symbol_bits: int,
            num_users: int,
            num_antennas: int,
            num_layers: int,
            hidden_dim: int,
            cov_type: CovarianceType = CovarianceType.FULL,
            init_cov_scale: float = 0.1,
            Pulse=False,
            OU=True,
            F=False,
    ):
        self.symbol_bits = symbol_bits
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rx_size = 2 * num_antennas
        self.block_input_size = self.rx_size + symbol_bits * num_users
        self.Pulse = Pulse
        self.cov_type = cov_type

        self.OU = OU
        self.F = F

        self.blocks = self.build_blocks()


    def build_blocks(self):
        layer_block=[]
        blocks = []
        base_model = base_model_generator(
            input_size=self.block_input_size,
            hidden_dim=self.hidden_dim,
            output_size=self.symbol_bits,
        )

        last_params = list(base_model.fc3.parameters())
        n_last = sum(p.numel() for p in last_params)
        base_params = list(base_model.fc1.parameters()) + list(base_model.fc2.parameters())
        n_hidden = sum(p.numel() for p in base_params)
        num_wanted_hidden = 150
        num_wanted_last = 30

        for layer_idx in range(self.num_layers):
            for user in range(self.num_users):
                # Define projection matrices and activation functions
                projection_mat_hidden, phi_hidden = define_projection_matrix_and_bias(n_hidden,num_wanted_hidden)
                projection_mat_last, phi_last = define_projection_matrix_and_bias(n_last,num_wanted_last)

                block = DeepSIC_Block(
                    base_model=base_model,
                    symbol_bits=self.symbol_bits,
                    nun_users=self.num_users,
                    num_antenas=self.num_antennas,
                    hidden_dim=self.hidden_dim,
                    projection_mat_hidden=projection_mat_hidden,
                    projection_mat_last=projection_mat_last,
                    phi_hidden=phi_hidden,
                    phi_last=phi_last,
                )
                layer_block.append(block)
            blocks.append(layer_block)
        return blocks





    def soft_decode(self, rx):
        """Soft decode the received signal using the DeepSIC architecture."""
        # generate inputs for first block
        initaial_inputs = torch.ones((1, self.block_input_size))
        initaial_inputs[0, :self.rx_size] = rx
        initaial_inputs[self.rx_size:] = 1 / self.symbol_bits
        inputs = initaial_inputs
        next_input = torch.zeros((1, self.block_input_size))
        next_input[0, :self.rx_size] = rx
        # Iterate over layers
        if self.OU == True:
            if self.cov_type != 'DLR':
                for layer_idx in range(self.num_layers):
                    offset = self.rx_size
                    for user in range(self.num_users):
                        if user == 0 and layer_idx != 0:
                            inputs = next_input
                        block = self.blocks[layer_idx][user]
                        next_input[offset:offset + self.symbol_bits] = F.softmax(block.forward(inputs), dim=-1)
                        offset += self.symbol_bits
                    inputs = next_input
            else:
                for layer_idx in range(self.num_layers):
                    offset = self.rx_size
                    for user in range(self.num_users):
                        if user == 0 and layer_idx != 0:
                            inputs = next_input
                        block = self.blocks[layer_idx][user]
                        next_input[offset:offset + self.symbol_bits] = F.softmax(block.forward(inputs), dim=-1)
                        offset += self.symbol_bits
                    inputs = next_input




        else:  # add F and beta and Q
            beta = 0.9
            Q = 0.001 * torch.eye(block.z_layers.numel())
            F = torch.eye(block.z_layers.numel())
            if self.cov_type != 'DLR':
                for layer_idx in range(self.num_layers):
                    offset = self.rx_size
                    for user in range(self.num_users):
                        if user == 0 and layer_idx != 0:
                            inputs = next_input
                        block = self.blocks[layer_idx][user]
                        next_input[offset:offset + self.symbol_bits] = F.softmax(block.forward(inputs), dim=-1)
                        offset += self.symbol_bits
                    inputs = next_input

            else:
                for layer_idx in range(self.num_layers):
                    offset = self.rx_size
                    for user in range(self.num_users):
                        if user == 0 and layer_idx != 0:
                            inputs = next_input
                        block = self.blocks[layer_idx][user]
                        next_input[offset:offset + self.symbol_bits] = F.softmax(block.forward(inputs), dim=-1)
                        offset += self.symbol_bits
                    inputs = next_input

        return inputs[self.rx_size:]




    def online_training(self,train_fn,rx,symbools):
        """Perform online training of the DeepSIC model each block independently."""
        ## symbols : list of true symbols for each user (2d tensor)

        #generate inputs for first block
        initaial_inputs = torch.ones((1,self.block_input_size))
        initaial_inputs[0,:self.rx_size] = rx
        initaial_inputs[self.rx_size:] = 1/self.symbol_bits
        inputs = initaial_inputs
        next_input = torch.zeros((1, self.block_input_size))
        next_input[0, :self.rx_size] = rx
        # Iterate over layers
        if self.OU == True:
            if self.cov_type != 'DLR':
                for layer_idx in range(self.num_layers):
                    offset = self.rx_size
                    for user in range(self.num_users):
                        if user == 0 and layer_idx != 0:
                            inputs = next_input
                        block = self.blocks[layer_idx][user]
                        #train layers
                        train_fn(block.z_layers,block.cov_layers,block , inputs , symbools[user] ,
                                 0.9,
                                 block.iniitial_cov_layers,
                                 block.initial_mean_layers)
                        #train last layer
                        train_fn(block.z_last, block.cov_last, block, inputs, symbools[user],
                                 0.9,
                                 block.iniitial_cov_last,
                                 block.initial_mean_last)
                        next_input[offset:offset + self.symbol_bits] = block.forward(inputs)
                        offset += self.symbol_bits
                    inputs = next_input

            else:
                for layer_idx in range(self.num_layers):
                    offset = self.rx_size
                    for user in range(self.num_users):
                        if user == 0 and layer_idx != 0:
                            inputs = next_input
                        block = self.blocks[layer_idx][user]
                        train_fn(block.z_layers,block.diag_layers,block.lr_cov_layers ,block , inputs ,
                                 symbools[user] ,
                                 0.9)
                        train_fn(block.z_last, block.diag_last, block.lr_cov_last, block, inputs,
                                 symbools[user],
                                 0.9)
                        next_input[offset:offset+self.symbol_bits] = block.forward(inputs)
                        offset += self.symbol_bits
                    inputs = next_input


        else: # add F and beta and Q
            beta = 0.9
            Q = 0.001*torch.eye(block.z_layers.numel())
            F = torch.eye(block.z_layers.numel())
            if self.cov_type != 'DLR':
                for layer_idx in range(self.num_layers):
                    offset = self.rx_size
                    for user in range(self.num_users):
                        if user == 0 and layer_idx != 0:
                            inputs = next_input
                        block = self.blocks[layer_idx][user]
                        train_fn(block.z_layers,block.cov_layers,block , inputs , symbools[user],
                                 beta,
                                 Q,
                                 F)
                        train_fn(block.z_last, block.cov_last, block, inputs, symbools[user],
                                 beta,
                                 Q,
                                 F)
                        next_input[offset:offset + self.symbol_bits] = block.forward(inputs)
                        offset += self.symbol_bits
                    inputs = next_input

            else:
                for layer_idx in range(self.num_layers):
                    offset = self.rx_size
                    for user in range(self.num_users):
                        if user == 0 and layer_idx != 0:
                            inputs = next_input
                        block = self.blocks[layer_idx][user]
                        train_fn(block.z_layers,block.cov_layers,block , inputs , symbools[user],
                                 beta,
                                 Q,
                                 F)
                        train_fn(block.z_last, block.cov_last, block, inputs, symbools[user],
                                 beta,
                                 Q,
                                 F)
                        next_input[offset:offset + self.symbol_bits] = block.forward(inputs)
                        offset += self.symbol_bits
                    inputs = next_input
        return


