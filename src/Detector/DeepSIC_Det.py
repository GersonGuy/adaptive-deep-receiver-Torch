from .DeepSIC_Block import DeepSIC_Block_double_proj, DeepSIC_Block_single_proj,DeepSIC_Block_No_proj
from src.Pulse.projection_fn import define_projection_matrix_and_bias
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Detector.base_model import base_model_generator


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
            cov_type: str,
            init_cov_scale: float = 0.1,
            Pulse = False,
            OU=True,
            F=False,
            block_method = 'double_proj'
    ):
        self.symbol_bits = symbol_bits
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rx_size = 2 * num_antennas
        self.block_input_size = self.rx_size + symbol_bits * (num_users-1)
        self.Pulse = Pulse
        self.cov_type = cov_type

        self.OU = OU
        self.F = F
        self.block_method = block_method.lower()

        self.blocks = self.build_blocks()


    def build_blocks(self):

        blocks = []
        base_model = base_model_generator(
            input_size=self.block_input_size,
            hidden_dim=self.hidden_dim,
            output_size=self.symbol_bits,
        )

        if self.block_method == 'double_proj':
            last_params = list(base_model.fc3.parameters())
            n_last = sum(p.numel() for p in last_params)
            base_params = list(base_model.fc1.parameters()) + list(base_model.fc2.parameters())
            n_hidden = sum(p.numel() for p in base_params)
            n_total = n_hidden+n_last
            num_wanted_hidden = 150
            num_wanted_last = 30
        else:
            layers_params = list(base_model.parameters())
            n_total = sum(p.numel() for p in layers_params)
            num_wanted_single = 350

        for layer_idx in range(self.num_layers):
            layer_block = []
            for user in range(self.num_users):
                if self.block_method == 'double_proj':
                    # Define projection matrices and activation functions
                    projection_mat_hidden, phi_hidden = define_projection_matrix_and_bias(n_hidden,num_wanted_hidden)
                    projection_mat_last, phi_last = define_projection_matrix_and_bias(n_last,num_wanted_last)

                    block = DeepSIC_Block_double_proj(
                        base_model=base_model,
                        symbol_bits=self.symbol_bits,
                        num_users=self.num_users,
                        num_antenas=self.num_antennas,
                        hidden_dim=self.hidden_dim,
                        projection_mat_hidden=projection_mat_hidden,
                        projection_mat_last=projection_mat_last,
                        phi_hidden=phi_hidden,
                        phi_last=phi_last,
                        cov_type= self.cov_type
                    )
                    layer_block.append(block)

                elif self.block_method == 'single_proj':
                    # Define projection matrices and activation functions
                    projection_mat_hidden, phi_hidden = define_projection_matrix_and_bias(n_total, num_wanted_single)

                    block = DeepSIC_Block_single_proj(
                        base_model=base_model,
                        symbol_bits=self.symbol_bits,
                        num_users=self.num_users,
                        num_antenas=self.num_antennas,
                        hidden_dim=self.hidden_dim,
                        projection_mat_hidden=projection_mat_hidden,
                        phi_hidden=phi_hidden,
                        cov_type=self.cov_type
                    )
                    layer_block.append(block)
                else:
                    raise "problem with config: projection mode "

            blocks.append(layer_block)
        return blocks


    def soft_decode_batch (self,rx):
        concat_prediction = None
        for train_rx in rx:
            prediction = self.soft_decode(train_rx)
            if concat_prediction is None:
                concat_prediction = prediction
            else:
                concat_prediction = torch.cat([concat_prediction,prediction],dim = 0)

        return concat_prediction


    def soft_decode(self, rx):
        """Soft decode the received signal using the DeepSIC architecture."""
        # generate inputs for first block
        inputs = torch.ones((1, self.block_input_size+self.symbol_bits))
        inputs[0, :self.rx_size] = rx
        inputs[0, self.rx_size:] = 0.5
        next_input = inputs.clone()
        for layer_idx in range(self.num_layers):
            for user in range(self.num_users):
                if user == 0 and layer_idx != 0:
                    inputs = next_input
                block = self.blocks[layer_idx][user]
                #generate inputs
                block_input,start,end = self.generate_input(user,self.rx_size,inputs)
                with torch.no_grad():
                    next_input[0,self.rx_size+start:self.rx_size+end] = block.forward(block_input)

            inputs = next_input
        return inputs[:, self.rx_size:]




    def train_batch (self,train_fn,rx,symbols):
        for train_rx , labels in zip(rx, symbols):
            self.online_training(train_fn,train_rx,labels)


    def online_training(self,train_fn,rx,symbols):
        """Perform online training of the DeepSIC model each block independently."""
        ## symbols : list of true symbols for each user (2d tensor)

        #generate inputs for first block
        inputs = torch.ones((1, self.block_input_size+self.symbol_bits),requires_grad=False)
        inputs[0, :self.rx_size] = rx
        inputs[0, self.rx_size:] = 0.5
        next_input = inputs.clone().detach()
        size = self.symbol_bits
        R = torch.eye(size) * 0.1
        size = self.symbol_bits
        R_last = torch.eye(size) * 0.1
        # Iterate over layers
        if self.OU == True:
            if self.cov_type != 'dlr':
                for layer_idx in range(self.num_layers):
                    for user in range(self.num_users):
                        block = self.blocks[layer_idx][user]
                        # generate input
                        block_input,start,end = self.generate_input(user,self.rx_size,inputs)

                        #train layers
                        mean_upd, cov_upd = train_fn(block.z_layers,block.cov_layers,block , block_input , symbols[user] ,
                                 R,0.9,
                                 block.initial_cov_layers,
                                 block.initial_mean_layers)
                        with torch.no_grad():
                            block.z_layers.copy_(mean_upd)
                            block.cov_layers.copy_(cov_upd)
                        #train last layer
                        if self.block_method == 'double_proj':
                            mean_upd, cov_upd = train_fn(block.z_last, block.cov_last, block, block_input, symbols[user],
                                     R_last,0.9,
                                     block.initial_cov_last,
                                     block.initial_mean_last)
                            with torch.no_grad():
                                block.z_last.copy_(mean_upd)
                                block.cov_last.copy_(cov_upd)
                        next_input[0, self.rx_size+start:self.rx_size+end] = block.forward(block_input)
                    #print(next_input)
                    inputs = next_input

            else:
                for layer_idx in range(self.num_layers):
                    for user in range(self.num_users):
                        block = self.blocks[layer_idx][user]
                        block_input,start,end = self.generate_input(user,self.rx_size,inputs)
                        mean_upd, prec_diag_upd, prec_low_rank_upd =train_fn(block.z_layers,block.diag_layers,block.lr_cov_layers ,block , block_input ,
                                 symbols[user] ,
                                 R,0.9,
                                 block.initial_diag_layers,
                                 block.initial_lr_cov_layers,
                                 block.initial_mean_layers)
                        with torch.no_grad():
                            block.z_layers.copy_(mean_upd)
                            block.diag_layers.copy_(prec_diag_upd)
                            block.lr_cov_layers.copy_(prec_low_rank_upd)

                        if self.block_method == 'double_proj':
                            mean_upd, prec_diag_upd, prec_low_rank_upd = train_fn(block.z_last, block.diag_last, block.lr_cov_last, block, block_input,
                                     symbols[user],
                                     R_last,0.9,
                                     block.initial_diag_last,
                                     block.initial_lr_cov_last,
                                     block.initial_mean_last)
                            with torch.no_grad():
                                block.z_last.copy_(mean_upd)
                                block.diag_last.copy_(prec_diag_upd)
                                block.lr_cov_last.copy_(prec_low_rank_upd)
                            #print(a)
                            next_input[0, self.rx_size + start:self.rx_size + end] = block.forward(block_input)


                    #print(next_input)
                    inputs = next_input


        else: # Q and F should be in different size each online train
            beta = 0.9
            Q = 0.001*torch.eye(block.z_layers.numel())
            F = torch.eye(block.z_layers.numel())
            if self.cov_type != 'DLR':
                for layer_idx in range(self.num_layers):
                    for user in range(self.num_users):
                        if user == 0 and layer_idx != 0:
                            inputs = next_input
                        block = self.blocks[layer_idx][user]
                        block_input, start, end =self.generate_input(user, self.rx_size, inputs)
                        train_fn(block.z_layers,block.cov_layers,block , block_input , symbols[user],
                                 beta,
                                 Q,
                                 F)
                        train_fn(block.z_last, block.cov_last, block, block_input, symbols[user],
                                 beta,
                                 Q,
                                 F)
                        next_input[0, start:end] = block.forward(block_input)

                    inputs = next_input

            else:
                for layer_idx in range(self.num_layers):
                    for user in range(self.num_users):
                        if user == 0 and layer_idx != 0:
                            inputs = next_input
                        block = self.blocks[layer_idx][user]
                        block_input, start, end = self.generate_input(user, self.rx_size, inputs)
                        train_fn(block.z_layers,block.cov_layers,block , block_input , symbols[user],
                                 beta,
                                 Q,
                                 F)
                        train_fn(block.z_last, block.cov_last, block, block_input, symbols[user],
                                 beta,
                                 Q,
                                 F)
                        next_input[0, start:end] = block.forward(block_input)
                    inputs = next_input
        return


    def generate_input(self,user,rx_size,inputs):
        rx_part = inputs[:, :rx_size]
        soft_bits = inputs[:, rx_size:]

        start = user * self.symbol_bits
        end = start + self.symbol_bits
        other_users = torch.cat([soft_bits[:, :start], soft_bits[:, end:]], dim=-1)
        block_input = torch.cat([rx_part, other_users], dim=-1)
        return block_input,start,end


class DeepSIC():
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
            cov_type: str,
            init_cov_scale: float = 0.1,
            Pulse = False,
            OU=True,
            F=False,
            block_method = 'double_proj'
    ):
        self.symbol_bits = symbol_bits
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rx_size = 2 * num_antennas
        self.block_input_size = self.rx_size + symbol_bits * (num_users-1)
        self.cov_type = cov_type

        self.OU = OU
        self.F = F

        self.blocks = self.build_blocks()


    def build_blocks(self):
        blocks = []
        base_model = base_model_generator(
            input_size=self.block_input_size,
            hidden_dim=self.hidden_dim,
            output_size=self.symbol_bits,
        )
        for layer_idx in range(self.num_layers):
            layer_block = []
            for user in range(self.num_users):
                block = DeepSIC_Block_No_proj(
                    base_model=base_model,
                    symbol_bits=self.symbol_bits,
                    num_users=self.num_users,
                    num_antenas=self.num_antennas,
                    hidden_dim=self.hidden_dim,
                    cov_type= self.cov_type
                )
                layer_block.append(block)

            blocks.append(layer_block)
        return blocks


    def soft_decode_batch(self, rx):
        concat_prediction = None
        for train_rx in rx:
            prediction = self.soft_decode(train_rx)
            if concat_prediction is None:
                concat_prediction = prediction
            else:
                concat_prediction = torch.cat([concat_prediction, prediction], dim=0)

        return concat_prediction

    def soft_decode(self, rx):
        """Soft decode the received signal using the DeepSIC architecture."""
        # generate inputs for first block
        inputs = torch.ones((1, self.block_input_size + self.symbol_bits))
        inputs[0, :self.rx_size] = rx
        inputs[0, self.rx_size:] = 0.5
        next_input = inputs.clone()
        for layer_idx in range(self.num_layers):
            for user in range(self.num_users):
                if user == 0 and layer_idx != 0:
                    inputs = next_input
                block = self.blocks[layer_idx][user]
                # generate inputs
                block_input, start, end = self.generate_input(user, self.rx_size, inputs)
                with torch.no_grad():
                    next_input[0, self.rx_size + start:self.rx_size + end] = block.forward(block_input)

            inputs = next_input
        return inputs[:, self.rx_size:]


    def train_batch(self, train_fn, rx, symbols):
        for train_rx, labels in zip(rx, symbols):
            self.online_training(train_fn, train_rx, labels)

    def online_training(self, train_fn, rx, symbols):
        """Perform online training of the DeepSIC model each block independently."""
        ## symbols : list of true symbols for each user (2d tensor)

        # generate inputs for first block
        inputs = torch.ones((1, self.block_input_size + self.symbol_bits), requires_grad=False)
        inputs[0, :self.rx_size] = rx
        inputs[0, self.rx_size:] = 0.5
        next_input = inputs.clone().detach()
        size = self.symbol_bits
        R = torch.eye(size) * 0.1
        size = self.symbol_bits
        R_last = torch.eye(size) * 0.1
        # Iterate over layers
        if self.OU == True:
            if self.cov_type != 'dlr':
                for layer_idx in range(self.num_layers):
                    for user in range(self.num_users):
                        block = self.blocks[layer_idx][user]
                        # generate input
                        block_input, start, end = self.generate_input(user, self.rx_size, inputs)

                        # train layers
                        mean_upd, cov_upd = train_fn(block.z_layers, block.cov_layers, block, block_input,
                                                     symbols[user],
                                                     R, 0.9,
                                                     block.initial_cov_layers,
                                                     block.initial_mean_layers)
                        with torch.no_grad():
                            block.z_layers.copy_(mean_upd)
                            block.cov_layers.copy_(cov_upd)
                        # train last layer
                        next_input[0, self.rx_size + start:self.rx_size + end] = block.forward(block_input)
                    # print(next_input)
                    inputs = next_input

            else:
                for layer_idx in range(self.num_layers):
                    for user in range(self.num_users):
                        block = self.blocks[layer_idx][user]
                        block_input, start, end = self.generate_input(user, self.rx_size, inputs)
                        mean_upd, prec_diag_upd, prec_low_rank_upd = train_fn(block.z_layers, block.diag_layers,
                                                                              block.lr_cov_layers, block,
                                                                              block_input,
                                                                              symbols[user],
                                                                              R, 0.9,
                                                                              block.initial_diag_layers,
                                                                              block.initial_lr_cov_layers,
                                                                              block.initial_mean_layers)
                        with torch.no_grad():
                            block.z_layers.copy_(mean_upd)
                            block.diag_layers.copy_(prec_diag_upd)
                            block.lr_cov_layers.copy_(prec_low_rank_upd)


                            next_input[0, self.rx_size + start:self.rx_size + end] = block.forward(block_input)

                    # print(next_input)
                    inputs = next_input



    def generate_input(self, user, rx_size, inputs):
        rx_part = inputs[:, :rx_size]
        soft_bits = inputs[:, rx_size:]

        start = user * self.symbol_bits
        end = start + self.symbol_bits
        other_users = torch.cat([soft_bits[:, :start], soft_bits[:, end:]], dim=-1)
        block_input = torch.cat([rx_part, other_users], dim=-1)
        return block_input, start, end

