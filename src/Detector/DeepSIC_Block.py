import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSIC_Block (nn.Module):
    def __init__(self,base_model, symbol_bits, num_users, num_antenas,hidden_dim ,projection_mat_hidden ,projection_mat_last,
                 phi_hidden,phi_last,cov_type='Full',Rank=10,pred_type='OU'):

        """
        Args:
            base_model (nn.Module): Base neural network model to be used in the block.
            symbol_bits (int): Number of bits per symbol.
            num_users (int): Number of users.
            num_antenas (int): Number of receive antennas.
            hidden_dim (int): Size of the hidden layer of the block.
            projection_mat_last (torch.Tensor): Projection matrix for hidden layer.
            projection_mat_hidden (torch.Tensor): Projection matrix for last layer.
            phi_hidden (torch.tensor): Activation function for hidden layer.
            phi_last (torch.tensor): Activation function for last layer.
            cov_type (CovarianceType, optional): Type of covariance for the parameters.

        """
        super().__init__()
        self.base_model = base_model
        self.symbol_bits = symbol_bits
        self.num_users = num_users
        self.num_antenas = num_antenas
        self.hidden_dim = hidden_dim
        self.cov_type = cov_type

        #projection matrices and biases
        self.A_hidden = projection_mat_hidden
        self.A_last = projection_mat_last
        self.phi_hidden = phi_hidden
        self.phi_last = phi_last
        # Latent variables
        self.z_layers = nn.Parameter(torch.randn(1,self.A.shape[1]))
        self.z_last = nn.Parameter(torch.randn(1,self.A_last.shape[1]))

        #generate covariance matrices
        self.generate_cov_matrix(cov_type,Rank)

        if pred_type == 'OU':
            self.generate_initial_cov_matrix()
            self.initial_mean_layers = self.z_layers.clone()
            self.initial_mean_last = self.z_last.clone()

        # base model parameter
        self.shapes = [p.shape for p in self.base_model.parameters()]
        self.sizes = [p.numel() for p in self.base_model.parameters()]
        self.total_params = sum(self.sizes)
        self.activation = self.extract_activations(base_model)

    def f_z(self, z_concat, obs):
        z_layers_size = self.z_layers.numel()
        self.z_layers = nn.Parameter(z_concat[:z_layers_size])
        self.z_last = nn.Parameter(z_concat[z_layers_size:])
        return self.forward(obs)



    def extract_activations(self,base_model):
        activations = []
        layers = list(base_model)

        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                if i + 1 < len(layers) and not isinstance(layers[i + 1], nn.Linear):
                    activations.append(layers[i + 1])
                else:
                    activations.append(None)

        return nn.ModuleList(activations)


    def forward(self, x):
        theta = self.expend()
        # Unravel theta into model parameters
        offset = 0
        params = []
        for shape, size in zip(self.shapes, self.sizes):
            params.append(theta[offset:offset + size].view(shape))
            offset += size
        for i in range(len(params)//2):
            x = F.linear(x, params[2*i], bias=params[2*i+1])
            activation = self.activation[i]
            if activation !=None:
                x = activation(x)
        return x



    def expend(self):
        """Expand parameters from vector to model parameters"""
        theta_hidden =  self.z_layers @ self.A_hidden  + self.phi_hidden
        theta_last = self.z_last @ self.A_last + self.phi_last
        theta = torch.cat([theta_hidden, theta_last], dim=0)
        return theta



    def generate_cov_matrix(self,cov_type,Rank):
        """Generate covariance matrix based on the specified type."""
        if  cov_type == 'Full':
            self.cov_layers = torch.eye(self.A_hidden.shape[0])
            self.cov_last = torch.eye(self.A_last.shape[0])
        elif cov_type == 'Diagonal':
            self.cov_layers = torch.ones(1,self.A_hidden.shape[0])
            self.cov_last = torch.ones(1,self.A_last.shape[0])
        else:
            self.diag_layers = torch.ones(1,self.A_hidden.shape[0])
            self.diag_last = torch.ones(1,self.A_last.shape[0])
            self.lr_cov_layers = torch.randn(self.A_hidden.shape[0],Rank)
            self.lr_cov_last = torch.randn(self.A_last.shape[0],Rank)
        return

    def generate_initaial_cov_matrix(self,cov_type):
        """Generate covariance matrix based on the specified type."""
        if  cov_type == 'Full':
            self.initial_cov_layers = self.cov_layers.clone()
            self.initial_cov_last =self.cov_last.clone()
        elif cov_type == 'Diagonal':
            self.initial_cov_layers= self.cov_layers.clone()
            self.initial_cov_last = self.cov_last.clone()
        else:
            self.initial_diag_layers = self.diag_layers.clone()
            self.initial_diag_last = self.diag_last.clone()
            self.initial_lr_cov_layers = self.lr_cov_layers.clone()
            self.initial_lr_cov_last = self.lr_cov_last.clone()
        return

