import torch

def define_projection_matrix_and_bias(total_parameter,num_wanted_params,method='xavier',learnable=False ):

    """Define projection matrix and bias function
    Args :
        total_parameter (int): Total number of parameters in the model part (hidden,last).
        num_wanted_params (int): Number of wanted parameters after projection.
        method (str, optional): Method for initializing the projection matrix. Defaults to 'xavier'.
        learnable (bool, optional): Whether the projection matrices is learnable. Defaults to False

    Returns:
        projection_mat (torch.Tensor): Projection matrix of shape (total_parameter, num_wanted_params).
        phi (torch.tensor): Bias function of shape (num_wanted_params,1).
    """
    A = torch.zeros((num_wanted_params, total_parameter))
    phi = torch.zeros((1,total_parameter))

    if method == 'xavier':
        torch.nn.init.xavier_uniform_(A)
        #torch.nn.init.xavier_uniform_(phi)
    elif method == 'normal':
        torch.nn.init.normal_(A, mean=0.0, std=1.0)
        torch.nn.init.normal_(phi, mean=0.0, std=1.0)
    else:
        raise ValueError("Unsupported initialization method")

    if learnable:
        A = torch.nn.Parameter(A)
        phi = torch.nn.Parameter(phi)

    return A,phi


