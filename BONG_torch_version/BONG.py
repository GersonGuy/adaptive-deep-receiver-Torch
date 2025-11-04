import numpy as np
import  torch
from functorch import jacrev

from .SSM_predict import predict_full_F, predict_full_OU, predict_diag_F, predict_diag_OU, predict_DLR_F, predict_DLR_OU

def Update_BONG_lin_full_F(mean, Cov, model, obs,y, R, F, beta, Q):
    """
    fn to compute BONG update  with linearization and full covariance matrix
    prediction takes action according to state transition matrix F

    Arg:
    Mean (np.array): Mean vector of the prior distribution
    Cov (np.array): Covariance matrix of the prior distribution
    model (Callable or nn.Module): supply forward function of the model
    obs (np.array): Observation vector
    y (np.array): Actual observation vector
    R (np.array): Observation noise covariance matrix
    F (np.array): State transition matrix
    beta (float): forgetting factor
    Q (np.array): Process noise covariance matrix


    Returns:
    Mean_upd (np.array): Mean vector of the posterior distribution
    Cov_upd (np.array): Covariance matrix of the posterior distribution
    """

    predict_mean,predict_cov = predict_full_F(mean, Cov, F, beta, Q)

    y_pred = model(obs)
    H = torch.autograd.functional.jacobian(model, obs)
    S = R + H @ predict_cov @ H.T
    C = predict_cov @ H.T
    K = torch.linalg.lstsq(S.T, C.T).solution.T
    mean_upd = predict_mean + K @ (y - y_pred)
    cov_upd = (torch.eye(Cov.shape[0]) - K @ H) @ predict_cov



    return mean_upd, cov_upd


def Update_BONG_lin_full_OU(mean, Cov, model, obs,y, R, gamma, initial_cov, init_mean):
    """
    fn to compute BONG update  with linearization and full covariance matrix
    prediction takes action according to Ornstein–Uhlenbeck proccess

    Arg:
    Mean (np.array): Mean vector of the prior distribution
    Cov (np.array): Covariance matrix of the prior distribution
    model (Callable or nn.Module): supply forward function of the model
    obs (np.array): Observation vector
    y (np.array): Actual observation vector
    R (np.array): Observation noise covariance matrix
    gamma (float): OU process parameter
    initial_cov (np.array): Initial covariance matrix for OU process
    init_mean (np.array): Initial mean vector for OU process

    Returns:
    Mean_upd (np.array): Mean vector of the posterior distribution
    Cov_upd (np.array): Covariance matrix of the posterior distribution
    """

    predict_mean,predict_cov = predict_full_OU(mean, Cov, gamma, initial_cov, init_mean)


    y_pred = model(obs)
    if mean is model.z_layers:
        z = model.z_layers.detach().clone().requires_grad_(True)
        H = torch.func.jacrev(lambda z_: model.forward_bong(obs, z_, 'layers'))(z)
    elif mean is model.z_last:
        z = model.z_last.detach().clone().requires_grad_(True)
        H = torch.func.jacrev(lambda z_: model.forward_bong(obs,z_, 'last'))(z)

    S = R + H @ predict_cov @ H.T
    C = predict_cov @ H.T
    K = torch.linalg.lstsq(S.T, C.T).solution.T
    mean_upd = predict_mean + K @ (y - y_pred)
    cov_upd = (torch.eye(Cov.shape[0]) - K @ H) @ predict_cov


    return mean_upd, cov_upd

def Update_BONG_lin_diag_F(mean, Cov, model, obs,y, R, F, beta, Q):
    """
    fn to compute BONG update  with linearization and diag covariance matrix
    prediction takes action according to state transition matrix F

    Arg:
    Mean (np.array): Mean vector of the prior distribution
    Cov (np.array): Covariance matrix of the prior distribution
    model (Callable or nn.Module): supply forward function of the model
    obs (np.array): Observation vector
    R (np.array): Observation noise covariance matrix
    F (np.array): State transition matrix
    beta (float): forgetting factor
    Q (np.array): Process noise covariance matrix


    Returns:
    Mean_upd (np.array): Mean vector of the posterior distribution
    Cov_upd (np.array): Covariance matrix of the posterior distribution
    """

    predict_mean,predict_cov = predict_diag_F(mean, Cov, F, beta, Q)

    y_pred = model(obs)
    H = torch.autograd.functional.jacobian(model, obs)
    S = R + H @ predict_cov @ H.T
    C = predict_cov @ H.T
    K = torch.linalg.lstsq(S.T, C.T).solution.T
    mean_upd = predict_mean + K @ (y - y_pred)
    cov_upd = (torch.eye(Cov.shape[0]) - K @ H) @ predict_cov
    cov_upd = torch.diag(torch.diag(cov_upd))  # enforce diagonal covariance

    return mean_upd, cov_upd


def Update_BONG_lin_diag_OU(mean, Cov, model, obs,y, R, gamma, initial_cov, init_mean):
    """
        fn to compute BONG update  with linearization and diag covariance matrix
        prediction takes action according OU proccess

        Arg:
        Mean (np.array): Mean vector of the prior distribution
        Cov (np.array): Covariance matrix of the prior distribution
        model (Callable or nn.Module): supply forward function of the model
        obs (np.array): Observation vector
        R (np.array): Observation noise covariance matrix
        F (np.array): State transition matrix
        beta (float): forgetting factor
        Q (np.array): Process noise covariance matrix


        Returns:
        Mean_upd (np.array): Mean vector of the posterior distribution
        Cov_upd (np.array): Covariance matrix of the posterior distribution
        """

    predict_mean,predict_cov = predict_diag_OU(mean, Cov, gamma, initial_cov, init_mean)

    y_pred = model(obs)
    H = torch.autograd.functional.jacobian(model, obs)
    S = R + H @ predict_cov @ H.T
    C = predict_cov @ H.T
    K = torch.linalg.lstsq(S.T, C.T).solution.T
    mean_upd = predict_mean + K @ (y - y_pred)
    cov_upd = (torch.eye(Cov.shape[0]) - K @ H) @ predict_cov
    cov_upd = torch.diag(torch.diag(cov_upd))  # enforce diagonal covariance

    return mean_upd, cov_upd


def Update_BONG_lin_DLR_F(mean, prec_diag, prec_low_rank, model, obs, R, F, beta, Q):
    """
    fn to compute BONG update  with linearization and diag covariance matrix
    prediction takes action according to state transition matrix F

    Arg:
    Mean (np.array): Mean vector of the prior distribution
    prec_diag: diagonal component of the prior covariance
    prec_low_rank: low rank component of the prior covariance
    model (Callable or nn.Module): supply forward function of the model
    obs (np.array): Observation vector
    R (np.array): Observation noise covariance matrix
    F (np.array): State transition matrix
    beta (float): forgetting factor
    Q (np.array): Process noise covariance matrix


    Returns:
    Mean_upd (np.array): Mean vector of the posterior distribution
    prec_diag_upd (np.array): Covariance matrix of the posterior distribution
    prec_low_rank_upd (np.array): Covariance matrix of the posterior distribution
    """

    # Prediction step
    predict_mean, prec_diag, prec_low_rank = predict_DLR_F(mean, prec_diag, prec_low_rank, F, beta, Q)


    y_pred = model(obs)
    H = torch.autograd.functional.jacobian(model, obs)
    P, L = prec_low_rank.shape
    R_chol = torch.linalg.cholesky(R)
    A = torch.linalg.solve(R_chol, np.eye(R.shape[0])).T
    HTA = H.T @ A
    prec_lr_tilde = torch.hstack([prec_low_rank, HTA.reshape(P, -1)])
    _, L_tilde = prec_lr_tilde.shape
    Dinv_Ltilde = prec_lr_tilde / prec_diag
    G = torch.linalg.pinv(torch.eye(L_tilde, device=R.device) + prec_lr_tilde.T @ Dinv_Ltilde) #matrix that inv in woodbury identity


    # Update step
    AAT = A @ A.T
    term1 = HTA @ AAT / prec_diag
    term2 = (Dinv_Ltilde @ G) @ (Dinv_Ltilde.T @ (HTA @ AAT))
    mean_update = term1 - term2
    mean_upd = predict_mean + mean_update @ (obs - y_pred)

    #reduce back to "L" the rank from "L+m"

    U, S, _ = torch.linalg.svd(prec_lr_tilde, full_matrices=False)
    U_new, S_new = U[:, :L], S[:L]
    U_extra, S_extra = U[:, L:], S[L:]

    prec_low_rank_upd = U_new * S_new
    extra_prec_lr = U_extra * S_extra
    add_diag = torch.einsum("ij,ij->i", extra_prec_lr, extra_prec_lr).unsqueeze(1)
    prec_diag_upd = prec_diag + add_diag

    return mean_upd, prec_diag_upd, prec_low_rank_upd

def Update_BONG_lin_DLR_OU(mean, prec_diag, prec_low_rank, model, obs, R, gamma, initial_prec_diag, initial_prec_low_rank, init_mean):
    """
    fn to compute BONG update  with linearization and diag covariance matrix
    prediction takes action according OU process

    Arg:
    Mean (np.array): Mean vector of the prior distribution
    prec_diag: diagonal component of the prior covariance
    prec_low_rank: low rank component of the prior covariance
    model (Callable or nn.Module): supply forward function of the model
    obs (np.array): Observation vector
    R (np.array): Observation noise covariance matrix
    F (np.array): State transition matrix
    beta (float): forgetting factor
    Q (np.array): Process noise covariance matrix


    Returns:
    Mean_upd (np.array): Mean vector of the posterior distribution
    prec_diag_upd (np.array): Covariance matrix of the posterior distribution
    prec_low_rank_upd (np.array): Covariance matrix of the posterior distribution
    """
    predict_mean, prec_diag, prec_low_rank = predict_DLR_OU(mean, prec_diag, prec_low_rank, gamma, initial_prec_diag, initial_prec_low_rank, init_mean)

    y_pred = model(obs)
    H = torch.autograd.functional.jacobian(model, obs)
    P, L = prec_low_rank.shape
    R_chol = torch.linalg.cholesky(R)
    A = torch.linalg.solve(R_chol, np.eye(R.shape[0])).T
    HTA = H.T @ A
    prec_lr_tilde = torch.hstack([prec_low_rank, HTA.reshape(P, -1)])
    _, L_tilde = prec_lr_tilde.shape
    Dinv_Ltilde = prec_lr_tilde / prec_diag
    G = torch.linalg.pinv(
        torch.eye(L_tilde, device=R.device) + prec_lr_tilde.T @ Dinv_Ltilde)  # matrix that inv in woodbury identity

    # Update step
    AAT = A @ A.T
    term1 = HTA @ AAT / prec_diag
    term2 = (Dinv_Ltilde @ G) @ (Dinv_Ltilde.T @ (HTA @ AAT))
    mean_update = term1 - term2
    mean_upd = predict_mean + mean_update @ (obs - y_pred)

    # reduce back to "L" the rank from "L+m"

    U, S, _ = torch.linalg.svd(prec_lr_tilde, full_matrices=False)
    U_new, S_new = U[:, :L], S[:L]
    U_extra, S_extra = U[:, L:], S[L:]

    prec_low_rank_upd = U_new * S_new
    extra_prec_lr = U_extra * S_extra
    add_diag = torch.einsum("ij,ij->i", extra_prec_lr, extra_prec_lr).unsqueeze(1)
    prec_diag_upd = prec_diag + add_diag

    return mean_upd, prec_diag_upd, prec_low_rank_upd



