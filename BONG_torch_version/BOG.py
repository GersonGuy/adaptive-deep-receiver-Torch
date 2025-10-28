
import pytorch as torch
import numpy as np

from SSM_predict import predict_full_F, predict_full_OU, predict_diag_F, predict_diag_OU, predict_DLR_F, predict_DLR_OU

def Update_BOG_lin_full_F(mean, Cov, model, obs ,y,learning_rate, R, F, beta, Q):
    """
    fn to compute BOG update  with linearization and full covariance matrix
    prediction takes action according to state transition matrix F

    Arg:
    Mean (np.array): Mean vector of the prior distribution
    Cov (np.array): Covariance matrix of the prior distribution
    model (Callable or nn.Module): supply forward function of the model
    obs (np.array): Observation vector
    y (np.array): Actual observation vector
    learning_rate (float): learning rate for the update step
    R (np.array): Observation noise covariance matrix
    F (np.array): State transition matrix
    beta (float): forgetting factor
    Q (np.array): Process noise covariance matrix


    Returns:
    Mean_upd (np.array): Mean vector of the posterior distribution
    Cov_upd (np.array): Covariance matrix of the posterior distribution
    """

    predict_mean ,predict_cov= predict_full_F(mean, Cov, F, beta, Q)

    y_pred = model(obs)
    H = torch.autograd.functional.jacobian(model, obs)
    update_term = H.T @ torch.linalg.inv(R) @ (y - y_pred)
    R_inv = torch.linalg.inv(R)
    prec_update = (
            -4 * learning_rate * predict_cov @ update_term @ (predict_mean.T)
            + 2 * learning_rate * predict_cov * (H.T @ R_inv) @ H @ predict_cov
    )
    cov_upd_inv = torch.linalg.inv(predict_cov) + prec_update
    cov_upd = torch.linalg.inv(cov_upd_inv)
    temp = cov_upd @ cov_upd_inv
    mean_upd = temp @ predict_mean + 2 * learning_rate * temp @ update_term

    return mean_upd, cov_upd



def Update_BOG_lin_full_OU(mean, Cov, model, obs, y, learning_rate, R, gamma, initial_Cov, init_mean):
    """
    fn to compute BOG update  with linearization and full covariance matrix
    prediction takes action according to state transition matrix F

    Arg:
    Mean (np.array): Mean vector of the prior distribution
    Cov (np.array): Covariance matrix of the prior distribution
    model (Callable or nn.Module): supply forward function of the model
    obs (np.array): Observation vector
    y (np.array): Actual observation vector
    learning_rate (float): learning rate for the update step
    R (np.array): Observation noise covariance matrix
    F (np.array): State transition matrix
    beta (float): forgetting factor
    Q (np.array): Process noise covariance matrix


    Returns:
    Mean_upd (np.array): Mean vector of the posterior distribution
    Cov_upd (np.array): Covariance matrix of the posterior distribution
    """

    predict_mean, predict_cov = predict_full_OU(mean, Cov, gamma, initial_Cov, init_mean)

    y_pred = model(obs)
    H = torch.autograd.functional.jacobian(model, obs)
    update_term = H.T @ torch.linalg.inv(R) @ (y - y_pred)
    R_inv = torch.linalg.inv(R)
    prec_update = (
            -4 * learning_rate * predict_cov @ update_term @ (predict_mean.T)
            + 2 * learning_rate * predict_cov * (H.T @ R_inv) @ H @ predict_cov
    )
    cov_upd_inv = torch.linalg.inv(predict_cov) + prec_update
    cov_upd = torch.linalg.inv(cov_upd_inv)
    temp = cov_upd @ cov_upd_inv
    mean_upd = temp @ predict_mean + 2 * learning_rate * temp @ update_term

    return mean_upd, cov_upd



def Update_BOG_lin_DLR_F(mean, prec_diag, prec_low_rank, model, obs, y, learning_rate, R, F, beta, Q):
    """
    fn to compute BOG update  with linearization and diagonal low rank covariance matrix
    prediction takes action according to state transition matrix F

    Arg:
    Mean (np.array): Mean vector of the prior distribution
    prec_diag (np.array): Diagonal component of the precision matrix
    prec_low_rank (np.array): Low rank component of the precision matrix
    model (Callable or nn.Module): supply forward function of the model
    obs (np.array): Observation vector
    y (np.array): Actual observation vector
    learning_rate (float): learning rate for the update step
    R (np.array): Observation noise covariance matrix
    F (np.array): State transition matrix
    beta (float): forgetting factor
    Q (np.array): Process noise covariance matrix


    Returns:
    Mean_upd (np.array): Mean vector of the posterior distribution
    prec_diag_upd (np.array): Diagonal component of the precision matrix
    prec_low_rank_upd (np.array): Low rank component of the precision matrix
    """
    predcit_mean, prec_diag, prec_lr = predict_DLR_F(mean, prec_diag, prec_low_rank, F, beta, Q)

    y_pred = model(obs)
    H = torch.autograd.functional.jacobian(model, obs)

    P, L = prec_low_rank.shape
    R_chol = torch.linalg.cholesky(R)
    A = torch.linalg.solve(R_chol, np.eye(R.shape[0])).T

    G = torch.linalg.inv(torch.eye(L)+prec_lr.T@(prec_lr/prec_diag))
    prec_update = H.T @ A
    B = prec_update / prec_diag - (prec_lr / prec_diag @ G) @ (
            (prec_lr / prec_diag).T @ prec_update)
    upd_mean = predcit_mean + learning_rate *H.T @ A @ A.T @ (y - y_pred)
    upd_prec_diag = prec_diag + learning_rate/2 * torch.diag(B @ B.T)
    upd_prc_lr = prec_lr + learning_rate* B @ B.T @prec_lr


    return upd_mean, upd_prec_diag, upd_prc_lr




def Update_BOG_lin_DLR_OU(mean, prec_diag, prec_low_rank, model, obs, y, learning_rate, R, gamma, initial_Cov, init_mean):
    """
    fn to compute BOG update  with linearization, reparametrization and diagonal low rank covariance matrix
    prediction takes action according to state transition matrix F

    Arg:
    Mean (np.array): Mean vector of the prior distribution
    prec_diag (np.array): Diagonal component of the precision matrix
    prec_low_rank (np.array): Low rank component of the precision matrix
    model (Callable or nn.Module): supply forward function of the model
    obs (np.array): Observation vector
    y (np.array): Actual observation vector
    learning_rate (float): learning rate for the update step
    R (np.array): Observation noise covariance matrix
    gamma (float): OU process parameter
    initial_Cov (np.array): Initial covariance matrix
    init_mean (np.array): Initial mean vector


    Returns:
    Mean_upd (np.array): Mean vector of the posterior distribution
    prec_diag_upd (np.array): Diagonal component of the precision matrix
    prec_low_rank_upd (np.array): Low rank component of the precision matrix
    """
    predcit_mean, prec_diag, prec_lr = predict_DLR_OU(mean, prec_diag, prec_low_rank, gamma, initial_Cov, init_mean)

    y_pred = model(obs)
    H = torch.autograd.functional.jacobian(model, obs)

    P, L = prec_low_rank.shape
    R_chol = torch.linalg.cholesky(R)
    A = torch.linalg.solve(R_chol, np.eye(R.shape[0])).T

    G = torch.linalg.inv(torch.eye(L) + prec_lr.T @ (prec_lr / prec_diag))
    prec_update = H.T @ A
    B = prec_update / prec_diag - (prec_lr / prec_diag @ G) @ (
            (prec_lr / prec_diag).T @ prec_update)
    upd_mean = predcit_mean + learning_rate * H.T @ A @ A.T @ (y - y_pred)
    upd_prec_diag = prec_diag + learning_rate / 2 * torch.diag(B @ B.T)
    upd_prc_lr = prec_lr + learning_rate * B @ B.T @ prec_lr

    return upd_mean, upd_prec_diag, upd_prc_lr




def Update_BOG_lin_diag_F(mean, cov, model, obs, y, learning_rate, R, F, beta, Q):
    """
    fn to compute BOG update  with linearization and diagonal covariance matrix
    prediction takes action according to state transition matrix F

    Arg:
    Mean (np.array): Mean vector of the prior distribution
    Cov (np.array): Diagonal component of the precision matrix
    model (Callable or nn.Module): supply forward function of the model
    obs (np.array): Observation vector
    y (np.array): Actual observation vector
    learning_rate (float): learning rate for the update step
    R (np.array): Observation noise covariance matrix
    F (np.array): State transition matrix
    beta (float): forgetting factor
    Q (np.array): Process noise covariance matrix


    Returns:
    Mean_upd (np.array): Mean vector of the posterior distribution
    prec_diag_upd (np.array): Diagonal component of the precision matrix
    """

    predict_mean, predict_cov = predict_diag_F(mean, cov, F, beta, Q)

    y_pred = model(obs)
    H = torch.autograd.functional.jacobian(model, obs)
    R_inv = torch.linalg.inv(R)
    update_term = H.T @ R_inv @ (y - y_pred)
    prec_update = (
            -4 * learning_rate * predict_cov @ update_term @ (predict_mean.T)
            + 2 * learning_rate * predict_cov * (H.T @ R_inv) @ H @ predict_cov
    )
    cov_upd_inv = torch.linalg.inv(predict_cov) + prec_update
    cov_upd = torch.linalg.inv(cov_upd_inv)
    temp = cov_upd @ cov_upd_inv
    mean_upd = temp @ predict_mean + 2 * learning_rate * temp @ update_term
    cov_upd_diag = torch.diag(torch.diag(cov_upd))

    return mean_upd, cov_upd_diag


def Update_BOG_lin_diag_OU(mean, cov, model, obs, y, learning_rate, R,initial_diag_Cov, init_mean, gamma):
    """
    fn to compute BOG update  with linearization and diagonal covariance matrix
    prediction takes action according to OU process

    Arg:
    Mean (np.array): Mean vector of the prior distribution
    Cov (np.array): Diagonal component of the precision matrix
    model (Callable or nn.Module): supply forward function of the model
    obs (np.array): Observation vector
    y (np.array): Actual observation vector
    learning_rate (float): learning rate for the update step
    R (np.array): Observation noise covariance matrix
    initial_Cov (np.array): Initial covariance matrix
    init_mean (np.array): Initial mean vector
    gamma (float): OU process parameter


    Returns:
    Mean_upd (np.array): Mean vector of the posterior distribution
    prec_diag_upd (np.array): Diagonal component of the precision matrix
    """

    predict_mean, predict_cov = predict_diag_OU(mean, cov, gamma, initial_diag_Cov, init_mean)

    y_pred = model(obs)
    H = torch.autograd.functional.jacobian(model, obs)
    R_inv = torch.linalg.inv(R)
    update_term = H.T @ R_inv @ (y - y_pred)
    prec_update = (
            -4 * learning_rate * predict_cov @ update_term @ (predict_mean.T)
            + 2 * learning_rate * predict_cov * (H.T @ R_inv) @ H @ predict_cov
    )
    cov_upd_inv = torch.linalg.inv(predict_cov) + prec_update
    cov_upd = torch.linalg.inv(cov_upd_inv)
    temp = cov_upd @ cov_upd_inv
    mean_upd = temp @ predict_mean + 2 * learning_rate * temp @ update_term
    cov_upd_diag = torch.diag(torch.diag(cov_upd))

    return mean_upd, cov_upd_diag



def Update_BOG_lin_full_OU_reparm(mean, Cov, model, obs, y, learning_rate, R, gamma, initial_Cov, init_mean):
    """
    fn to compute BOG update  with linearization,rparametrization and full covariance matrix
    prediction takes action according OU process

    Arg:
    Mean (np.array): Mean vector of the prior distribution
    Cov (np.array): Covariance matrix of the prior distribution
    model (Callable or nn.Module): supply forward function of the model
    obs (np.array): Observation vector
    y (np.array): Actual observation vector
    learning_rate (float): learning rate for the update step
    R (np.array): Observation noise covariance matrix
    gamma (float): OU process parameter
    initial_Cov (np.array): Initial covariance matrix
    init_mean (np.array): Initial mean vector


    Returns:
    Mean_upd (np.array): Mean vector of the posterior distribution
    Cov_upd (np.array): Covariance matrix of the posterior distribution
    """
    predict_mean, predict_cov = predict_full_OU(mean, Cov, gamma, initial_Cov, init_mean)

    y_pred = model(obs)
    H = torch.autograd.functional.jacobian(model, obs)
    R_inv = torch.linalg.inv(R)
    update_term = H.T @ R_inv @ (y - y_pred)
    upd_mean = predict_mean +   learning_rate  @ update_term
    upd_cov = predict_cov -  learning_rate/2 *  H.T @ R_inv @ H
    return upd_mean, upd_cov


def Update_BOG_lin_full_F_reparm(mean, Cov, model, obs, y, learning_rate, R, F, beta, Q):
    """
    fn to compute BOG update  with linearization, reparametrization and full covariance matrix
    prediction takes action according to state transition matrix F

    Arg:
    Mean (np.array): Mean vector of the prior distribution
    Cov (np.array): Covariance matrix of the prior distribution
    model (Callable or nn.Module): supply forward function of the model
    obs (np.array): Observation vector
    y (np.array): Actual observation vector
    learning_rate (float): learning rate for the update step
    R (np.array): Observation noise covariance matrix
    F (np.array): State transition matrix
    beta (float): forgetting factor
    Q (np.array): Process noise covariance matrix


    Returns:
    Mean_upd (np.array): Mean vector of the posterior distribution
    Cov_upd (np.array): Covariance matrix of the posterior distribution
    """
    predict_mean, predict_cov = predict_full_F(mean, Cov, F, beta, Q)

    y_pred = model(obs)
    H = torch.autograd.functional.jacobian(model, obs)
    R_inv = torch.linalg.inv(R)
    update_term = H.T @ R_inv @ (y - y_pred)
    upd_mean = predict_mean + learning_rate @ update_term
    upd_cov = predict_cov - learning_rate / 2 * H.T @ R_inv @ H
    return upd_mean, upd_cov





