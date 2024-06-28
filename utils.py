import torch
import numpy as np
from data import get_dataset
from scipy.linalg import sqrtm, det, pinv
from sklearn.decomposition import NMF

def get_data(theta_xz, theta_yx, theta_yw, p_source, p_target, total):
    """
    Generates datasets for source and target distributions.
    
    Args:
        theta_xz (torch.Tensor): Transformation matrix for X given Z.
        theta_yx (torch.Tensor): Transformation matrix for Y given X.
        theta_yw (torch.Tensor): Transformation matrix for Y given W.
        p_source (float): Probability parameter for the source distribution.
        p_target (float): Probability parameter for the target distribution.
        total (int): Total number of samples to generate.

    Returns:
        tuple: Two tuples containing the source and target datasets respectively.
    """
    # Source distribution data
    U_source, Z_source, W_source, X_source, Y_source = get_dataset(theta_xz, theta_yx, theta_yw, p_source, total)
    
    # Target distribution data
    U_target, Z_target, W_target, X_target, Y_target = get_dataset(theta_xz, theta_yx, theta_yw, p_target, total)
    
    return (Z_source.numpy(), U_source.numpy(), W_source.numpy(), X_source.numpy(), Y_source.numpy()), \
           (Z_target.numpy(), U_target.numpy(), W_target.numpy(), X_target.numpy(), Y_target.numpy())


def get_probabilities(model, Z, A):
    """
    Computes the softmax probabilities from the trained model.
    
    Args:
        model (model): Trained model.
        Z (numpy.ndarray): Feature matrix Z.
        A (numpy.ndarray): Feature matrix A.

    Returns:
        numpy.ndarray: Probability matrix reshaped to (|Z|, |A|, |Y|) or (|Z|, |A|, |W|).
    """
    num_Z = Z.shape[1]
    num_A = A.shape[1]
    num_classes = model.linear.out_features

    # Generate all possible one-hot vectors for Z
    possible_Z = np.eye(num_Z)
    possible_A = np.eye(num_A)
    
    probabilities = []
    
    for z in possible_Z:
        for a in possible_A:
            ZA = np.hstack((z.reshape(1, -1), a.reshape(1, -1)))
            ZA_tensor = torch.tensor(ZA, dtype=torch.float32)
            with torch.no_grad():
                probs = model(ZA_tensor)
                probs = torch.softmax(probs, dim=1).numpy()
                probabilities.append(probs)
    
    probabilities = np.array(probabilities).reshape((num_Z, num_A, num_classes))
    
    return probabilities

def estimate_q_Z_given_A(model, A, num_classes_Z, num_features_A):
    """
    Estimate q(Z|a) using the given model.
    
    Args:
        model (model): Trained model.
        A (numpy.ndarray): Feature matrix A.
        num_classes_Z (int): Number of classes for Z.
        num_features_A (int): Number of features for A.

    Returns:
        numpy.ndarray: Estimated q(Z|a) matrix.
    """
    num_A = A.shape[1]
    num_classes = model.linear.out_features

    # Generate all possible one-hot vectors for A
    possible_A = np.eye(num_A)
    
    probabilities = []
    
    for a in possible_A:
        A_sample = a.reshape(1, -1)
        A_tensor = torch.tensor(A_sample, dtype=torch.float32)
        with torch.no_grad():
            probs = model(A_tensor)
            probs = torch.softmax(probs, dim=1).numpy()
            probabilities.append(probs)
    
    probabilities = np.array(probabilities).reshape((num_features_A, num_classes_Z))
    return probabilities

# linear solve for Q(Epsilon | Z, A) using linalg.lstsq
def solve_for_q_epsilon_given_ZA(p_W_given_epsilon, q_W_given_ZA):
    print("p_W_given_epsilon shape before transpose:", p_W_given_epsilon.shape)  # Debug
    print("q_W_given_ZA shape before transpose:", q_W_given_ZA.shape)  # Debug
    p_W_given_epsilon = p_W_given_epsilon.T
    q_W_given_ZA = q_W_given_ZA.T
    print("p_W_given_epsilon shape after transpose:", p_W_given_epsilon.shape)  # Debug
    print("q_W_given_ZA shape after transpose:", q_W_given_ZA.shape)  # Debug
    
    # Check dimensions and adjust if necessary
    num_ZA = p_W_given_epsilon.shape[1]
    num_epsilon = p_W_given_epsilon.shape[0]
    num_W = q_W_given_ZA.shape[1]
    
    if q_W_given_ZA.shape[0] != num_ZA:
        raise ValueError("Dimensions of q_W_given_ZA and p_W_given_epsilon do not match after transpose.")
    
    # Solve by least squares
    q_epsilon_given_Z_and_A, _, _, _ = np.linalg.lstsq(p_W_given_epsilon, q_W_given_ZA, rcond=None)
    print("q_epsilon_given_Z_and_A shape after lstsq:", q_epsilon_given_Z_and_A.shape)  # Debug
    return q_epsilon_given_Z_and_A.T


def volume_regularized_nmf(B, n_components, w_vol=0.1, delta=1e-8, n_iter=1000, err_cut=1e-8):
    """
    Volume-regularized NMF implementation in Python.
    
    Args:
        B (np.ndarray): Input matrix to factorize.
        n_components (int): Number of components for factorization.
        w_vol (float): Weight for volume regularization.
        delta (float): Regularization term for logdet.
        n_iter (int): Number of iterations.
        err_cut (float): Convergence criterion.
        
    Returns:
        tuple: Factorized matrices C, R.
    """
    np.random.seed(0)  # For reproducibility

    # Initialize C and R using standard NMF
    nmf = NMF(n_components=n_components, init='random', random_state=0, max_iter=500)
    C = nmf.fit_transform(B)
    R = nmf.components_

    for iteration in range(n_iter):
        # Update R using volume regularization
        err_prev = np.linalg.norm(B - np.dot(C, R), 'fro')**2

        # Update R with regularization
        R = np.maximum(np.dot(pinv(C), B), 0)  # Standard NMF update
        R = R - w_vol * (np.dot(R, R.T) + delta * np.eye(R.shape[0]))  # Volume regularization update
        R = np.maximum(R, 0)  # Ensure non-negativity

        # Update C to satisfy the simplex constraint
        C = np.maximum(np.dot(B, pinv(R)), 0)
        C = C / C.sum(axis=0)

        err_post = np.linalg.norm(B - np.dot(C, R), 'fro')**2

        # Check for convergence
        if np.abs(err_prev - err_post) / err_prev < err_cut:
            break

    return C, R
