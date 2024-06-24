import torch
import numpy as np
from data import get_dataset

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

# # Estimate q(Z|a) with standard supervised learning again
# def estimate_q_Z_given_A(Z, A, model_class=LogisticRegression):
#     return train_and_estimate_p(A, Z, model_class)