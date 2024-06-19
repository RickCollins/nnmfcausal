import torch
import numpy as np
from data import get_dataset
from models import LogisticRegression
from sklearn.decomposition import NMF #placeholder for volmin factorization


# Dummy theta matrices for example
theta_xz = torch.tensor([
    [3.9, 0.1, 0.05, 0.],
    [0.2, 3.1, 0.05, 0.],
    [0., 0.1, 3.85, 0.],
    [0.19, 0.1, 0.05, 3.99],
])

theta_yx = torch.tensor([
    [1.9, 0.1, 0.05, 0.],
    [0.2, 2.1, 0.05, 0.],
    [0., 0.1, 1.85, 0.],
    [0.19, 0.1, 0.05, 1.99],
])

theta_yw = torch.tensor([
    [2.9, 0.1, 0.05, 0.],
    [0.2, 3.1, 0.05, 0.],
    [0., 0.1, 2.85, 0.],
    [0.19, 0.1, 0.05, 2.99],
])

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

# def train_and_estimate_p(Z, A, Y, model_class=LogisticRegression):
#     """
#     Trains a model to estimate the distribution p(Y|Z, A) or p(W|Z, A).
    
#     Args:
#         Z (numpy.ndarray): Feature matrix Z.
#         A (numpy.ndarray): Feature matrix A.
#         Y (numpy.ndarray): Target matrix Y.
#         model_class (class): The model class to use for training. Defaults to LogisticRegression.

#     Returns:
#         model: Trained model.
#     """
#     ZA = np.hstack((Z, A))
#     model = model_class(input_dim=ZA.shape[1], num_classes=Y.shape[1])
#     model.train(torch.tensor(ZA, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
#     return model

def get_probabilities(model, Z, A):
    """
    Computes the softmax probabilities from the trained model.
    
    Args:
        model (model): Trained model.
        Z (numpy.ndarray): Feature matrix Z.
        A (numpy.ndarray): Feature matrix A.

    Returns:
        numpy.ndarray: Probability matrix.
    """
    ZA = np.hstack((Z, A))
    ZA_tensor = torch.tensor(ZA, dtype=torch.float32)
    with torch.no_grad():
        probs = model(ZA_tensor)
        probs = torch.softmax(probs, dim=1).numpy()
    return probs

# def factorize(prob_matrix, n_components):
#     """
#     Factorizes the probability matrix using Non-negative Matrix Factorization (NMF).
    
#     Args:
#         prob_matrix (numpy.ndarray): Probability matrix to factorize.
#         n_components (int): Number of components for factorization.

#     Returns:
#         tuple: W and H matrices from the NMF factorization.
#     """
#     nmf = NMF(n_components=n_components, init='random', random_state=0)
#     W = nmf.fit_transform(prob_matrix)
#     H = nmf.components_
#     return W, H

# def estimate_q_W_given_ZA(Z, A, W, model_class=LogisticRegression):
#     # use the same model class as the one used to estimate p(W|Z,A) as default, train and estimate p(W|Z,A)
#     return train_and_estimate_p(Z, A, W, model_class)

# linear solve for Q(Epsilon | Z, A) using linalg.lstsq
def solve_for_q_epsilon_given_ZA(p_W_given_Epsilon, q_W_given_ZA):
    p_W_given_Epsilon = p_W_given_Epsilon.T
    q_W_given_ZA = q_W_given_ZA.T
    q_epsilon_given_Z_and_A, _, _, _ = np.linalg.lstsq(p_W_given_Epsilon, q_W_given_ZA, rcond=None)
    return q_epsilon_given_Z_and_A.T

# # Estimate q(Z|a) with standard supervised learning again
# def estimate_q_Z_given_A(Z, A, model_class=LogisticRegression):
#     return train_and_estimate_p(A, Z, model_class)

def main():

    # =============================================================================
    # Parameters
    # =============================================================================
    p_source = 0.98
    p_target = 0.2
    total = 100

    source_data, target_data = get_data(theta_xz, theta_yx, theta_yw, p_source, p_target, total)
    Z_source, U_source, W_source, X_source, Y_source = source_data
    Z_target, U_target, W_target, X_target, Y_target = target_data

    # =============================================================================
    # Step 1: Estimate p(Y|Z,a) and p(W|Z,a)
    # =============================================================================

    # Train model to estimate p(Y|Z,a)
    ZA_source = np.hstack((Z_source, U_source)) # we go from 4 features to 4 + 1 = 5 features
    # print("ZA SOURCE", ZA_source)
    # print("ZA SOURCE SHAPE", ZA_source.shape)
    # print("Z SOURCE SHAPE", Z_source.shape)
    # print("U SOURCE SHAPE", U_source.shape)
    model_Y = LogisticRegression(input_dim=ZA_source.shape[1], num_classes=Y_source.shape[1])
    model_Y.train(torch.tensor(ZA_source, dtype=torch.float32), torch.tensor(Y_source, dtype=torch.float32))
    p_Y_given_ZA = get_probabilities(model_Y, Z_source, U_source)

    # Train model to estimate p(W|Z,a)
    ZA_target = np.hstack((Z_target, U_target))
    model_W = LogisticRegression(input_dim=ZA_source.shape[1], num_classes=W_source.shape[1])
    model_W.train(torch.tensor(ZA_source, dtype=torch.float32), torch.tensor(W_source, dtype=torch.float32))
    p_W_given_ZA = get_probabilities(model_W, Z_source, U_source)

    # =============================================================================
    # Step 2: Factorize[p(Y|Z,a); p(W|Z,a)] into [p(Y| \Epsilon,a); p(W| \Epsilon)] and p(\Epsilon | Z,a), using volmin
    # =============================================================================

    # Stack the probability matrices
    stacked_matrix = np.vstack((p_Y_given_ZA, p_W_given_ZA))

    # Determine the number of components for epsilon
    num_epsilon = min(W_source.shape[1], Z_source.shape[1])

    # Perform NMF factorization (Placeholder for volmin factorization)
    nmf = NMF(n_components=num_epsilon, init='random', random_state=0)
    W = nmf.fit_transform(stacked_matrix)
    H = nmf.components_

    # Extract the factorized matrices
    p_Y_given_epsilon = W[:Y_source.shape[1], :]
    p_W_given_epsilon = W[Y_source.shape[1]:, :]
    p_epsilon_given_ZA = H

    # =============================================================================
    # Step 3: Estimate q(W|Z,a)
    # =============================================================================

    # Train model to estimate q(W|Z,a) using the target data
    ZA_target = np.hstack((Z_target, U_target))
    model_q_W = LogisticRegression(input_dim=ZA_target.shape[1], num_classes=W_target.shape[1])
    model_q_W.train(torch.tensor(ZA_target, dtype=torch.float32), torch.tensor(W_target, dtype=torch.float32))
    q_W_given_ZA = get_probabilities(model_q_W, Z_target, U_target)

    # =============================================================================
    # Step 4: Linear solve for q(epsilon|Z,a)
    # =============================================================================
    # Solves for the distribution q(ε|Z, A) using least squares.
    p_W_given_epsilon = p_W_given_epsilon.T #from step 2
    q_W_given_ZA = q_W_given_ZA.T # from step 3
    q_epsilon_given_Z_and_A, _, _, _ = np.linalg.lstsq(p_W_given_epsilon, q_W_given_ZA, rcond=None) #solve by least squares
    q_epsilon_given_Z_and_A = q_epsilon_given_Z_and_A.T

    # =============================================================================
    # Step 5: Estimate vector q(Z|a)
    # =============================================================================
    # Train model to estimate q(Z|a) using the target data
    model_q_Z = LogisticRegression(input_dim=W_target.shape[1], num_classes=Z_target.shape[1])
    model_q_Z.train(torch.tensor(W_target, dtype=torch.float32), torch.tensor(Z_target, dtype=torch.float32))
    q_Z_given_A = get_probabilities(model_q_Z, W_target, W_target)

    # =============================================================================
    # Step 6: Compute q(Y|a)
    # =============================================================================
    # Use the components to find q(Y|a)
    q_Y_given_A = np.dot(np.dot(p_Y_given_epsilon, q_epsilon_given_Z_and_A), q_Z_given_A)

    # Print out some samples to verify
    print("Source Data Samples:")
    print("Z:", Z_source[:5])
    print("U:", U_source[:5])
    print("W:", W_source[:5])
    print("X:", X_source[:5])
    print("Y:", Y_source[:5])
    
    print("\nTarget Data Samples:")
    print("Z:", Z_target[:5])
    print("U:", U_target[:5])
    print("W:", W_target[:5])
    print("X:", X_target[:5])
    print("Y:", Y_target[:5])

    # Print the computed q(Y|a)
    print("\nComputed q(Y|a):")
    print(q_Y_given_A)

if __name__ == "__main__":
    main()