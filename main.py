import torch
import numpy as np
from utils import get_data, get_probabilities, solve_for_q_epsilon_given_ZA
from models import LogisticRegression
from sklearn.decomposition import NMF  # Placeholder for volmin factorization

def main():

    # =============================================================================
    # Parameters
    # =============================================================================
    p_source = 0.98
    p_target = 0.2
    total = 10

    # Dummy theta matrices for example
    theta_xz = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    theta_yx = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    theta_yw = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

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

    print("Step 1 done")

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

    print("Step 2 done")

    # =============================================================================
    # Step 3: Estimate q(W|Z,a)
    # =============================================================================

    # Train model to estimate q(W|Z,a) using the target data
    ZA_target = np.hstack((Z_target, U_target))
    model_q_W = LogisticRegression(input_dim=ZA_target.shape[1], num_classes=W_target.shape[1])
    model_q_W.train(torch.tensor(ZA_target, dtype=torch.float32), torch.tensor(W_target, dtype=torch.float32))
    q_W_given_ZA = get_probabilities(model_q_W, Z_target, U_target)

    print("Step 3 done")

    # =============================================================================
    # Step 4: Linear solve for q(epsilon|Z,a)
    # =============================================================================
    # Solves for the distribution q(ε|Z, A) using least squares.
    p_W_given_epsilon = p_W_given_epsilon.T  # from step 2
    q_W_given_ZA = q_W_given_ZA.T  # from step 3

    print("p_W_given_epsilon shape before transpose:", p_W_given_epsilon.shape)  # Debug
    print("q_W_given_ZA shape before transpose:", q_W_given_ZA.shape)  # Debug

    q_epsilon_given_Z_and_A, _, _, _ = np.linalg.lstsq(p_W_given_epsilon, q_W_given_ZA, rcond=None)  # solve by least squares
    q_epsilon_given_Z_and_A = q_epsilon_given_Z_and_A.T

    print("p_W_given_epsilon shape after transpose:", p_W_given_epsilon.shape)  # Debug
    print("q_W_given_ZA shape after transpose:", q_W_given_ZA.shape)  # Debug
    print("q_epsilon_given_Z_and_A shape after lstsq:", q_epsilon_given_Z_and_A.shape)  # Debug


    # =============================================================================
    # Step 5: Estimate vector q(Z|a)
    # =============================================================================
    WA_target = np.hstack((W_target, U_target))
    print("WA_target shape:", WA_target.shape)  # Debug print statement
    print("W_target shape:", W_target.shape)  # Debug print statement
    print("U_target shape:", U_target.shape)  # Debug print statement

    # Train model to estimate q(Z|a)
    model_q_Z = LogisticRegression(input_dim=WA_target.shape[1], num_classes=Z_target.shape[1])
    model_q_Z.train(torch.tensor(WA_target, dtype=torch.float32), torch.tensor(Z_target, dtype=torch.float32))
    q_Z_given_A = get_probabilities(model_q_Z, W_target, U_target)

    print("Step 5 done")


    # =============================================================================
    # Step 6: Compute q(Y|a)
    # =============================================================================
    # Use the components to find q(Y|a)
    print("p_Y_given_epsilon shape:", p_Y_given_epsilon.shape)  # Debug print statement
    print("q_epsilon_given_Z_and_A shape:", q_epsilon_given_Z_and_A.shape)  # Debug print statement
    print("q_Z_given_A shape:", q_Z_given_A.shape)  # Debug print statement

    # Ensure the dimensions match for matrix multiplication
    q_epsilon_given_Z_and_A = q_epsilon_given_Z_and_A[:p_Y_given_epsilon.shape[1], :q_Z_given_A.shape[0]]

    # Use the components to find q(Y|a)
    q_Y_given_A = np.dot(np.dot(p_Y_given_epsilon, q_epsilon_given_Z_and_A), q_Z_given_A)

    print("Step 6 done")


    # =============================================================================
    # Testing
    # =============================================================================

    print("Commence Testing...")

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

    print("\nTesting Done")

if __name__ == "__main__":
    main()