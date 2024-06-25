import torch
import numpy as np
from utils import get_data, get_probabilities, solve_for_q_epsilon_given_ZA, estimate_q_Z_given_A
from models import LogisticRegression
from sklearn.decomposition import NMF  # Placeholder for volmin factorization

def main():
    step1_debug = False
    step2_debug = False
    step3_debug = False
    step4_debug = False
    step5_debug = False
    step6_debug = True

    # =============================================================================
    # Parameters
    # =============================================================================
    p_source = 0.98
    p_target = 0.2
    total = 10
    vec1 = torch.tensor([0.1,0.1,0.4,0.4]) # The probabilities of the four possible values for Z if U = 0 (vec2 if = 1). Just here so we know that Z can be 1-4. 

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
    Z_source, A_source, W_source, epsilon_source, Y_source = source_data
    Z_target, A_target, W_target, epsilon_target, Y_target = target_data
    print("Z_source shape:", Z_source.shape)
    print("A_source shape:", A_source.shape)
    print("W_source shape:", W_source.shape)
    print("epsilon_source shape:", epsilon_source.shape)
    print("Y_source shape:", Y_source.shape)

    num_classes_Y = Y_source.shape[1]
    num_classes_W = W_source.shape[1]
    num_features_Z = Z_source.shape[1]
    num_features_A = A_source.shape[1]

    # =============================================================================
    # Step 1: Estimate p(Y|Z,a) and p(W|Z,a)
    # =============================================================================

    # Train model to estimate p(Y|Z,a)
    # By stacking with A, we condition on A by including all values of A in the input
    ZA_source = np.hstack((Z_source, A_source))  # We go from 4 features to 4 + 1 = 5 features
    if step1_debug:
        print("ZA_source.shape", ZA_source.shape)  # Debug print statement
    model_Y = LogisticRegression(input_dim=ZA_source.shape[1], num_classes=Y_source.shape[1])
    model_Y.train(torch.tensor(ZA_source, dtype=torch.float32), torch.tensor(Y_source, dtype=torch.float32))
    p_Y_given_ZA = get_probabilities(model_Y, Z_source, A_source)
    if step1_debug:
        print("p_Y_given_ZA", p_Y_given_ZA)
        print("p_Y_given_ZA shape:", p_Y_given_ZA.shape)  # Debug print statement

    # Verify the shape of p_Y_given_ZA
    assert p_Y_given_ZA.shape == (num_features_Z, num_features_A, num_classes_Y), f"p_Y_given_ZA shape mismatch: {p_Y_given_ZA.shape}"
    assert np.allclose(p_Y_given_ZA.sum(axis=2), 1.0), "p_Y_given_ZA rows do not sum to 1"
    print("Step 1: p_Y_given_ZA shape and sum are correct.")

    # Train model to estimate p(W|Z,a)
    model_W = LogisticRegression(input_dim=ZA_source.shape[1], num_classes=W_source.shape[1])
    model_W.train(torch.tensor(ZA_source, dtype=torch.float32), torch.tensor(W_source, dtype=torch.float32))
    p_W_given_ZA = get_probabilities(model_W, Z_source, A_source)

    if step1_debug:
        print("p_W_given_ZA shape:", p_W_given_ZA.shape)  # Debug print statement
        print("p_W_given_ZA", p_W_given_ZA)  # Debug print statement

    # Verify the shape of p_W_given_ZA
    assert p_W_given_ZA.shape == (num_features_Z, num_features_A, num_classes_W), f"p_W_given_ZA shape mismatch: {p_W_given_ZA.shape}"
    assert np.allclose(p_W_given_ZA.sum(axis=2), 1.0), "p_W_given_ZA rows do not sum to 1"
    print("Step 1: p_W_given_ZA shape and sum are correct.")

    print("STEP 1 DONE")

    # =============================================================================
    # Step 2: Factorize[p(Y|Z,a); p(W|Z,a)] into [p(Y| \Epsilon,a); p(W| \Epsilon)] and p(\Epsilon | Z,a), using volmin
    # =============================================================================

    # extract the matrices from the previous outputs p_Y_given_ZA and p_W_given_ZA
    p_Y_given_ZA_matrix = p_Y_given_ZA.reshape(num_features_Z, num_classes_Y)
    p_W_given_ZA_matrix = p_W_given_ZA.reshape(num_features_Z, num_classes_W)
    if step2_debug:
        print("p_Y_given_ZA_matrix shape:", p_Y_given_ZA_matrix.shape)  # Debug print statement
        print("p_W_given_ZA_matrix shape:", p_W_given_ZA_matrix.shape)  # Debug print statement
        print("p_Y_given_ZA_matrix:", p_Y_given_ZA_matrix)  # Debug print statement
        print("p_W_given_ZA_matrix:", p_W_given_ZA_matrix)  # Debug print statement
    # Stack the probability matrices
    stacked_matrix = np.vstack((p_Y_given_ZA_matrix, p_W_given_ZA_matrix)) # this should be a |Y| x |Z| matrix stacked on top of a |W| x |Z| matrix (for specific a)
    if step2_debug:
        print("stack_matrix shape",stacked_matrix.shape)  # Debug print statement # so this is like 10 x 4 and 10 x 4 for a 20 x 4 I think.

    # Determine the number of components for epsilon
    num_epsilon = min(W_source.shape[1], Z_source.shape[1]) # Remember we need this to be less than the min of |W| and |Z|. Consider changing this as a hyperparameter
    if step2_debug:
        print("num_epsilon", num_epsilon)  # Debug print statement

    # Perform NMF factorization (Placeholder for volmin factorization)
    nmf = NMF(n_components=num_epsilon, init='random', random_state=0) #n_components is epsilon as it's the inner dimension in the factorisation
    W = nmf.fit_transform(stacked_matrix)
    H = nmf.components_

    if step2_debug:
        print("W shape:", W.shape)
        print("H shape:", H.shape)
        print("W:", W)
        print("H:", H)

    # Extract the factorized matrices
    p_Y_given_epsilon = W[:num_classes_Y, :] # |Y| x |\Epsilon| matrix for specific a #CHECK NUM_CLASSES_Y IS THE ONE
    p_W_given_epsilon = W[num_classes_Y:, :] # |W| x |\Epsilon| matrix for specific a #CHECK NUM_CLASSES_Y IS THE ONE
    p_epsilon_given_ZA = H # |\Epsilon| x |Z| matrix for specific a

    if step2_debug:
        # Print shapes to debug
        print("p_Y_given_epsilon shape:", p_Y_given_epsilon.shape)
        print("p_W_given_epsilon shape:", p_W_given_epsilon.shape)
        print("p_epsilon_given_ZA shape:", p_epsilon_given_ZA.shape)
        print("ZA_source shape:", ZA_source.shape)
        # Verify the shapes of the factorized matrices
        assert p_Y_given_epsilon.shape == (num_classes_Y, num_epsilon), f"p_Y_given_epsilon shape mismatch: {p_Y_given_epsilon.shape}" #CHECK NUM_CLASSES_Y IS THE ONE
        assert p_W_given_epsilon.shape == (num_classes_W, num_epsilon), f"p_W_given_epsilon shape mismatch: {p_W_given_epsilon.shape}" #CHECK NUM_CLASSES_W IS THE ONE
        expected_shape = (num_epsilon, num_features_Z)  # Z_source[1] should be 4
        print("expected_shape", expected_shape)
        assert p_epsilon_given_ZA.shape == expected_shape, f"p_epsilon_given_ZA shape mismatch: {p_epsilon_given_ZA.shape}"
        print("Step 2: Factorization shapes are correct.")

        # Verify reconstruction
        reconstructed_stacked_matrix = np.dot(W, H)
        assert np.allclose(stacked_matrix, reconstructed_stacked_matrix, atol=1e-2), "Reconstructed matrix is not close to the original"
        print("Step 2: Reconstruction is correct.")

    print("STEP 2 DONE")

    # =============================================================================
    # Step 3: Estimate q(W|Z,a)
    # =============================================================================

    # Train model to estimate q(W|Z,a) using the target data
    ZA_target = np.hstack((Z_target, A_target))
    model_q_W = LogisticRegression(input_dim=ZA_target.shape[1], num_classes=W_target.shape[1])
    model_q_W.train(torch.tensor(ZA_target, dtype=torch.float32), torch.tensor(W_target, dtype=torch.float32))
    q_W_given_ZA = get_probabilities(model_q_W, Z_target, A_target)

    if step3_debug:
        # Verify the shape of q_W_given_ZA
        assert q_W_given_ZA.shape == (num_features_Z, num_features_A, num_classes_W), f"q_W_given_ZA shape mismatch: {q_W_given_ZA.shape}"
        assert np.allclose(q_W_given_ZA.sum(axis=2), 1.0), "q_W_given_ZA rows do not sum to 1"
        print("Step 3: q_W_given_ZA shape and sum are correct.")

    print("STEP 3 DONE")

    # =============================================================================
    # Step 4: Linear solve for q(epsilon|Z,a)
    # =============================================================================

    # extract matrix from output of step 3: q_W_given_ZA
    q_W_given_ZA_matrix = q_W_given_ZA.reshape(num_features_Z * num_features_A, num_classes_W) #TODO: check inside the reshape here. Product?????

    if step4_debug:
        print("q_W_given_ZA_matrix shape:", q_W_given_ZA_matrix.shape)

    # Solves for the distribution q(ε|Z, A) using least squares.
    if step4_debug:
        print("p_W_given_epsilon shape before transpose:", p_W_given_epsilon.shape)  # Debug

    p_W_given_epsilon = p_W_given_epsilon.T  # From step 2 # |\Epsilon| x |W| matrix for specific a
    q_W_given_ZA_matrix = q_W_given_ZA_matrix.T  # From step 3

    if step4_debug:
        print("p_W_given_epsilon shape after transpose:", p_W_given_epsilon.shape)  # Debug
        print("q_W_given_ZA shape after transpose:", q_W_given_ZA_matrix.shape)  # Debug

    q_epsilon_given_Z_and_A, _, _, _ = np.linalg.lstsq(p_W_given_epsilon, q_W_given_ZA_matrix, rcond=None)  # Solve by least squares
    q_epsilon_given_Z_and_A = q_epsilon_given_Z_and_A.T

    if step4_debug:
        print("q_epsilon_given_Z_and_A shape after lstsq:", q_epsilon_given_Z_and_A.shape)  # Debug

    if step4_debug:
        # Verify the shape of q_epsilon_given_Z_and_A
        assert q_epsilon_given_Z_and_A.shape == (num_epsilon, q_W_given_ZA_matrix.shape[1]), f"q_epsilon_given_Z_and_A shape mismatch: {q_epsilon_given_Z_and_A.shape}"
        print("Step 4: q_epsilon_given_Z_and_A shape is correct.")

    print("STEP 4 DONE")


    # =============================================================================
    # Step 5: Estimate vector q(Z|a)
    # =============================================================================
    if step5_debug:
        print("A_target shape:", A_target.shape)  # Debug print statement

    # Train model to estimate q(Z|a)
    model_q_Z = LogisticRegression(input_dim=A_target.shape[1], num_classes=Z_target.shape[1])
    model_q_Z.train(torch.tensor(A_target, dtype=torch.float32), torch.tensor(Z_target, dtype=torch.float32))
    q_Z_given_A = estimate_q_Z_given_A(model_q_Z, A_target, num_features_Z, num_features_A)

    if step5_debug:
        # Verify the shape of q_Z_given_A
        assert q_Z_given_A.shape == (num_features_A, num_features_Z), f"q_Z_given_A shape mismatch: {q_Z_given_A.shape}"
        assert np.allclose(q_Z_given_A.sum(axis=1), 1.0), "q_Z_given_A rows do not sum to 1"
        print("Step 5: q_Z_given_A shape and sum are correct.")

    print("STEP 5 DONE")


    # =============================================================================
    # Step 6: Compute q(Y|a)
    # =============================================================================
    # Use the components to find q(Y|a)
    print("p_Y_given_epsilon shape:", p_Y_given_epsilon.shape)  # Debug print statement
    print("q_epsilon_given_Z_and_A shape:", q_epsilon_given_Z_and_A.shape)  # Debug print statement
    print("q_Z_given_A shape:", q_Z_given_A.shape)  # Debug print statement

    # Reshape q_Z_given_A to ensure correct matrix multiplication
    q_Z_given_A_reshaped = q_Z_given_A.T

    if step6_debug:
        print("q_Z_given_A_reshaped shape:", q_Z_given_A_reshaped.shape)  # Debug print statement

    # Use the components to find q(Y|a)
    q_Y_given_A = p_Y_given_epsilon@(q_epsilon_given_Z_and_A@q_Z_given_A_reshaped)
    print(q_Y_given_A.shape)

    # Normalize the output to ensure the columns sum to 1
    q_Y_given_A_normalized = q_Y_given_A / q_Y_given_A.sum(axis=0, keepdims=True)

    if step6_debug:
        # Verify the shape of q_Y_given_A
        assert q_Y_given_A_normalized.shape == (num_classes_Y, num_features_A), f"q_Y_given_A shape mismatch: {q_Y_given_A_normalized.shape}"
        assert np.allclose(q_Y_given_A_normalized.sum(axis=0), 1.0), "q_Y_given_A columns do not sum to 1"
        print("Step 6: q_Y_given_A shape and sum are correct.")

    print("Step 6 done")

    print(q_Y_given_A_normalized)

    # =============================================================================
    # Testing
    # =============================================================================

    # print("Commence Testing...")

    # # Print out some samples to verify
    # print("Source Data Samples:")
    # print("Z:", Z_source[:5])
    # print("A:", A_source[:5])
    # print("W:", W_source[:5])
    # print("epsilon:", epsilon_source[:5])
    # print("Y:", Y_source[:5])
    
    # print("\nTarget Data Samples:")
    # print("Z:", Z_target[:5])
    # print("A:", A_target[:5])
    # print("W:", W_target[:5])
    # print("epsilon:", epsilon_target[:5])
    # print("Y:", Y_target[:5])

    # # Print the computed q(Y|a)
    # print("\nComputed q(Y|a):")
    # print(q_Y_given_A)

    # print("\nTesting Done")

if __name__ == "__main__":
    main()