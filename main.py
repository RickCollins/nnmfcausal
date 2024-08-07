import torch
import numpy as np
from utils import get_data, get_probabilities, estimate_q_Z_given_A, volume_regularized_nmf
from models import LogisticRegression, COVAR
from sklearn.decomposition import NMF  # Placeholder for volmin factorization
from volmin_nmf import *
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

def main():

    # =============================================================================
    # Debugging
    # =============================================================================

    parameters_debug = True
    step1_debug = True
    step2_debug = True
    step3_debug = False
    step4_debug = False
    step5_debug = False
    step6_debug = False
    testing = False
    parameter_tuning = False
    np.random.seed(1)
    torch.manual_seed(0)

    # =============================================================================
    # Parameters
    # =============================================================================
    p_source = 0.8
    p_target = 0.2
    total = 10
    factorisation_atol = 1e-1

    # Step 2 parameters
    specific_a_index = 0  # First value of A

    # "sklearn" for sklearn's NMF
    # "volmin_1" for volmin NMF, adapted from https://github.com/kharchenkolab/vrnmf
    # "volmin_2" for volmin NMF, adapted from https://github.com/bm424/mvcnmf/blob/master/mvcnmf.py
    nmf_method = "volmin_2" 
    # parameters for volmin NMF
    w_vol = 5
    delta = 1e-8
    n_iter = 100000
    err_cut = 1e-8

    # Dummy theta matrices for example
    #theta_xz = torch.tensor([
    #    [1.0, 0.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0, 0.0],
    #     [0.0, 0.0, 1.0, 0.0],
    #     [0.0, 0.0, 0.0, 1.0] 
    # ])

    theta_xz = torch.tensor([
        [0.5, 0.2, 0.2, 0.1],
        [0.1, 0.5, 0.2, 0.2],
        [0.2, 0.1, 0.5, 0.2],
        [0.2, 0.2, 0.1, 0.5]
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
    Z_source, epsilon_source, W_source, A_source, Y_source = source_data
    Z_target, epsilon_target, W_target, A_target, Y_target = target_data
    if parameters_debug:
        print("Z_source shape:", Z_source.shape)
        print("A_source shape:", A_source.shape)
        print("W_source shape:", W_source.shape)
        print("epsilon_source shape:", epsilon_source.shape)
        print("Y_source shape:", Y_source.shape)
        print("Z_source:", Z_source)
        print("A_source:", A_source)
        print("W_source:", W_source)
        print("epsilon_source:", epsilon_source)
        print("Y_source:", Y_source)

    
    num_classes_Y = Y_source.shape[1]
    num_classes_W = W_source.shape[1]
    num_features_Z = Z_source.shape[1]
    num_features_A = A_source.shape[1]

    sum_epsilon = np.sum(epsilon_source)
    print("Sum of epsilon_source:", sum_epsilon.item())

    # =============================================================================
    # Step 1: Estimate p(Y|Z,a) and p(W|Z,a)
    # =============================================================================

    # Train model to estimate p(Y|Z,a)
    # By stacking with A, we condition on A by including all values of A in the input
    ZA_source = np.hstack((Z_source, A_source))  # We go from 4 features to 4 + 4 = 8 features
    if step1_debug:
        print("ZA_source.shape", ZA_source.shape)  # Debug print statement
        print("ZA_source", ZA_source)  # Debug print statement

    ############### LOGISTIC REGRESSION VERSION ###############

    model_Y = LogisticRegression(input_dim=ZA_source.shape[1], num_classes=Y_source.shape[1])
    model_Y.train(torch.tensor(ZA_source, dtype=torch.float32), torch.tensor(Y_source, dtype=torch.float32))
    p_Y_given_ZA = get_probabilities(model_Y, Z_source, A_source)

    ############### SKLEARN VERSION ###############

    # model_Y = SklearnLogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
    # model_Y.fit(ZA_source, np.argmax(Y_source, axis=1))
    # Y_train_pred = model_Y.predict(ZA_source)
    # Y_train_true = np.argmax(Y_source, axis=1)
    # accuracy_Y_train = np.mean(Y_train_pred == Y_train_true)
    # print(f"Accuracy of model_Y on training set: {accuracy_Y_train:.4f}")
    # p_Y_given_ZA = model_Y.predict_proba(ZA_source).reshape(num_features_Z, num_features_A, num_classes_Y)

    if step1_debug:
        print("p_Y_given_ZA", p_Y_given_ZA)
        print("p_Y_given_ZA shape:", p_Y_given_ZA.shape)  # Debug print statement

    # Verify the shape of p_Y_given_ZA
    assert p_Y_given_ZA.shape == (num_features_Z, num_features_A, num_classes_Y), f"p_Y_given_ZA shape mismatch: {p_Y_given_ZA.shape}"
    assert np.allclose(p_Y_given_ZA.sum(axis=2), 1.0), "p_Y_given_ZA rows do not sum to 1"
    if step1_debug:
        print("Step 1: p_Y_given_ZA shape and sum are correct.")

    ############### LOGISTIC REGRESSION VERSION ###############
    #Train model to estimate p(W|Z,a)
    model_W = LogisticRegression(input_dim=ZA_source.shape[1], num_classes=W_source.shape[1])
    model_W.train(torch.tensor(ZA_source, dtype=torch.float32), torch.tensor(W_source, dtype=torch.float32))
    
    p_W_given_ZA = get_probabilities(model_W, Z_source, A_source)

    ############### SKLEARN VERSION ###############

    # model_W = SklearnLogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
    # model_W.fit(ZA_source, np.argmax(W_source, axis=1))
    # W_train_pred = model_W.predict(ZA_source)
    # W_train_true = np.argmax(W_source, axis=1)
    # accuracy_W_train = np.mean(W_train_pred == W_train_true)
    # print(f"Accuracy of model_W on training set: {accuracy_W_train:.4f}")
    # p_W_given_ZA = model_W.predict_proba(ZA_source).reshape(num_features_Z, num_features_A, num_classes_W)

    if step1_debug:
        print("p_W_given_ZA shape:", p_W_given_ZA.shape)  # Debug print statement
        print("p_W_given_ZA", p_W_given_ZA)  # Debug print statement

    # Verify the shape of p_W_given_ZA
    assert p_W_given_ZA.shape == (num_features_Z, num_features_A, num_classes_W), f"p_W_given_ZA shape mismatch: {p_W_given_ZA.shape}"
    assert np.allclose(p_W_given_ZA.sum(axis=2), 1.0), "p_W_given_ZA rows do not sum to 1"
    if step1_debug:
        print("Step 1: p_W_given_ZA shape and sum are correct.")

    
    accuracy_Y = model_Y.eval(torch.tensor(ZA_source, dtype=torch.float32), torch.tensor(Y_source, dtype=torch.float32))
    accuracy_W = model_W.eval(torch.tensor(ZA_source, dtype=torch.float32), torch.tensor(W_source, dtype=torch.float32))
    print(f"Accuracy of model_Y: {accuracy_Y}")
    print(f"Accuracy of model_W: {accuracy_W}")

    print("STEP 1 DONE")

    # =============================================================================
    # Step 2: Factorize[p(Y|Z,a); p(W|Z,a)] into [p(Y| \Epsilon,a); p(W| \Epsilon)] and p(\Epsilon | Z,a), using volmin
    # =============================================================================

    # extract the matrices from the previous outputs p_Y_given_ZA and p_W_given_ZA.
    # We do this for a specific value of a, so we can just take the first value of A for now. 
    # specific_a_index is a parameter that can be changed at the top to get the factorization for a different value of a.

    p_Y_given_ZA_matrix = p_Y_given_ZA[:, specific_a_index, :]
    p_W_given_ZA_matrix = p_W_given_ZA[:, specific_a_index, :]
    #p_Y_given_ZA_matrix = p_Y_given_ZA.reshape(num_features_Z, num_classes_Y) 
    #p_W_given_ZA_matrix = p_W_given_ZA.reshape(num_features_Z, num_classes_W) 
    if step2_debug:
        print("p_Y_given_ZA_matrix shape:", p_Y_given_ZA_matrix.shape)  # Debug print statement
        print("p_W_given_ZA_matrix shape:", p_W_given_ZA_matrix.shape)  # Debug print statement
        print("p_Y_given_ZA_matrix:", p_Y_given_ZA_matrix)  # Debug print statement
        print("p_W_given_ZA_matrix:", p_W_given_ZA_matrix)  # Debug print statement
    # Stack the probability matrices
    stacked_matrix = np.vstack((p_Y_given_ZA_matrix, p_W_given_ZA_matrix)) # this should be a |Y| x |Z| matrix stacked on top of a |W| x |Z| matrix (for specific a)
    stacked_matrix = stacked_matrix / stacked_matrix.sum(axis=0, keepdims=True) # Normalise the matrix #CHECK THIS
    if step2_debug:
        print("stack_matrix shape",stacked_matrix.shape)  # Debug print statement # so this is like 10 x 4 and 10 x 4 for a 20 x 4 I think.
        print("stacked_matrix:", stacked_matrix)  # Debug print statement

    # Determine the number of components for epsilon
    num_epsilon = 2 # min(W_source.shape[1], Z_source.shape[1]) # Remember we need this to be less than the min of |W| and |Z|. Consider changing this as a hyperparameter
    if step2_debug:
        print("num_epsilon", num_epsilon)  # Debug print statement

    # Perform NMF factorization (method depends on nmf_method)
    # NMF using sklearn.decomposition.NMF
    if nmf_method == "sklearn":
        nmf = NMF(n_components=num_epsilon, init='random', random_state=0) #n_components is epsilon as it's the inner dimension in the factorisation
        W = nmf.fit_transform(stacked_matrix)
        H = nmf.components_

    # NMF using volmin NMF Method 1
    #elif nmf_method == "volmin_1":
       # W, H = volume_regularized_nmf(stacked_matrix, num_epsilon, w_vol, delta, n_iter, err_cut)

    # NMF using volmin NMF Method 2
    elif nmf_method == "volmin_2":
        W, H = mvc_nmf(stacked_matrix.T, num_epsilon, w_vol, n_iter, err_cut) # Transpose the matrix to match the input format of the function

    if step2_debug:
        print("W shape:", W.shape)
        print("H shape:", H.shape)
        print("W:", W)
        print("H:", H)
    
    ######### parameter tuning start #########
    if parameter_tuning:
        param_grid = {
        'w_vol': [0.01, 0.1, 0.5, 10],
        'n_iter': [1000, 10000],
        'err_cut': [1e-4, 1e-6, 1e-8]
    }
        for w_vol in param_grid['w_vol']:
            for n_iter in param_grid['n_iter']:
                for err_cut in param_grid['err_cut']:
                    p_Y_given_ZA_matrix = p_Y_given_ZA[:, specific_a_index, :]
                    p_W_given_ZA_matrix = p_W_given_ZA[:, specific_a_index, :]
                    
                    W, H = mvc_nmf(stacked_matrix.T, num_epsilon, w_vol, n_iter, err_cut) # Transpose the matrix to match the input format of the function
                    
                    # Here you can define your metric to compare W with vec1 and vec2
                    # For now, we will just print the matrices
                    print(f"Parameters: w_vol={w_vol}, n_iter={n_iter}, err_cut={err_cut}")
                    print("p_W_given_epsilon:", W[num_classes_Y:, :])

    ######### parameter tuning end #########



    # Extract the factorized matrices
    p_Y_given_epsilon = W[:num_classes_Y, :] # |Y| x |\Epsilon| matrix for specific a, the first num_classes_Y rows #CHECK NUM_CLASSES_Y IS THE ONE
    p_W_given_epsilon = W[num_classes_Y:, :] # |W| x |\Epsilon| matrix for specific a, the rest of the rows #CHECK NUM_CLASSES_Y IS THE ONE
    p_epsilon_given_ZA = H # |\Epsilon| x |Z| matrix for specific a

    if step2_debug:
        # Print shapes to debug
        print("p_Y_given_epsilon shape:", p_Y_given_epsilon.shape)
        print("p_W_given_epsilon shape:", p_W_given_epsilon.shape)
        print("p_epsilon_given_ZA shape:", p_epsilon_given_ZA.shape)
        print("ZA_source shape:", ZA_source.shape)
        print("p_Y_given_epsilon:", p_Y_given_epsilon)
        print("p_W_given_epsilon (should be comparable to vec1 and vec2):", p_W_given_epsilon)
        #print("p_W_given_epsilon marginalised:", np.sum(p_W_given_epsilon, axis=1).reshape(-1, 1))
        # Verify the shapes of the factorized matrices
        assert p_Y_given_epsilon.shape == (num_classes_Y, num_epsilon), f"p_Y_given_epsilon shape mismatch: {p_Y_given_epsilon.shape}" #CHECK NUM_CLASSES_Y IS THE ONE
        assert p_W_given_epsilon.shape == (num_classes_W, num_epsilon), f"p_W_given_epsilon shape mismatch: {p_W_given_epsilon.shape}" #CHECK NUM_CLASSES_W IS THE ONE
        expected_shape = (num_epsilon, num_features_Z)  # Z_source[1] should be 4
        print("expected_shape", expected_shape)
        assert p_epsilon_given_ZA.shape == expected_shape, f"p_epsilon_given_ZA shape mismatch: {p_epsilon_given_ZA.shape}"
        print("Step 2: Factorization shapes are correct.")

        # Verify reconstruction
        reconstructed_stacked_matrix = np.dot(W, H)
        if step2_debug:
            print("stacked_matrix:", stacked_matrix)
            print("reconstructed_stacked_matrix:", reconstructed_stacked_matrix)
        assert np.allclose(stacked_matrix, reconstructed_stacked_matrix, atol = factorisation_atol), "Reconstructed matrix is not close to the original"
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
    if step6_debug:
        print("p_Y_given_epsilon shape:", p_Y_given_epsilon.shape)  # Debug print statement
        print("q_epsilon_given_Z_and_A shape:", q_epsilon_given_Z_and_A.shape)  # Debug print statement
        print("q_Z_given_A shape:", q_Z_given_A.shape)  # Debug print statement

    # Reshape q_Z_given_A to ensure correct matrix multiplication
    q_Z_given_A_reshaped = q_Z_given_A.T

    if step6_debug:
        print("q_Z_given_A_reshaped shape:", q_Z_given_A_reshaped.shape)  # Debug print statement

    # Use the components to find q(Y|a)
    q_Y_given_A = p_Y_given_epsilon@(q_epsilon_given_Z_and_A@q_Z_given_A_reshaped)
    if step6_debug:
        print(q_Y_given_A.shape)

    # Normalize the output to ensure the columns sum to 1
    q_Y_given_A_normalized = q_Y_given_A / q_Y_given_A.sum(axis=0, keepdims=True)

    if step6_debug:
        # Verify the shape of q_Y_given_A
        assert q_Y_given_A_normalized.shape == (num_classes_Y, num_features_A), f"q_Y_given_A shape mismatch: {q_Y_given_A_normalized.shape}"
        assert np.allclose(q_Y_given_A_normalized.sum(axis=0), 1.0), "q_Y_given_A columns do not sum to 1"
        print("Step 6: q_Y_given_A shape and sum are correct.")

    print("STEP 6 DONE")

    print("Normalised q(Y|a):")
    print(q_Y_given_A_normalized)

    # =============================================================================
    # Testing
    # =============================================================================

    if testing:
        print("Commence Testing...")

        # Print out some samples to verify
        print("Source Data Samples:")
        print("Z:", Z_source[:5])
        print("A:", A_source[:5])
        print("W:", W_source[:5])
        print("epsilon:", epsilon_source[:5])
        print("Y:", Y_source[:5])
        
        print("\nTarget Data Samples:")
        print("Z:", Z_target[:5])
        print("A:", A_target[:5])
        print("W:", W_target[:5])
        print("epsilon:", epsilon_target[:5])
        print("Y:", Y_target[:5])

        # Print the computed q(Y|a)
        print("\nComputed q(Y|a):")
        print(q_Y_given_A_normalized)

        print("\nTesting Done")

if __name__ == "__main__":
    main()