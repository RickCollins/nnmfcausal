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
            print(ZA_tensor)
            with torch.no_grad():
                logits = model(ZA_tensor)
                print(logits)
                prob_1 = torch.sigmoid(logits).numpy()[0][0]  # Probability of class 1
                prob_0 = 1 - prob_1  # Probability of class 0
                print([prob_0, prob_1])
                probabilities.append([prob_0, prob_1])
    
    probabilities = np.array(probabilities).reshape((num_Z, num_A, 2))
    
    return probabilities

def get_probabilities_one_hot(model, Z, A):
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
                probabilities.append(probs[0])
    
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
    #np.random.seed(0)  # For reproducibility

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

def volnmf_logdet(C, X, R, R_constraint="pos", majorate=False, extrapolate=True, qmax=100,
                  w_vol=1e-1, delta=1, err_cut=1e-3, n_iter=1000):
    # Initial assignments
    W = np.transpose(R)
    W_update = W.copy()
    H = np.transpose(C)
    FM = np.linalg.inv(np.transpose(W) @ W + delta * np.eye(W.shape[0]))

    # Iteration variables
    iter = 1
    err = 1e5
    q = [1, (1 + np.sqrt(5)) / 5]

    # Main iteration loop
    while err > err_cut and iter < n_iter:
        W_prev = W.copy()
        
        if majorate:
            Y = W_prev
            FM = np.linalg.inv(np.transpose(Y) @ Y + delta * np.eye(Y.shape[0]))
        
        if R_constraint == "pos":
            Lip = np.sqrt(np.sum((H @ H.T + w_vol * FM) ** 2))
            gradF = W_update @ (H @ H.T + w_vol * FM) - X.T @ H
            W = W_update - gradF / Lip
            W[W < 0] = 0
        else:
            W = X.T @ H.T @ np.linalg.inv(H @ H.T + w_vol * FM)

        if extrapolate:
            extr = (q[iter - 1] - 1) / q[iter]
            W_update = W + extr * (W - W_prev)
        else:
            W_update = W

        # Error calculation and iteration increment
        err = np.sum((W - W_prev) ** 2) / np.sum(W ** 2)
        iter += 1
        q.append(min(qmax, (1 + np.sqrt(1 + 4 * q[iter - 1] ** 2)) / 2))

    return W.T

def volnmf_estimate(B, C, R, Q, domain="covariance", volf='logdet', R_majorate=False,
                    wvol=None, delta=1e-8, n_iter=1000, err_cut=1e-8,
                    vol_iter=100, c_iter=100,
                    extrapolate=True, accelerate=True, acc_C=4/5, acc_R=3/4,
                    C_constraint="col", C_bound=1, R_constraint="pos",
                    verbose=True, record=100, Ctrue=None, mutation_run=False):
    
    # Initialization
    iter = 1
    err = 1e5
    rvol = []
    aff_mean = []
    info_record = []
    eigens = 1
    R_update = R.copy()
    C_update = C.copy()
    tot_update_prev = 0
    tot_update = 0

    while iter < n_iter and err > err_cut:
        # Domain check
        if domain == "covariance":
            X = np.dot(B, Q)
        else:
            X = B

        # Update R
        err_prev = np.sum((X - np.dot(C_update, R))**2)
        if volf == "logdet":
            vol_prev = np.log(np.linalg.det(np.dot(R, R.T) + delta * np.eye(R.shape[0])))
        elif volf == "det":
            vol_prev = np.linalg.det(np.dot(R, R.T))
        R_prev = R.copy()

        # Update R based on the volume function
        if volf == "logdet":
            R = volnmf_logdet(C_update, X, R_update, R_constraint=R_constraint, extrapolate=extrapolate, majorate=R_majorate,
                              w_vol=wvol, delta=delta, err_cut=1e-100, n_iter=vol_iter)
        elif volf == "det":
            R = volnmf_det(C_update, X, R_update, posit=False, w_vol=wvol, eigen_cut=1e-20, err_cut=1e-100, n_iter=vol_iter)

        ### Post-update calculations
        err_post = np.sum((X - np.dot(C_update, R))**2)
        if volf == "logdet":
            vol_post = np.log(np.linalg.det(np.dot(R, R.T) + delta * np.eye(R.shape[0])))
        elif volf == "det":
            vol_post = np.linalg.det(np.dot(R, R.T))
        rvol.append(vol_post)

        ### Update C
        C_prev = C.copy()
        if C_constraint == "col":
            C = volnmf_simplex_col(X, R, C_prev=C_update, bound=C_bound, extrapolate=extrapolate, err_cut=1e-100, n_iter=c_iter)
        else:
            C = volnmf_simplex_row(X, R, C_prev=C_update, meq=1)
        err_post_C = np.sum((X - np.dot(C, R_update))**2)

        # Accelerate C if possible
        if accelerate:
            C_update = C + acc_C * (C - C_prev)
            R_update = R + acc_R * (R - R_prev)

            # Ensure non-negativity
            C_update = np.maximum(C_update, 0)
            R_update = np.maximum(R_update, 0)

            err_update = np.sum((X - np.dot(C, R_update))**2)
            vol_update = np.log(np.linalg.det(np.dot(R_update, R_update.T) + delta * np.eye(R_update.shape[0])))
            tot_update = err_update + wvol * vol_update

            if tot_update > tot_update_prev:
                C_update = C.copy()
                R_update = R.copy()
        else:
            C_update = C.copy()
            R_update = R.copy()

        tot_update_prev = tot_update

        ### optimize Q
        if domain == "covariance":
            Q_init = volnmf_procrustes(np.dot(C_update, R_update), B)

        err = np.sum((C_update - C_prev)**2) / np.sum(C_update**2)
        eigens = np.linalg.eigvals(np.dot(R_update, R_update.T))
        aff = 1

        if Ctrue is not None:
            if C_bound is None:
                aff = np.apply_along_axis(lambda x: np.max(np.abs(np.corrcoef(x, Ctrue))), axis=1, arr=C_update)
            else:
                aff = np.apply_along_axis(lambda x: np.max(np.abs(np.corrcoef(x * C_bound, Ctrue))), axis=1,
                                          arr=C_update)

        aff_mean.append(np.mean(aff))

        iter += 1

    return {'C': C_update, 'R': R_update, 'Q': Q_init, 'iter': iter, 'err': err, 'info_record': info_record}

def volnmf_main(vol, B, volnmf=None, n_comp=3, n_reduce=None,
                do_nmf=True, iter_nmf=100, seed=None,
                domain="covariance", volf='logdet',
                wvol=None, delta=1e-8, n_iter=500, err_cut=1e-16,
                vol_iter=20, c_iter=20,
                extrapolate=True, accelerate=False, acc_C=4/5, acc_R=3/4,
                C_constraint="col", C_bound=1, R_constraint="pos", R_majorate=False,
                C_init=None, R_init=None, Q_init=None, anchor=None, Ctrue=None,
                verbose=True, record=100, verbose_nmf=False, record_nmf=None, mutation_run=False):

    # Initialize Q_init if None
    if Q_init is None:
        Q_init = np.eye(n_reduce, n_comp)  # Identity matrix with n_reduce rows and n_comp columns

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Use NMF to initialize C_init and R_init if they are not provided
    if C_init is None or R_init is None:
        nmf_model = NMF(n_components=n_comp, init='random', random_state=seed)
        C_init = nmf_model.fit_transform(B)
        R_init = nmf_model.components_

    C_rand, R_rand, Q_rand = C_init.copy(), R_init.copy(), Q_init.copy()

    if wvol is None:
        wvol = 0

    # Print message indicating the start of volume-regularized NMF
    print('Run volume-regularized NMF...')

    # Run volume-regularized NMF
    vol_solution = volnmf_estimate(B, C_init, R_init, Q_init,
                                   domain=domain, volf=volf, R_majorate=R_majorate,
                                   wvol=wvol, delta=delta, n_iter=n_iter, err_cut=err_cut,
                                   vol_iter=vol_iter, c_iter=c_iter,
                                   extrapolate=extrapolate, accelerate=accelerate, acc_C=acc_C, acc_R=acc_R,
                                   C_constraint=C_constraint, C_bound=C_bound, R_constraint=R_constraint,
                                   verbose=verbose, record=record, Ctrue=Ctrue, mutation_run=mutation_run)
    
    # Print done message
    print('Done')

    # Return the results
    return {
        'C': vol_solution['C'], 'R': vol_solution['R'], 'Q': vol_solution['Q'],
        'C_init': C_init, 'R_init': R_init, 'Q_init': Q_init,
        'C_rand': C_rand, 'R_rand': R_rand, 'Q_rand': Q_rand,
        'rec': vol_solution['info_record']
    }