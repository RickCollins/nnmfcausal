import torch
from data import get_dataset
from models import Oracle, LogisticRegression, COVAR, MFReg, MFRegGrad

# theta_xz,theta_yx,theta_yw = torch.rand(4,4),torch.rand(4,4),torch.rand(4,4)

theta_xz = torch.tensor([
    [3.9,0.1,0.05,0.],
    [0.2,3.1,0.05,0.],
    [0.,0.1,3.85,0.],
    [0.19,0.1,0.05,3.99],
])

theta_yx = torch.tensor([
    [1.9,0.1,0.05,0.],
    [0.2,2.1,0.05,0.],
    [0.,0.1,1.85,0.],
    [0.19,0.1,0.05,1.99],
])

theta_yw = torch.tensor([
    [2.9,0.1,0.05,0.],
    [0.2,3.1,0.05,0.],
    [0.,0.1,2.85,0.],
    [0.19,0.1,0.05,2.99],
])

if __name__=="__main__":
    DEVICE='cpu'
    total = 100000
    p1,p2 = 0.98,0.2
    U,Z,W,X,Y = get_dataset(theta_xz,theta_yx,theta_yw,p=p1,total=total)
    U_test,Z_test,W_test,X_test,Y_test = get_dataset(theta_xz,theta_yx,theta_yw,p=p2,total=total)
    U_adapt,Z_adapt,W_adapt,X_adapt,Y_adapt = get_dataset(theta_xz,theta_yx,theta_yw,p=p2,total=total)

    oracle = Oracle()
    acc1 = oracle.eval(X,W,U,Y,theta_yx,theta_yw)
    acc2 = oracle.eval(X_test,W_test,U_test,Y_test,theta_yx,theta_yw)
    print(f"Oracle | Train acc:{acc1*100}% | Test acc:{acc2*100}%")

    logreg = LogisticRegression(4,4)
    logreg.train(X, Y, device=DEVICE)
    acc1 = logreg.eval(X, Y)
    acc2 = logreg.eval(X_test, Y_test)
    print(f"Logreg | Train acc:{acc1*100}% | Test acc:{acc2*100}%")

    covar = COVAR(4,4)
    covar.train(X, Y, X_adapt, device=DEVICE)
    acc1 = covar.eval(X, Y)
    acc2 = covar.eval(X_test, Y_test)
    print(f"COVAR | Train acc:{acc1*100}% | Test acc:{acc2*100}%")

    # mfreg = MFRegGrad(4, 4, 4, 4, 2)
    # mfreg.train(X, Y, Z, W, learning_rate=0.1, epochs=100, batch_size=32, verbose=False, device=DEVICE)
    # mfreg.adapt(X_adapt, W_adapt, learning_rate=0.1, epochs=100, batch_size=32, verbose=False, device=DEVICE)
    # acc2 = mfreg.eval_test(X_test, Y_test)
    # accu = mfreg.eval_u(X_test, U_test)
    # print(f"MFreg (gradient) | Test acc:{acc2*100}% | U acc:{accu*100}%")

    mfreg = MFReg(4, 4, 4, 4, 2)
    mfreg.train(X, Y, Z, W, device=DEVICE)
    mfreg.adapt(X_adapt, W_adapt, device=DEVICE)
    acc1 = mfreg.eval_train(X_test, Y_test)
    acc2 = mfreg.eval_test(X_test, Y_test)
    accu = mfreg.eval_u(X_test, U_test)
    accw = mfreg.eval_w(X_test, W_test)
    print(f"MFreg | Train acc:{acc1*100}%  | Test acc:{acc2*100}% | U acc:{accu*100}% | W acc:{accw*100}%")