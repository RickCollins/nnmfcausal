import torch

def get_tuple(theta_xz,theta_yx,theta_yw,p=0.2):
    vec1 = torch.tensor([0.1,0.1,0.4,0.4])
    vec2 = torch.tensor([0.4,0.4,0.1,0.1])

    U = torch.bernoulli(torch.tensor([p])).item()

    cat0 = torch.distributions.categorical.Categorical(vec1) 
    cat1 = torch.distributions.categorical.Categorical(vec2) 

    z = torch.nn.functional.one_hot((cat0.sample()*U + cat1.sample()*(1-U)).long(), num_classes = 4).unsqueeze(0).float()
    w = torch.nn.functional.one_hot((cat0.sample()*U + cat1.sample()*(1-U)).long(), num_classes = 4).unsqueeze(0).float()

    probx = (vec1 * U + vec2 * (1-U)).unsqueeze(0)
    proby = (vec1 * U + vec2 * (1-U)).unsqueeze(0)

    probx = probx * (theta_xz @ z.T).T
    #print(probx)
    x = torch.nn.functional.one_hot(torch.distributions.categorical.Categorical(probx).sample().long(), num_classes = 4).float()
    #print(x)
    # x = torch.nn.functional.one_hot(torch.argmax(probx, dim=1).long(), num_classes = 4).float()

    proby = proby * (theta_yx @ x.T + theta_yw @ w.T).T
    y = torch.nn.functional.one_hot(torch.distributions.categorical.Categorical(proby).sample().long(), num_classes = 4).float()
    #y = (proby.sum(dim=1) > 0.5).float().unsqueeze(0)
    #y = torch.nn.functional.one_hot(torch.argmax(proby).long(), num_classes = 4).float()

    return U,z,w,x,y

def get_dataset(theta_xz,theta_yx,theta_yw,p,total):
    U,Z,W,X,Y = [],[],[],[],[]
    for _ in range(total):
        u,z,w,x,y = get_tuple(theta_xz,theta_yx,theta_yw,p)
        U.append(torch.tensor([[u]]))
        Z.append(z)
        W.append(w)
        X.append(x)
        Y.append(y)
    # print(u.shape,z.shape,w.shape,x.shape,y.shape)
    U = torch.cat(U,0)
    Z = torch.cat(Z,0)
    W = torch.cat(W,0)
    X = torch.cat(X,0)
    Y = torch.cat(Y,0)

    return U,Z,W,X,Y

if __name__=="__main__":
    get_tuple(torch.rand(4,4),torch.rand(4,4),torch.rand(4,4),0.2)
    #print(get_dataset(torch.rand(4,4),torch.rand(4,4),torch.rand(4,4),0.2,10))