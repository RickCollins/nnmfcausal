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
    y_prob = proby.sum(dim=1)
    y = (y_prob > 0.5).float().unsqueeze(0)

    return U,z,w,x,y

def get_dataset(theta_a_z,theta_y_a,theta_y_w, p, total):
    U,Z,W,X,Y = [],[],[],[],[]
    for _ in range(total):
        u,z,w,x,y = get_tuple(theta_a_z,theta_y_a,theta_y_w,p)
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


def get_tuple_new(theta_a_z,theta_y_a,theta_y_w,theta_y_epsilon, theta_a_epsilon, p=0.2):
    vec1 = torch.tensor([0.1,0.1,0.4,0.4])
    vec2 = torch.tensor([0.4,0.4,0.1,0.1])

    epsilon = torch.bernoulli(torch.tensor([p])).float()

    cat0 = torch.distributions.categorical.Categorical(vec1) 
    cat1 = torch.distributions.categorical.Categorical(vec2) 

    z = torch.nn.functional.one_hot((cat0.sample()*epsilon + cat1.sample()*(1-epsilon)).long(), num_classes = 4).unsqueeze(0).float() #discrete so categorical depending on epsilon (paper has Normal as cont)
    w = torch.nn.functional.one_hot((cat0.sample()*epsilon + cat1.sample()*(1-epsilon)).long(), num_classes = 4).unsqueeze(0).float() #discrete so categorical depending on epsilon (paper has Normal as cont)

    proba = (vec1 * epsilon + vec2 * (1-epsilon)).unsqueeze(0)  
    proby = (vec1 * epsilon + vec2 * (1-epsilon)).unsqueeze(0)
    
    print("proba shape",proba.shape)
    print("theta_a_z shape",theta_a_z.shape)
    print("z.T shape",z.T.shape)
    print("theta_a_epsilon shape",theta_a_epsilon.shape)
    print("epsilon.T shape",epsilon.T.shape)


    proba = proba * (theta_a_z @ z.T + theta_a_epsilon * epsilon.T).T # a is a function of z and epsilon from the graph
    a = torch.bernoulli(torch.sigmoid(proba).sample().long(), num_classes = 4).float()

    proby = proby * (theta_y_a @ a.T + theta_y_w @ w.T + theta_y_epsilon * epsilon.T).T # y is a function of a, w and epsilon from the graph
    #y = torch.bernoulli(torch.nn.Sigmoid(proby).sample().long(), num_classes = 4).float()
    y = torch.bernoulli(torch.sigmoid(proby).squeeze()).float()

    return epsilon,z,w,a,y

def get_dataset_new(theta_a_z,theta_y_a,theta_y_w,theta_y_epsilon, theta_a_epsilon, p, total):
    U,Z,W,X,Y = [],[],[],[],[]
    for _ in range(total):
        u,z,w,x,y = get_tuple_new(theta_a_z,theta_y_a,theta_y_w,theta_y_epsilon, theta_a_epsilon, p)
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