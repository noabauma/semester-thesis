import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing

#%matplotlib inline
#%load_ext autoreload
#%autoreload 2

#torch.manual_seed(42)   #prefix random seed
np.set_printoptions(precision=4)

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=3
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=3, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

#This numbers are arbitrary
def getReferenceData():
  y=[]
  y.append(86.0)		#WCA on graphene		[°Degree]		(Werder et al. 2003)
  y.append(106.925)		#WCA in CNT	(96,0)		[°Degree]		(Werder et al. 2001)
  #y.append(0.775)		#Viscosity 				[mPa*s]			(Thomas et al. 2010)
  y.append(9500.0)		#Friction coefficient	[kg/(s*m^2)]		(Falk et al. 2010)
  return y



def main():
    training_iterations = 1000000

    double_prec = True
    cuda = True
    train = False

    dtype_ = np.float64 if double_prec else np.float32

    train_data = np.loadtxt("result.csv", delimiter=',', dtype=dtype_)
    
    train_x = train_data[:100,1:5]
    train_y = train_data[:100,(5,6,8)]  #don't use viscosity

    keep = ~np.isnan(train_y[:,0]) & ~np.isnan(train_y[:,1]) & (train_y[:,0] != 0.0) & (train_y[:,1] != 0.0) #remove rows with nan or 0.0
    train_x = train_x[keep] 
    train_y = train_y[keep] 

    #np.nan_to_num(train_y, copy=False, nan=0.0) #some wca couldn't be calculated, so either use 0° or 180°

    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

    train_x = train_x.contiguous()  #see: https://github.com/cezannec/capsule_net_pytorch/issues/4
    train_y = train_y.contiguous()

    #train_x = torch.linspace(0, 1, 100)
    #train_y = torch.stack([torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2, torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,], -1)

    print(train_x.shape)
    print(train_y.shape)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
    model = MultitaskGPModel(train_x, train_y, likelihood)

    # use double precision
    if double_prec:
        model = model.double()
        likelihood = likelihood.double()
    
    # activate cuda
    if cuda:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()

    if train:
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            if i % 100 == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()

        torch.save(model.state_dict(), "save_state/model_state.pth")
        torch.save(likelihood.state_dict(), "save_state/likelihood_state.pth")
    else:
        state_dict = torch.load("save_state/model_state.pth")
        model.load_state_dict(state_dict)
    
    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        A_eps_min = 0.01       #[kcal/mol]
        A_eps_max = 2.0
    
        B_min = 0.0            #[1]
        B_max = 2.0
    
        sigma_min = 2.0        #[A]
        sigma_max = 5.0
    
        rc_min = 5.0           #[A]
        rc_max = 15.0

        parameters = np.loadtxt("samples-1000.csv", delimiter=',', dtype=dtype_, skiprows=1)
        N = parameters.shape[0]
        test_x = np.array([(A_eps_max - A_eps_min)*parameters[:,0] + A_eps_min, (B_max - B_min)*parameters[:,1] + B_min, (sigma_max - sigma_min)*parameters[:,2] + sigma_min, (rc_max - rc_min)*parameters[:,3] + rc_min])
        test_x = test_x.T
        test_x = torch.from_numpy(test_x)
        test_x = test_x.contiguous()
        test_x = test_x.cuda()

        predictions = likelihood(model(test_x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
    
    #goal parameters
    referenceData = np.array(getReferenceData()).reshape((1, -1))

    if cuda:
        test_x = test_x.cpu()
        mean   = mean.cpu()
        lower  = lower.cpu()
        upper  = upper.cpu()

    mean   = mean.numpy()
    test_x = test_x.numpy()
    lower  = lower.numpy()
    upper  = upper.numpy()

    #transform variables such that (mean = 0 and var = 1)
    transform = False
    if transform:
        scaler = preprocessing.StandardScaler().fit(mean)
        y_scaled = scaler.transform(mean)
        goal_scaled = scaler.transform(referenceData)
        lower_scaled = scaler.transform(lower)
        upper_scaled = scaler.transform(upper)
    else:
        y_scaled = mean
        goal_scaled = referenceData
        lower_scaled = lower
        upper_scaled = upper


    dtype = [('index', int), ('error', float)]
    error = []

    temp = upper_scaled - lower_scaled
    diff = (temp @ temp.T).diagonal()
    print(diff.shape)
    for i in range(N):
        error.append((i, (goal_scaled[0,0] - y_scaled[i,0])**2 + (goal_scaled[0,1] - y_scaled[i,1])**2 + (goal_scaled[0,2] - y_scaled[i,2])**2 + diff[i]))
    error = np.array(error, dtype=dtype)
    error = np.sort(error, order='error')

    idx = np.empty(N, dtype=np.int32)
    for i in range(N):
        idx[i] = error[i][0]

    y = mean[idx]
    x = test_x[idx]
    lower = lower[idx]
    upper = upper[idx]
    diff  = diff[idx]

    print("[wca_grs, wca_cnt, fric. coef] \t [A_eps, B, sigma, rc]\n")
    top = 10
    t = 1
    i = 0
    y_old0 = y[0,0]
    y_old1 = y[0,1]
    print(t, ". ", y[0], " ", x[0], " ", lower[0], " ", upper[0])
    t += 1
    while(t <= top and i < N):
        if y_old0 != y[i,0] or y_old1 != y[i,1]:
            print(t, ". ", y[i], " ", x[i], " ", lower[i], " ", upper[i])
            y_old0 = y[i,0]
            y_old1 = y[i,1]
            t += 1
        i += 1
    

    """
    # Initialize plots
    f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 51).cuda()
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    # use the data from the cpu for plotting
    mean  = mean.cpu()
    lower = lower.cpu()
    upper = upper.cpu()

    train_x = train_x.cpu()
    train_y = train_y.cpu()
    test_x  = test_x.cpu()
        
    # This contains predictions for both tasks, flattened out
    # The first half of the predictions is for the first task
    # The second half is for the second task

    # Plot training data as black stars
    y1_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), 'b')
    # Shade in confidence 
    y1_ax.fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
    y1_ax.set_ylim([-3, 3])
    y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y1_ax.set_title('Observed Values (Likelihood)')

    # Plot training data as black stars
    y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), 'b')
    # Shade in confidence 
    y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
    y2_ax.set_ylim([-3, 3])
    y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y2_ax.set_title('Observed Values (Likelihood)')

    plt.show()
    """
    

if __name__ == "__main__":
    main()