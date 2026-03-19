from libmoonRAW.problem.synthetic.mop import BaseMOP
import numpy as np
import torch


class F1(BaseMOP):
    def __init__(self, n_var=6, n_obj=2, lbound=np.zeros(6), ubound=np.ones(6)):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )
        # self.n_dim = n_var
        # self.n_var = n_var
        # self.n_obj = 2
        # self.lbound = torch.zeros(n_var).float()
        # self.ubound = torch.ones(n_var).float()
        self.problem_name = 'F1'

    def _evaluate_torch(self, x):
        n = x.shape[1]
        # assert torch.isnan(x).any() == False, "nan in x"

        sum1 = sum2 =  0.0
        count1 = count2 =  0.0

        for i in range(2,n+1):
            yi = x[:,i-1] - torch.pow(2 * x[:,0] - 1, 2)
            yi = yi * yi

            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 )))

        # assert torch.isnan(f1).any() == False, "nan in f1"
        # assert (x[:,0] >= 0.0).all(), f"x1 < 0: {x[:,0]}"
        # assert torch.isnan(sum2).any() == False, "nan in sum2"
        # assert torch.isnan(f2).any() == False, f"nan in f2: {f2}"

        objs = torch.stack([f1,f2]).T
        return objs

    def _evaluate_numpy(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0.0
        count1 = count2 = 0.0

        for i in range(2, n + 1):
            yi = x[:, i - 1] - np.power(2 * x[:, 0] - 1, 2)
            yi = yi * yi

            if i % 2 == 0:
                sum2 += yi
                count2 += 1.0
            else:
                sum1 += yi
                count1 += 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - np.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = np.stack([f1, f2]).T

        return objs

    def get_pf(self, n_pareto_points: int = 100):
        f1 = np.linspace(0, 1, n_pareto_points)
        f2 = 1 - np.sqrt(f1)
        return np.stack((f1, f2), axis=1)


class F2(BaseMOP):
    def __init__(self, n_var=6, n_obj=2, lbound=np.zeros(6), ubound=np.ones(6)):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )
        self.problem_name = 'F2'
    
    def _evaluate_torch(self, x):
        n = x.shape[1]
       
        sum1 = sum2 =  0.0
        count1 = count2 =  0.0
            
        for i in range(2,n+1):
            theta = 1.0 + 3.0*(i-2)/(n - 2)
            yi    = x[:,i-1] - torch.pow(x[:,0], 0.5*theta)
            yi    = yi * yi
            
            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0/count1 * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
    def _evaluate_numpy(self, x):
        n = x.shape[1]
       
        sum1 = sum2 =  0.0
        count1 = count2 =  0.0
            
        for i in range(2,n+1):
            theta = 1.0 + 3.0*(i-2)/(n - 2)
            yi    = x[:,i-1] - np.power(x[:,0], 0.5*theta)
            yi    = yi * yi
            
            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0/count1 * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - np.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = np.stack([f1,f2]).T
        
        return objs
    
    def get_pf(self, n_pareto_points: int = 100):
        f1 = np.linspace(0, 1, n_pareto_points)
        f2 = 1 - np.sqrt(f1)
        return np.stack((f1, f2), axis=1)


class F3(BaseMOP):
    def __init__(self, n_var=6, n_obj=2, lbound=np.zeros(6), ubound=np.ones(6)):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )
        self.problem_name = 'F3'
    
    def _evaluate_torch(self, x):
        n = x.shape[1]
       
        sum1 = sum2 = 0.0
        count1 = count2 = 0.0
        
        for i in range(2,n+1):
            xi = x[:,i-1]
            yi = xi - (torch.sin(4.0*np.pi* x[:,0]  + i*np.pi/n) + 1) / 2
            yi = yi * yi 
            
            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0
       
        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0])) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs

    def _evaluate_numpy(self, x):
        n = x.shape[1]
       
        sum1 = sum2 = 0.0
        count1 = count2 = 0.0
        
        for i in range(2,n+1):
            xi = x[:,i-1]
            yi = xi - (np.sin(4.0*np.pi* x[:,0]  + i*np.pi/n) + 1) / 2
            yi = yi * yi 
            
            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0
       
        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - np.sqrt(x[:,0])) 
        
        objs = np.stack([f1,f2]).T
        
        return objs
    
    def get_pf(self, n_pareto_points: int = 100):
        f1 = np.linspace(0, 1, n_pareto_points)
        f2 = 1 - np.sqrt(f1)
        return np.stack((f1, f2), axis=1)


class F4(BaseMOP):
    def __init__(self, n_var=6, n_obj=2, lbound=np.zeros(6), ubound=np.ones(6)):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )
        self.problem_name = 'F4'
    
    def _evaluate_torch(self, x):
        n = x.shape[1]
       
        sum1 = sum2 = 0
        count1 = count2 = 0
        
        for i in range(2,n+1):
            xi = -1.0 + 2.0*x[:,i-1]
 
            if i % 2 == 0:
                yi = xi - 0.8 * x[:,0] * torch.sin(4.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - 0.8* x[:,0] * torch.cos(4.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0
       
        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
    def _evaluate_numpy(self, x):
        n = x.shape[1]
       
        sum1 = sum2 = 0
        count1 = count2 = 0
        
        for i in range(2,n+1):
            xi = -1.0 + 2.0*x[:,i-1]
 
            if i % 2 == 0:
                yi = xi - 0.8 * x[:,0] * np.sin(4.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - 0.8* x[:,0] * np.cos(4.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0
       
        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - np.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = np.stack([f1,f2]).T
        
        return objs
    
    def get_pf(self, n_pareto_points: int = 100):
        f1 = np.linspace(0, 1, n_pareto_points)
        f2 = 1 - np.sqrt(f1)
        return np.stack((f1, f2), axis=1)


class F5(BaseMOP):
    def __init__(self, n_var=6, n_obj=2, lbound=np.zeros(6), ubound=np.ones(6)):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )
        self.problem_name = 'F5'
    
    def _evaluate_torch(self, x):
        n = x.shape[1]
       
        sum1 = sum2 = 0
        count1 = count2 = 0
        
        for i in range(2,n+1):
            xi = -1.0 + 2.0*x[:,i-1]
 
            if i % 2 == 0:
                yi = xi - 0.8 * x[:,0] * torch.sin(4.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - 0.8 * x[:,0] * torch.cos((4.0*np.pi*x[:,0] + i*np.pi/n)/3)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0
       
        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
    def _evaluate_numpy(self, x):
        n = x.shape[1]
       
        sum1 = sum2 = 0
        count1 = count2 = 0
        
        for i in range(2,n+1):
            xi = -1.0 + 2.0*x[:,i-1]
 
            if i % 2 == 0:
                yi = xi - 0.8 * x[:,0] * np.sin(4.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - 0.8 * x[:,0] * np.cos((4.0*np.pi*x[:,0] + i*np.pi/n)/3)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0
       
        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - np.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = np.stack([f1,f2]).T
        
        return objs
    
    def get_pf(self, n_pareto_points: int = 100):
        f1 = np.linspace(0, 1, n_pareto_points)
        f2 = 1 - np.sqrt(f1)
        return np.stack((f1, f2), axis=1)


class F6(BaseMOP):
    def __init__(self, n_var=6, n_obj=2, lbound=np.zeros(6), ubound=np.ones(6)):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )
        self.problem_name = 'F6'

    def _evaluate_torch(self, x):
        n = x.shape[1]
       
        sum1 = sum2 = 0
        count1 = count2 = 0
        
        for i in range(2,n+1):
            xi = -1.0 + 2.0*x[:,i-1]
 
            if i % 2 == 0:
                yi = xi - (0.3 * x[:,0] ** 2 * torch.cos(12.0*np.pi*x[:,0] + 4 *i*np.pi/n) + 0.6 * x[:,0]) * torch.sin(6.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - (0.3 * x[:,0] ** 2 * torch.cos(12.0*np.pi*x[:,0] + 4 *i*np.pi/n) + 0.6 * x[:,0]) * torch.cos(6.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0
       
        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
    def _evaluate_numpy(self, x):
        n = x.shape[1]
       
        sum1 = sum2 = 0
        count1 = count2 = 0
        
        for i in range(2,n+1):
            xi = -1.0 + 2.0*x[:,i-1]
 
            if i % 2 == 0:
                yi = xi - (0.3 * x[:,0] ** 2 * np.cos(12.0*np.pi*x[:,0] + 4 *i*np.pi/n) + 0.6 * x[:,0]) * np.sin(6.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - (0.3 * x[:,0] ** 2 * np.cos(12.0*np.pi*x[:,0] + 4 *i*np.pi/n) + 0.6 * x[:,0]) * np.cos(6.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0
       
        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - np.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = np.stack([f1,f2]).T
        
        return objs
    
    def get_pf(self, n_pareto_points: int = 100):
        f1 = np.linspace(0, 1, n_pareto_points)
        f2 = 1 - np.sqrt(f1)
        return np.stack((f1, f2), axis=1)
        

class F7(BaseMOP):
    def __init__(self):
        self.n_dim = 2
        self.n_obj = 2
        self.lbound = torch.tensor([-1, -1])
        self.ubound = torch.tensor([1, 1])

    def forward(self, theta):
        def h1(theta):
            return torch.log(torch.max(torch.abs(0.5 * (-theta[:,0]) - 7) - torch.tanh(-theta[:,1]), torch.tensor(0.000005))) + 6

        def h2(theta):
            return torch.log(torch.max(torch.abs(0.5 * (-theta[:,0]) + 3) - torch.tanh(-theta[:,1]) + 2, torch.tensor(0.000005))) + 6

        def g1(theta):
            return (((-theta[:,0] + 7) ** 2 + 0.1 * ((-theta[:,1]) ** 2 - 8) ** 2) / 10) - 20

        def g2(theta):
            return (((-theta[:,0] - 7) ** 2 + 0.1 * ((-theta[:,1]) ** 2 - 8) ** 2) / 10) - 20

        def c1(theta):
            return torch.max(torch.tanh(0.5 * theta[:,0]), torch.tensor(0.0))

        def c2(theta):
            return torch.max(torch.tanh(-0.5 * theta[:,1]), torch.tensor(0.0))

        def f1(theta):
            return c1(theta) * h1(theta) + c2(theta) * g1(theta)

        def f2(theta):
            return c1(theta) * h2(theta) + c2(theta) * g2(theta)

        return torch.stack([f1(theta), f2(theta)]).T



if __name__ == '__main__':


    x = np.random.random((100, 30))
    problem = F1(n_var=6)

    y = problem.evaluate(x)
    print(y)
    print(y.shape)
    print()




