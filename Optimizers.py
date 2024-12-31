import numpy as np

class BaseOptimizer:
    def __init__(self, lr, **kwargs):
        """
        Base optimizer class to be inherited by specific optimizer classes.
        """
        self.lr = lr
        self.params = kwargs  # Store any additional parameters for the optimizer

    def update(self, model, gradients):
        """
        This function should be overridden by subclasses.
        """
        raise NotImplementedError("The 'update' method must be implemented in subclasses.")

class SGD(BaseOptimizer):
    def __init__(self, lr = 0.001, weight_decay = 1e-4):
        super().__init__(lr, wd = weight_decay)
        self.wd = weight_decay

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]):
        n = len(params)

        for i in range(n):
            params[i] *= (1 - self.lr * self.wd)  # Apply weight decay
            params[i] -= self.lr * grads[i]      # Gradient update
        return

class Momentum(BaseOptimizer):
    def __init__(self, lr = 0.001, beta = 0.9):
        super().__init__(lr, beta=beta)
        self.beta = beta
        self.initialized = False
        self.u = []

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]):
        # Initialize momentum terms (u) on the first call
        if not self.initialized:
            self.u = [np.zeros_like(p) for p in params]
            self.initialized = True

        # Perform the momentum update
        for i in range(len(params)):
            self.u[i] = self.beta * self.u[i] + self.lr * grads[i]  # Update velocity
            params[i] -= self.u[i]  # Update parameters

        return 
        

class Nesterov(BaseOptimizer):
    def __init__(self, lr, beta):
        super().__init__(lr, beta = beta)
        self.beta = beta
        self.curr_params: list[np.ndarray] = [] #Store cuurent value of parameters
        self.u: list[np.ndarray] = []
        self.initialized = False


    def update(self, params: list[np.ndarray], grads: list[np.ndarray]):
        # Initialize direction terms (u) on the first call
        if not self.initialized:
            self.u = [np.zeros_like(p) for p in params]
            self.curr_params = params
            self.initialized = True
        
        for i in range(len(params)):
            self.u[i] = self.beta * self.u[i] + self.lr * grads[i]  # Update velocity
            params[i] = self.curr_params[i] - self.u[i]  # Update parameters

            #Notice that we need to update parameters from stored original parameter
            #that was assigned in get_look_ahead_direction method

            self.curr_params[i] = params[i]             #update the curr_params with updated params value
            
            #This update is optional and is kept to ensure functioning of Nesterov in case it is
            #called without look ahead. In that case it will be just momentum update

        return
    
    def get_look_ahead_direction(self, params: list[np.ndarray]):

        if not self.initialized:
            self.u = [np.zeros_like(p) for p in params]
            self.curr_params = [np.zeros_like(p) for p in params]
            self.initialized = True

        n = len(params)
        for i in range(len(params)):
            #Store the current value of parameter for updating it in update method
            self.curr_params[i] = params[i]

            #update the current params with look ahead direction to compute look_ahead gradient
            params[i] -= self.beta * self.u[i]
        return


class RMSProp(BaseOptimizer):
    def __init__(self, lr, beta):
        super().__init__(lr, beta=beta)
        self.beta = beta
        self.initialized = False
        self.u = []
        self.eps = 1e-6

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]):
        if not self.initialized:
            self.u = [np.zeros_like(p) for p in params]
            self.initialized = True

        # Perform the momentum update
        for i in range(len(params)):
            self.u[i] = self.beta * self.u[i] + (1-self.beta) * (grads[i]**2)  # Update velocity
            params[i] -= self.lr / np.sqrt(self.u[i] + self.eps) * grads[i]  # Update parameters

        return 

class Adam(BaseOptimizer):
    def __init__(self, lr = 0.001, beta1 = 0.9, beta2 = 0.99, epsilon=1e-8):
        super().__init__(lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.iter = 1
        self.initialized = False
        self.m: list[np.ndarray] = []
        self.v: list[np.ndarray] = []

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]):
        if not self.initialized:
            self.v = [np.zeros_like(p) for p in params]
            self.m = [np.zeros_like(p) for p in params]
            self.initialized = True
        
        #Perform Adam update
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1)*grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2)*(grads[i]**2)
            m_hat = self.m[i]/(1-self.beta1 ** self.iter)
            v_hat = self.v[i]/(1-self.beta2 ** self.iter)

            #update the parameter and increase the iter count
            params[i] -= self.lr/(np.sqrt(v_hat) + self.epsilon) * m_hat
            self.iter += 1

        return

class Nadam(BaseOptimizer):
    def __init__(self, lr, beta1, beta2, epsilon=1e-8):
        super().__init__(lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.iter = 1
        self.initialized = False
        self.m: list[np.ndarray] = []
        self.v: list[np.ndarray] = []

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]):
        if not self.initialized:
            self.v = [np.zeros_like(p) for p in params]
            self.m = [np.zeros_like(p) for p in params]
            self.initialized = True
        
        #Perform NAdam update
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1)*grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2)*(grads[i]**2)
            m_hat = self.m[i]/(1-self.beta1 ** self.iter)
            v_hat = self.v[i]/(1-self.beta2 ** self.iter)
            grad_hat = grads[i]/(1-self.beta1 ** self.iter)

            #update the parameter and increase the iter count
            params[i] -= self.lr/(np.sqrt(v_hat) + self.epsilon) * (self.beta1 * m_hat \
                                    + (1-self.beta1)*grad_hat)
            self.iter += 1

        return

