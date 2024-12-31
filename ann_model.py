import numpy as np

class Linear:
    def __init__(self, n_in: int, n_out: int):
        self.weight = np.random.uniform(-np.sqrt(6 / (n_in + n_out)),
                                         np.sqrt(6 / (n_in + n_out)),
                                         size=(n_out, n_in))
        self.bias = np.zeros((n_out, 1))
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        self.output: np.ndarray = None
        self.input: np.ndarray = None
    
    def forward(self, x: np.ndarray):
        self.input = x
        self.output = self.weight @ x + self.bias
        return self.output

    def backward(self, grad_a: np.ndarray):
        m = self.output.shape[1]                   #Number of batches processed
        self.grad_weight = grad_a @ (self.input).T / m
        self.grad_bias = np.sum(grad_a, axis = 1, keepdims= True)/m
        return (self.weight).T @ grad_a

    def __call__(self, x: np.ndarray):
        return self.forward(x)

class ReLU:
    def __init__(self):
        self.input: np.ndarray = None
        pass

    def forward(self, x: np.ndarray):
        self.input = x
        return np.maximum(x, 0)
    
    def backward(self, grad_h: np.ndarray):
        dh_da = (self.input > 0).astype(int)
        return grad_h * dh_da
    
    def __call__(self, x: np.ndarray):
        return self.forward(x)

class softmax:
    def __init__(self):
        self.output: np.ndarray = None
        pass

    def forward(self, x: np.ndarray):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=0, keepdims=True)
        return self.output
    
    def backward(self, grad_h: np.ndarray):
        K,N = self.output.shape
        y_diag = np.zeros((K, K, N))
        for i in range(N):
            y_diag[:, :, i] = np.diag(self.output[:, i])

        # Compute the outer product h[:, col] @ h[:, col].T for all columns
        outer_product = np.einsum('ik,jk->ijk', self.output, self.output)  # Shape (K, K, N)

        # Compute local gradient
        dh_da = y_diag - outer_product  # Shape (K, K, N) 

        #compute the backward passed gradient
        grad_a = np.einsum('ijk,jk->ik', dh_da , grad_h)
 
        return grad_a
    
    def __call__(self, x: np.ndarray):
        return self.forward(x)
    
class cross_entropy_loss:
    def __init__(self):
        pass

    def forward(self, ytarget: np.ndarray, ypred: np.ndarray):
        return -np.sum(ytarget * np.log(ypred + 1e-8))
    
    def backward(self, ytarget: np.ndarray, ypred: np.ndarray):
        return -1* ytarget / (ypred + 1e-8)
    
    def __call__(self, ytarget: np.ndarray, ypred: np.ndarray):
        return self.forward(ytarget, ypred)
    
    
class Optimization:
    def __init__(self, model: callable, optimizer: callable):
        
        self.num_layers = len(model.layers)
        self.params: list[np.ndarray] = []
        self.grads: list[np.ndarray] = []
        self.model = model
        self.optimizer = optimizer

        #initialize parameters and gradients
        for layer_items in model.layers:
            self.params.append(layer_items['layer'].weight)
            self.params.append(layer_items['layer'].bias)

            self.grads.append(layer_items['layer'].grad_weight)
            self.grads.append(layer_items['layer'].grad_bias)

    #perform backward pass to compute gradients and accumulate them in respective gradient 
    #variables of different layers
    def backward_pass(self, ytarget: np.ndarray,
                       ypred: np.ndarray, lossfunc: callable = None):
        grad_h = lossfunc.backward(ytarget, ypred)
        for layer_item in reversed(self.model.layers):
            grad_a = layer_item['activation'].backward(grad_h)
            grad_h = layer_item['layer'].backward(grad_a)

    def update_params(self):

        #iterate through the layer list to record layer parameters and gradients
        for i, layer_items in enumerate(self.model.layers):
            self.params[2*i] = layer_items['layer'].weight
            self.params[2*i+1] = layer_items['layer'].bias

            self.grads[2*i] = layer_items['layer'].grad_weight
            self.grads[2*i+1] = layer_items['layer'].grad_bias
        
        #call the optimizer to update the parameters
        self.optimizer.update(self.params, self.grads)

        #assign the updated parameters to model paramaters
        for i, layer_items in enumerate(self.model.layers):
            layer_items['layer'].weight = self.params[2*i]
            layer_items['layer'].bias = self.params[2*i+1]

            layer_items['layer'].grad_weight = self.grads[2*i]
            layer_items['layer'].grad_bias = self.grads[2*i+1]


class MLLclassification:
    def __init__(self, in_features, out_features, neurons, activations):
        num_layers = len(neurons)
        self.layers = []
        for i in range(num_layers):
            if i==0:
                layer = Linear(n_in = in_features, n_out = neurons[i])
                activation = activations[i]()
            else:
                layer = Linear(n_in = neurons[i-1], n_out = neurons[i])
                activation = activations[i]()
            
            self.layers.append({
                'layer': layer,
                'activation': activation
            })
        
    def forward(self, x: np.ndarray):
        h = x
        for layer_item in self.layers:
            a = layer_item['layer'](h)
            h = layer_item['activation'](a)
        
        return h

