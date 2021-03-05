import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    
    #Copied from cw1
    X = np.append(X, np.ones((X.shape[0],1)), axis = 1)#Bias trick
    X = X.astype(np.float64)
    
    #New part
    X_mean = np.mean(X)
    X_std = np.std(X)
    
    #Normalizing input
    X_norm = (X - X_mean) / X_std
    
    return X_norm


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    #Copied from CW1 task3a.py
    multi_class_cross_entro_error = np.mean(-np.sum(targets * np.log(outputs), axis = 1))
    
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    return multi_class_cross_entro_error


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        
        

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer
        
        #Initializing activations and forwards objects as empty arrays
        self.activations = []
        self.forwards = []
        # Initialize the weights
        self.ws = []
        
        prev = self.I
        
        self.moment = []
        
        if use_improved_weight_init:
            for size in self.neurons_per_layer:
                w_shape = (prev, size)
                momentum = np.zeros(w_shape)
                self.moment.append(momentum)
                print("Initialize weight to shape:", w_shape)
                w = np.random.normal(0, 1/np.sqrt(prev),w_shape)
                self.ws.append(w)
                prev = size
        else:
            for size in self.neurons_per_layer:
                w_shape = (prev, size)
                momentum = np.zeros(w_shape)
                self.moment.append(momentum)
                print("Initialize weight to shape:", w_shape)
                w = np.random.uniform(-1,1,w_shape)
                self.ws.append(w)
                prev = size
        
        self.grads = [None for i in range(len(self.ws))]
        
            

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For peforming the backward pass, you can save intermediate activations in varialbes in the forward pass.
        # such as self.hidden_layer_ouput = ...
        # The goal is to translate the softmax from cw1 with 1 layer to this multilayer
        # 
        # 
        
        forwards = []
        activations = []
        previous = X
        
        activations.append(previous)
        for i in range(len(self.ws) -1 ):
            z = np.dot(previous, self.ws[i])
            
            #Sigmoid
            previous = 1 / (1 + np.exp(-z))
            
            #Appending results to forwards and activations
            forwards.append(z) 
            activations.append(previous)
            
        z = np.dot(previous,self.ws[-1])
        forwards.append(z)

        #softmax
        previous = np.exp(z) / np.sum(np.exp(z), axis=1)[:, None]
        
        #Adding the variables forwards and activations to the objects in the class
        self.forwards = forwards
        self.activations = activations
        
        return previous

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        self.grads = []
        
        activations = self.activations
        forwards = self.forwards
        
        for i in range(len(self.ws)):
            if i == 0:
                del_k = - (targets - outputs)
                self.grads.append(np.dot(del_k.T, activations[-1]).T/X.shape[0])
            else:
                if self.use_improved_sigmoid:
                    d_sig = 2.5/np.cosh((3*forwards[-i-1])/4) 
                    #x is forwards[-i-1]
                    #d_sig = 1.7159*np.tanh((2*forwards[-i-1])/3) from litterature
                    del_j = np.dot(del_k, self.ws[-i].T) * d_sig
                    grad_w_ij = np.dot(activations[-i-1].T, del_j)
                    self.grads.insert(0, grad_w_ij/X.shape[0])
                    del_k = del_j
                else:
                    #all hidden layer
                    d_sig = (1/(1 + np.exp(-forwards[-i-1])))*(1 - (1/(1 + np.exp(-forwards[-i-1]))))
                    del_j = np.dot(del_k, self.ws[-i].T)*d_sig
                    grad_w_ij = np.dot(activations[-i-1].T,del_j)
                    self.grads.insert(0, grad_w_ij/X.shape[0]) #insert in the beginning of list
                    del_k = del_j
        
        #zj = np.dot(X, self.ws[0])
        #aj = 1/(1 + np.exp(-zj))

        # del_k = - (targets - outputs)
        # grad_w_kj = (np.dot(del_k.T, activations[-1])).T #outputlayer

        # der_sig = (1/(1 + np.exp(-zj)))*(1 - (1/(1 + np.exp(-zj))))
        
        # del_j = np.dot(self.ws[1], del_k.T) * der_sig.T
        # grad_w_ij = np.dot(del_j, X).T

        # self.grads = [grad_w_ij.T/X.shape[0], grad_w_kj.T/X.shape[0]]
        # self.grads = [grad_w_ij/X.shape[0], grad_w_kj/X.shape[0]]

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."
                
        return self.grads   
# %%
    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    #Copied from CW1 task3a.py
    oh_array = [[1 if i == item else 0 for i in range(num_classes)]
                   for item in Y]
    oh_array = np.array(oh_array)
    # print (oh_array)
    return oh_array


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        gradient_approximation_test(model, X_train, Y_train)
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    # gradient_approximation_test(model, X_train, Y_train) // Skal denne være inni løkka?
