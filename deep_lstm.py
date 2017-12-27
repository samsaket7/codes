import numpy as np

class Deep_lstm_Network:
    """
    A deep lstm network without peephole connection. The network consists of an
    input layer, multiple hidden layers(also called recurrent layers) and an 
    output layer. The dimension of each layer is passed as an input(num_nodes). 

    For loss computation the network uses L2 loss function but code be easily 
    extended for other losses also.

    Note that this class contains its own training function which supports minibatch
    training option. For parameter update gradient descent with momentum is used. But
    code can easily be extended for other optimization methods. 

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

# ------------------------------------------------------
# Deep Lstm Network Constructor
# ----------------------------------------------------------
    def __init__(self,num_nodes):
        """
        Initialize a new Deep Lstm Network. 

        Inputs:
        - num_nodes: A list of integers giving the size of input layer,
                       hidden layers and ouput layer.
        """

        self.num_nodes = num_nodes
        self.recurrent_layers = len(num_nodes)-2   #num of hidden layers
        self.params = {}                           #dict for parameter storage

        #Network parameters
        for layer in range(self.recurrent_layers):
            self.params['Wf'+str(layer)] = np.random.randn(num_nodes[layer],num_nodes[layer+1])
            self.params['Uf'+str(layer)] = np.random.randn(num_nodes[layer+1],num_nodes[layer+1])
            self.params['bf'+str(layer)] = np.zeros(num_nodes[layer+1])
            self.params['Wa'+str(layer)] = np.random.randn(num_nodes[layer],num_nodes[layer+1])
            self.params['Ua'+str(layer)] = np.random.randn(num_nodes[layer+1],num_nodes[layer+1])
            self.params['ba'+str(layer)] = np.zeros(num_nodes[layer+1])
            self.params['Wi'+str(layer)] = np.random.randn(num_nodes[layer],num_nodes[layer+1])
            self.params['Ui'+str(layer)] = np.random.randn(num_nodes[layer+1],num_nodes[layer+1])
            self.params['bi'+str(layer)] = np.zeros(num_nodes[layer+1])
            self.params['Wo'+str(layer)] = np.random.randn(num_nodes[layer],num_nodes[layer+1])
            self.params['Uo'+str(layer)] = np.random.randn(num_nodes[layer+1],num_nodes[layer+1])
            self.params['bo'+str(layer)] = np.zeros(num_nodes[layer+1])

        self.params['Wy'] = np.random.randn(num_nodes[-2],num_nodes[-1])
        self.params['by'] = np.zeros(num_nodes[-1])

        # Default training options
        self.max_epoch = 100
        self.learning_rate = 0.01
        self.num_batches = 1
        self.bptt_trunc = 5
        self.initialmomentum = 0.5
        self.finalmomentum = 0.9
        self.initialepochs = 5


# ----------------------------------------------------------
# Training Options
# ----------------------------------------------------------
    def options(self, max_epoch=100, num_batches=1, learning_rate=0.01,
                        bptt_trunc = 5):
        """
        Overwrites the default training option.
        Inputs:
        - Self explanatory
        """
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.num_batches = num_batches
        self.bptt_trunc = bptt_trunc


# ----------------------------------------------------------
# Forward Pass
# ----------------------------------------------------------
    def sigmoid(self,x):
	    return 1.0/(1+np.exp(-x))

    def forward_recurrent_step(self,X,prev_hidden_state,prev_c_state,Wf,Uf,bf,Wa,Ua,ba,Wi,Ui,bi,Wo,Uo,bo):
        """
        Computes the forward pass for the recurrent layers.
    
        The ht and ct are calculated using xt,ht-1 and ct-1.ft,it,at and ot are intermediate variables.
    
        Inputs:
    
        Returns a tuple of:
        """
        f_act = np.dot(X,Wf) + np.dot(prev_hidden_state,Uf) + bf
        a_act = np.dot(X,Wa) + np.dot(prev_hidden_state,Ua) + ba
        i_act = np.dot(X,Wi) + np.dot(prev_hidden_state,Ui) + bi
        o_act = np.dot(X,Wo) + np.dot(prev_hidden_state,Uo) + bo
        #using sigmoid and tanh activation fn
        ft = self.sigmoid(f_act)
        at = np.tanh(a_act)
        it = self.sigmoid(i_act)
        ot = self.sigmoid(o_act)

        ct = prev_c_state*ft + it*at
        ht = ot*np.tanh(ct)

        cache = (ht,ct,ot,it,at,ft,X,prev_hidden_state,prev_c_state)
        return ht,ct,cache

    def forward_output_step(self,X,W,b):
        """
        Computes the forward pass for the output layer.
    
        The input x is transformed into output y using the weight and biases.
        Sigmoid is currently used as an activation function
    
        Inputs:
        - x: A numpy array containing input data, of shape (batch_size,last_hidden_layer_dims)
        - w: A numpy array of weights, of shape (last_hidden_layer_dims,output_layer_dims)
        - b: A numpy array of biases, of shape (output_layer_dims,)
    
        Returns a tuple of:
        - out: output, of shape (batch_size,output_layer_dims)
        - cache: (X, W, b)
        """
        act = np.dot(X,W) + b
        #using sigmoid in output layer
        ypred = self.sigmoid(act)
        cache = (ypred,X)
        return ypred,cache


# ----------------------------------------------------------
    def forward_unit_time(self,hidden_states,c_states,X):
        """
        Implements the forward pass for unit time in  Deep lstm network. It sequentially 
        computes the output at the current layer using output of the previous layer(in depth) and 
        hidden_state and c_state values of the previous time step(of same layer). This is repeated for 
        each hidden layer. Finally, output y at this time step is computed by feeding final hidden layer into output_layer.
           
          
        Inputs:
        - X            : A numpy array containing input data of shape (batch_size,input_layer_size)
        - hidden_states: hidden_states of previous timestep,a list of shape(num_hidden_layers,batch_size,hidden_layer_size)
        - c_states     : c_states of previous timestep,a list of shape(num_hidden_layers,batch_size,hidden_layer_size)

        Returns :
        - Ypred        : A numpy array of predicted output at the given time step of shape (batch_size,output_layer_size) 
        - cache        : A list of caches of each layer for given time step( To be used in backprop)
        - hidden_states: hidden_states of current timestep,a list of shape(num_hidden_layers,batch_size,hidden_layer_size)
        - c_states     : c_states of current timestep,a list of shape(num_hidden_layers,batch_size,hidden_layer_size)
        """
        Cache = list()

        #Recurrence stage
        for layer in range(self.recurrent_layers):
            hidden_states[layer],c_states[layer],cache = self.forward_recurrent_step(X,hidden_states[layer],c_states[layer],
                                                                                self.params['Wf'+str(layer)], 
                                                                                self.params['Uf'+str(layer)], 
                                                                                self.params['bf'+str(layer)], 
                                                                                self.params['Wa'+str(layer)], 
                                                                                self.params['Ua'+str(layer)], 
                                                                                self.params['ba'+str(layer)], 
                                                                                self.params['Wi'+str(layer)], 
                                                                                self.params['Ui'+str(layer)], 
                                                                                self.params['bi'+str(layer)], 
                                                                                self.params['Wo'+str(layer)], 
                                                                                self.params['Uo'+str(layer)], 
                                                                                self.params['bo'+str(layer)]) 
            X = hidden_states[layer]
            Cache.append(cache)

        #Output Stage
        ypred,cache = self.forward_output_step(X,self.params['Wy'],self.params['by'])
        Cache.append(cache)

        return ypred,hidden_states,c_states,Cache

# ----------------------------------------------------------
    def forward_propogation(self,X):
        """
        Implements the forward pass for the Deep lstm network. It sequentially 
        computes the output at the current time step using the input at that
        time step and the hidden state and c_state of the previous time step. Hidden states
        and c_state at t=0 are intiallized to 0(for each layer) and for the given time step they are
        internally stored, so that they can be used for next time step.
           
          
        Inputs:
        - X: A numpy array containing input data of shape (time_steps(seq_length),batch_size,input_layer_size)

        Returns :
        - Ypred: A list of predicted output for each time step of shape (time_steps(seq_length),batch_size,output_layer_size) 
        - cache: A list of caches of each time step( To be used in backprop)
        """
        [seq_length, num_data, num_dim] = np.shape(X)
        Cache = list()
        Ypred = list()
        #Set initial states to 0
        hidden_states = [None]*self.recurrent_layers
        c_states      = [None]*self.recurrent_layers
        for layer in range(self.recurrent_layers):
            hidden_states[layer] = np.zeros((num_data,self.num_nodes[layer+1]))
            c_states[layer]      = np.zeros((num_data,self.num_nodes[layer+1]))

        for t in range(seq_length):
            ypred,hidden_states,c_states,cache = self.forward_unit_time(hidden_states,c_states,X[t])
            Cache.append(cache)
            Ypred.append(ypred)

        return Ypred,Cache


# ----------------------------------------------------------
# Loss and its derivate computation
# ----------------------------------------------------------
    def compute_loss(self,Y,Ypred):
        """
        Computes the loss and gradient for L2 loss function.

        Inputs:
        - Ypred: Predicted output, a list of shape(time_steps(seq_length),batch_size,output_layer_size)
        - Y    : Actual_output, numpy array of same shape as Ypred

        Returns a tuple of:
        - loss: Scalar giving the loss
        - dYpred: Gradient of the loss with respect to ypred
        """
        #Currently only supports L2 loss
        Ypred = np.asarray(Ypred)
        loss = 0.5*np.sum((Y-Ypred)**2)
        dYpred = (Ypred-Y)
        return loss,dYpred



# ----------------------------------------------------------
# Backward Pass
# ----------------------------------------------------------
    def backward_recurrent_step(self,dh,dc,cache,Wf,Uf,bf,Wa,Ua,ba,Wi,Ui,bi,Wo,Uo,bo):
        """
        Computes the backward pass.
    
        Inputs:
    
        Returns a tuple of:
        """
        (ht,ct,ot,it,at,ft,X,prev_ht,prev_ct) =cache
        grads = {}
        dho = dh*np.tanh(ct)*ot*(1-ot)
        dc  = dh*ot*(1-np.tanh(ct)**2)+dc
        dhf = dc*prev_ct*ft*(1-ft)
        dhi = dc*at*it*(1-it)
        dha = dc*it*(1-at**2)

        grads['Wf'] = np.dot(X.T,dhf)
        grads['Wa'] = np.dot(X.T,dha)
        grads['Wi'] = np.dot(X.T,dhi)
        grads['Wo'] = np.dot(X.T,dho)
        grads['Uf'] = np.dot(prev_ht.T,dhf)
        grads['Ua'] = np.dot(prev_ht.T,dha)
        grads['Ui'] = np.dot(prev_ht.T,dhi)
        grads['Uo'] = np.dot(prev_ht.T,dho)

        grads['bf'] = np.sum(dhf,axis=0)
        grads['ba'] = np.sum(dha,axis=0)
        grads['bi'] = np.sum(dhi,axis=0)
        grads['bo'] = np.sum(dho,axis=0)

        dh = np.dot(dho,Uo.T) + np.dot(dha,Ua.T) + np.dot(dhi,Ui.T) + np.dot(dhf,Uf.T) 
        dX = np.dot(dho,Wo.T) + np.dot(dha,Wa.T) + np.dot(dhi,Wi.T) + np.dot(dhf,Wf.T) 
        dc = dc*ft

        return dX,dh,dc,grads

    def backward_output_step(self,dypred,cache,W,b):
        """
        Computes the backward pass for the output layer.
        Sigmoid used as activation.
   
        Inputs:
        - dypred: Upstream derivative, of shape (batch_size,output_layer_dims)
        - params: W and b
        - cache: Tuple of:
          - ypred: Ouput data, of shape (batch_size,output_layer_dims)
          - X: Input data, of shape (batch_size,last_hidden_layer_dims)
   
        Returns a tuple of:
        - dx: Gradient with respect to X, of shape (batch_size,last_hidden_layer_dims)
        - dw: Gradient with respect to W, of shape (last_hidden_layer_dims,output_layer_dims)
        - db: Gradient with respect to b, of shape (output_layer_dims,)
    
        """
        (ypred,X)=cache
        dact = (ypred)*(1-ypred)*dypred
        dX   = np.dot(dact,W.T)
        dW   = np.dot(X.T,dact)
        db   = np.sum(dact,axis=0)

        return dX,dW,db

# ----------------------------------------------------------
    def backward_unit_time(self,dy,dh,dc,Cache):
        """
        Implements the backward pass for unit time in  Deep lstm network. It sequentially 
        computes the output at the current layer using output of the previous layer(in depth) and 
        hidden_state and c_state values of the previous time step(of same layer). This is repeated for 
        each hidden layer. Finally, output y at this time step is computed by feeding final hidden layer into output_layer.
           
          
        Inputs:
        - X            : A numpy array containing input data of shape (batch_size,input_layer_size)
        - hidden_states: hidden_states of previous timestep,a list of shape(num_hidden_layers,batch_size,hidden_layer_size)
        - c_states     : c_states of previous timestep,a list of shape(num_hidden_layers,batch_size,hidden_layer_size)

        Returns :
        - Ypred        : A numpy array of predicted output at the given time step of shape (batch_size,output_layer_size) 
        - cache        : A list of caches of each layer for given time step( To be used in backprop)
        - hidden_states: hidden_states of current timestep,a list of shape(num_hidden_layers,batch_size,hidden_layer_size)
        - c_states     : c_states of current timestep,a list of shape(num_hidden_layers,batch_size,hidden_layer_size)
        """
        Grads = {}
        #Output Stage
        dx,dWy,dby = self.backward_output_step(dy,Cache[self.recurrent_layers],self.params['Wy'],self.params['by'])
        Grads['Wy'] = dWy
        Grads['by'] = dby
       
        #Recurrence Stage
        for layer in reversed(range(self.recurrent_layers)):
            dy=dx+dh[layer]
            dx,dhprev,dcprev,grads = self.backward_recurrent_step(dy,dc[layer],Cache[layer],
                                                         self.params['Wf'+str(layer)], 
                                                         self.params['Uf'+str(layer)], 
                                                         self.params['bf'+str(layer)], 
                                                         self.params['Wa'+str(layer)], 
                                                         self.params['Ua'+str(layer)], 
                                                         self.params['ba'+str(layer)], 
                                                         self.params['Wi'+str(layer)], 
                                                         self.params['Ui'+str(layer)], 
                                                         self.params['bi'+str(layer)], 
                                                         self.params['Wo'+str(layer)], 
                                                         self.params['Uo'+str(layer)], 
                                                         self.params['bo'+str(layer)]) 
            dh[layer] = dhprev
            dc[layer] = dcprev
            #Update gradients
            for key in grads:
                Grads[key+str(layer)] = grads[key]
            

        return dh,dc,Grads


# ----------------------------------------------------------
    def backward_propogation(self,dY,Cache):
        """
        Implements the backward pass for the Deep lstm network. It sequentially 
        computes the grads at the current time step using the output derivative at that
        time step and the derivative of the prev(in backward direction i.e t+1) hidden state and c_state. Derivative of Hidden states
        and c_state at t=0 are intiallized to 0(for each layer) and for the given time step the derivative of hidden state and cstate
        are internally stored, so that they can be used for next time step calculation.
           
          
        Inputs:
        - dY: Upstream derivative, numpy array of shape (time_steps(seq_length),batch_size,output_layer_dims)

        Returns :
        - Grads: A dict contating derivative of all the network parameters 
        """
        [seq_length, num_data, num_dim] = np.shape(dY)
        Grads = {}
        #Set initial state derivatives to 0
        dh = [None]*self.recurrent_layers
        dc = [None]*self.recurrent_layers
        for layer in range(self.recurrent_layers):
            dh[layer] = np.zeros((num_data,self.num_nodes[layer+1]))
            dc[layer] = np.zeros((num_data,self.num_nodes[layer+1]))

        for t in reversed(range(seq_length)):
            dh,dc,grads = self.backward_unit_time(dY[t],dh,dc,Cache[t])

            #Update gradients
            for key in grads:
                Grads[key] = Grads.get(key,np.zeros_like(grads[key]))
                Grads[key] += grads[key]

        return Grads


# ----------------------------------------------------------
# Param Update
# ----------------------------------------------------------
    def momentum_update(params,grads,velocity,alpha,momentum):
        """
        Performs stochastic gradient descent with momentum.
    
        config format:
        - learning_rate: Scalar learning rate.
        - momentum: Scalar between 0 and 1 giving the momentum value.
          Setting momentum = 0 reduces to sgd.
        - velocity: A numpy array of the same shape as w and dw used to store a
          moving average of the gradients.
        """
        for key in grads:
            velocity[key] = velocity.get(key,np.zeros_like(params[key]))
            velocity[key] = momentum*velocity[key] - alpha*grads[key]
            params[key] += velocity

        return params,velocity


# ----------------------------------------------------------
# Training
# ----------------------------------------------------------
    def train(self,data,target):

        velocity = {}
        [seq_length, num_train, num_dim] = np.shape(data)
        batch_size = int(floor(num_train/self.num_batches))
        LOSS = list()
        alpha = self.learning_rate
        momentum = self.finalmomentum

        for epoch in range(self.max_epoch):
            for batch in range(self.num_batches):
                #batch formation
                batch_mask = np.random.choice(num_train, batch_size)
                X_batch = data[:,batch_mask,:]
                Y_batch = target[:,batch_mask,:]
                #backprop
                Ypred, cache = self.forward_propogation(X_batch)
                loss,dYpred = self.compute_loss(Y_batch,Ypred)
                grads = self.backward_propogation(dYpred,cache)
                self.params,velocity=self.momentum_update(self.params,grads,velocity,alpha,momentum)

                LOSS.append(loss)



# ----------------------------------------------------------
# Grad Check
# ----------------------------------------------------------

    def grad_check(self):
        seq_length, num_data = 2,5
        epsilon = 1e-8
        data = np.random.randn(seq_length,num_data,self.num_nodes[0])
        target = np.random.randn(seq_length,num_data,self.num_nodes[-1])

        Ypred,cache = self.forward_propogation(data)
        loss,dYpred = self.compute_loss(target,Ypred)
        grads = self.backward_propogation(dYpred,cache)

        for key in self.params:
            self.params[key]+=epsilon
            Ypred,_ = self.forward_propogation(data)
            loss1,_ = self.compute_loss(target,Ypred)
            self.params[key]-=2*epsilon
            Ypred,_ = self.forward_propogation(data)
            loss2,_ = self.compute_loss(target,Ypred)
            first_difference = (loss1 - loss2)/(2*epsilon)
            print ("First_difference= ",first_difference,"   grad[",key,"] =",grads[key])
            self.params[key]+=epsilon


def __main__():
		num_nodes = [1, 1, 1]
		model = Deep_lstm_Network(num_nodes)
		model.grad_check()

__main__()



