import numpy as np

class Deep_Neural_Network:

# ------------------------------------------------------
# Deep Neural Network Constructor
# ----------------------------------------------------------
    def __init__(self,num_nodes):

        self.num_nodes = num_nodes
        self.recurrent_layers = len(num_nodes)-2
        self.params = {}

        for layer in range(self.recurrent_layers):
            self.params['W'+str(layer)] = np.zeros((num_nodes[layer],num_nodes[layer+1]))
            self.params['U'+str(layer)] = np.zeros((num_nodes[layer+1],num_nodes[layer+1]))
            self.params['b'+str(layer)] = np.zeros(num_nodes[layer+1])

        self.params['Wy'] = np.zeros((num_nodes[-2],num_nodes[-1]))
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

        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.num_batches = num_batches
        self.bptt_trunc = bptt_trunc


# ----------------------------------------------------------
# Forward Pass
# ----------------------------------------------------------
    def sigmoid(self,x):
	    return 1.0/(1+np.exp(-x))

    def forward_recurrent_step(self,X,prev_hidden_state,W,U,b):
        act = np.dot(X,W) + np.dot(prev_hidden_state,U) + b
        #using sigmoid activation fn
        hidden_state = self.sigmoid(act)
        cache = (hidden_state,X,prev_hidden_state)
        return hidden_state,cache

    def forward_output_step(self,X,W,b):
        act = np.dot(X,W) + b
        #using sigmoid in output layer
        ypred = self.sigmoid(act)
        cache = (ypred,X)
        return ypred,cache


# ----------------------------------------------------------
    def forward_unit_time(self,hidden_states,X):
        Cache = list()

        #Recurrence stage
        for layer in range(self.recurrent_layers):
            hidden_states[layer],cache = self.forward_recurrent_step(X,hidden_states[layer],
                                                           self.params['W'+str(layer)],
                                                           self.params['U'+str(layer)],
                                                           self.params['b'+str(layer)])
            X = hidden_states[layer]
            Cache.append(cache)

        #Output Stage
        ypred,cache = self.forward_output_step(X,self.params['Wy'],self.params['by'])
        Cache.append(cache)

        return ypred,hidden_states,Cache


# ----------------------------------------------------------
    def forward_propogation(self,X):
        [seq_length, num_data, num_dim] = np.shape(X)
        Cache = list()
        Ypred = list()
        #Set initial states to 0
        hidden_states = [None]*self.recurrent_layers
        for layer in range(self.recurrent_layers):
            hidden_states[layer] = np.zeros((num_data,self.num_nodes[layer+1]))

        for t in range(seq_length):
            ypred,hidden_states,cache = self.forward_unit_time(hidden_states,X[t])
            Cache.append(cache)
            Ypred.append(ypred)

        return Ypred,Cache


# ----------------------------------------------------------
# Loss and its derivate computation
# ----------------------------------------------------------
    def compute_loss(self,Y,Ypred):
        #Currently only supports L2 loss
        [seq_length, num_data, num_dim] = np.shape(Y)
        Ypred = np.asarray(Ypred)
        loss = 0.5*np.sum((Y-Ypred)**2)
        dYpred = (Ypred-Y)
        return loss,dYpred



# ----------------------------------------------------------
# Backward Pass
# ----------------------------------------------------------
    def backward_recurrent_step(self,dh,cache,W,U,b):
        (hidden_state,X,prev_hidden_state)=cache
        grads = {}
        dact   = (hidden_state)*(1-hidden_state)*dh
        grads['W']     = np.dot(X.T,dact)
        grads['b']     = np.sum(dact,axis=0)
        grads['U']     = np.dot(prev_hidden_state.T,dact)
        dhprev = np.dot(dact,U.T)
        dX     = np.dot(dact,W.T)
        
        return dX,dhprev,grads

    def backward_output_step(self,dypred,cache,W,b):
        (ypred,X)=cache
        dact = (ypred)*(1-ypred)*dypred
        dX   = np.dot(dact,W.T)
        dW   = np.dot(X.T,dact)
        db   = np.sum(dact,axis=0)

        return dX,dW,db

# ----------------------------------------------------------
    def backward_unit_time(self,dy,dh,Cache):
        Grads = {}
        #Output Stage
        dx,dWy,dby = self.backward_output_step(dy,Cache[self.recurrent_layers],self.params['Wy'],self.params['by'])
        Grads['Wy'] = dWy
        Grads['by'] = dby
       
        #Recurrence Stage
        for layer in reversed(range(self.recurrent_layers)):
            dy=dx+dh[layer]
            dx,dhprev,grads = self.backward_recurrent_step(dy,Cache[layer],
                                                  self.params['W'+str(layer)],
                                                  self.params['U'+str(layer)],
                                                  self.params['b'+str(layer)])
            dh[layer] = dhprev
            #Update gradients
            for key in grads:
                Grads[key+str(layer)] = grads[key]
            

        return dh,Grads


# ----------------------------------------------------------
    def backward_propogation(self,dY,Cache):
        [seq_length, num_data, num_dim] = np.shape(dY)
        Grads = {}
        #Set initial state derivatives to 0
        dh = [None]*self.recurrent_layers
        for layer in range(self.recurrent_layers):
            dh[layer] = np.zeros((num_data,self.num_nodes[layer+1]))

        for t in reversed(range(seq_length)):
            dh,grads = self.backward_unit_time(dY[t],dh,Cache[t])

            #Update gradients
            for key in grads:
                Grads[key] = Grads.get(key,np.zeros_like(grads[key]))
                Grads[key] += grads[key]

        return Grads


# ----------------------------------------------------------
# Param Update
# ----------------------------------------------------------

    def momentum_update(params,grads,velocity,alpha,momentum):
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
		model = Deep_Neural_Network(num_nodes)
		model.grad_check()

__main__()



