import tensorflow as tf


import numpy as np
EPS = 1e-8
num_states = 5

class environment():
    def __init__(self):
        self.num_states = num_states
        self.state = np.random.randint(0,self.num_states) #Returns a random state
        #List out our bandits. Currently arms 4, 3, and 2 (respectively) are the most optimal.

    def reset(self):
        self.state = np.random.randint(0,self.num_states) #Returns a random state for each episode.
        return self.state
        
    def step(self,action):
        self.state = self.state - action
        reward = -1.0*10.0*((self.state)-0.87)**2.0+5.0
        return self.state, reward




def value_function(state):
    n_hidden1 = 400  
    n_hidden2 = 400
    n_outputs = 1
    
    with tf.variable_scope("value_network"):
        init_xavier = tf.contrib.layers.xavier_initializer()
        
        hidden1 = tf.layers.dense(state, n_hidden1, tf.nn.elu, init_xavier)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.elu, init_xavier) 
        V = tf.layers.dense(hidden2, n_outputs, None, init_xavier)
    return V


def policy_network(state):
    n_hidden1 = 40
    n_hidden2 = 40
    n_outputs = 1
    
    with tf.variable_scope("policy_network"):
        init_xavier = tf.contrib.layers.xavier_initializer()
        
        hidden1 = tf.layers.dense(state, n_hidden1, tf.nn.elu, init_xavier)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.elu, init_xavier)
        mu = tf.layers.dense(hidden2, n_outputs, None, init_xavier)
        sigma = tf.layers.dense(hidden2, n_outputs, None, init_xavier)
        sigma = tf.nn.softplus(sigma) + 1e-5
        norm_dist = tf.contrib.distributions.Normal(mu, sigma)
        action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
        action_tf_var = tf.clip_by_value(action_tf_var, 0.0, num_states)
    return action_tf_var, norm_dist



class agent():
    def __init__(self, lr, s_size,a_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[1],dtype=tf.int32)
        state_in_OH = tf.one_hot(self.state_in,s_size)

        self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1],dtype=tf.float32)

        fc1 = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_in_OH, 0),
                num_outputs=8,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer())
        
        output = tf.contrib.layers.fully_connected(
                inputs=fc1,
                num_outputs=a_size,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer())
        

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        '''
        #Discrete actions:
        self.output = tf.squeeze(tf.nn.softmax(output))
        self.chosen_action = tf.argmax(self.output,0)
        self.responsible_weight = tf.slice(self.output,tf.cast(self.action_holder,tf.int32),[1])
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        '''
        #Continuous actions:
        self.output = tf.squeeze(output)
        self.log_std = tf.get_variable(name='log_std', initializer=-0.5 * np.ones(a_size, dtype=np.float32))
        std = tf.exp(self.log_std)
        self.chosen_action = self.output + tf.random_normal(tf.shape(self.output)) * std
        self.chosen_action = tf.reshape(self.chosen_action,[])
        target_policy = tf.contrib.distributions.Normal(loc=self.output, scale=tf.exp(self.log_std))
        behavior_policy = tf.contrib.distributions.Normal(loc=2.0, scale=2.0)
        #behavior_policy = tf.contrib.distributions.Uniform(low=-5.0, high=8.0)

        importance_weight = tf.div(target_policy.prob(self.action_holder), behavior_policy.prob(self.action_holder) )
        self.loss = -1.0 * tf.log(target_policy.prob(self.action_holder)) * self.reward_holder
     

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # Compute the gradients for a list of variables.
        grads_and_vars = optimizer.compute_gradients(self.loss)
        # Ask the optimizer to apply the capped gradients.
        self.update = optimizer.apply_gradients(grads_and_vars)

        #self.update = optimizer.minimize(self.loss)


tf.reset_default_graph() #Clear the Tensorflow graph.
env = environment() #Load the bandits.
a_size = 1  #env.num_actions
#myAgent = agent(lr=0.001,s_size=env.num_states,a_size=a_size) #Load the agent.
myAgent = agent(lr=0.001,s_size=env.num_states,a_size=a_size) #Load the agent.
weights = tf.trainable_variables()[0] #The weights we will evaluate to look into the network.

total_episodes = 40000 #Set total number of episodes to train agent on.
total_reward = np.zeros([env.num_states]) #Set scoreboard for bandits to 0.

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 1
    while i < total_episodes:
        s = env.reset() #Get a state from the environment.
        
        #Choose either a random action or one from our network
        if False:
            #action = np.random.uniform(low=-5.0, high=8.0, size=None)
            action = np.random.normal(loc=2.0, scale=2.0, size=None)
            _, reward = env.step(action) #Get our reward for taking an action
 
            #Update the network.
            feed_dict={myAgent.reward_holder:[reward],myAgent.action_holder:[action],myAgent.state_in:[s]}
            _,ww = sess.run([myAgent.update,weights], feed_dict=feed_dict)
            if i % 500 == 0:
              print("Behavior policy: state-> " + str(s) + " action-> " + str(action) + " reward-> " + str(reward))


        else:
            action = sess.run(myAgent.chosen_action,feed_dict={myAgent.state_in:[s]})
            _, reward = env.step(action) #Get our reward for taking an action.

            #Update the network.
            feed_dict={myAgent.reward_holder:[reward],myAgent.action_holder:[action],myAgent.state_in:[s]}
            _,ww = sess.run([myAgent.update,weights], feed_dict=feed_dict)
            if i % 500 == 0:
              print("Target policy: state-> " + str(s) + " action-> " + str(action) + " reward-> " + str(reward))            


        i += 1



'''
def gaussian_probability(x, mu, log_std):
    #loc = mu | scale = sigma
    dist = tf.contrib.distributions.Normal(loc=mu, scale=tf.exp(log_std))
    prob = dist.prob(x)
    return prob



def gaussian_likelihood(x, mu, log_std):
    #loc = mu | scale = sigma
    dist = tf.contrib.distributions.Normal(loc=mu, scale=tf.exp(log_std))
    log_prob = tf.log ( dist.prob(x) )
    #log_prob = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2.0 + 2.0*log_std + np.log(2.0*np.pi))
    return log_prob
'''
