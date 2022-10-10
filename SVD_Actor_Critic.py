import tensorflow as tf
import numpy as np
from math import *
import gym  #requires OpenAI gym installed
from scipy.stats import norm


class environment_piston():
    def __init__(self):
        self.p = np.array([np.random.uniform(-pi/2.0, pi/2.0), np.random.uniform(-pi/2.0, pi/2.0), np.random.uniform(-pi/2.0, pi/2.0)])

    def reset(self):
        self.p = np.array([np.random.uniform(-pi/2.0, pi/2.0), np.random.uniform(-pi/2.0, pi/2.0), np.random.uniform(-pi/2.0, pi/2.0)])
        self.dp = np.array([ self.p[0]-self.p[1], self.p[1]-self.p[2], self.p[2]-self.p[0] ])
        self.state = np.reshape( np.stack([ -self.dp, 2.0*self.dp ] ), [6] )
        return self.state
        
    def step(self,action):
        assert self.p.shape == action.shape
        action[0] = 0.0
        self.p = self.p + action
        self.dp = np.array([ self.p[0]-self.p[1], self.p[1]-self.p[2], self.p[2]-self.p[0] ])
        #self.state = np.reshape( np.stack([ np.sin(self.dp) - self.dp, np.cos(self.dp) + self.dp] ), [6] )
        self.state = np.reshape( np.stack([ -self.dp, 2.0*self.dp ] ), [6] )
        reward = (pi/2.0) - np.abs(self.dp)
        return self.state, reward


env = environment_piston()

A = np.array([[1.0,-1.0,0.0],[0.0,1.0,-1.0],[-1.0,0.0,1.0]])

tf.reset_default_graph()

input_dims = 6
n_outputs = 3
state_placeholder = tf.placeholder(tf.float32, [None, input_dims]) 

def value_function(state):
    n_hidden1 = 400  
    n_hidden2 = 400
    v_outputs = 1
    
    with tf.variable_scope("value_network"):
        init_xavier = tf.contrib.layers.xavier_initializer()
        
        hidden1 = tf.layers.dense(state, n_hidden1, tf.nn.elu, init_xavier)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.elu, init_xavier) 
        V = tf.layers.dense(hidden2, v_outputs, None, init_xavier)
    return V



def policy_network(state):
    n_hidden1 = 40
    n_hidden2 = 40

    with tf.variable_scope("policy_network"):
        init_xavier = tf.contrib.layers.xavier_initializer()
        
        hidden1 = tf.layers.dense(state, n_hidden1, tf.nn.elu, init_xavier)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.elu, init_xavier)
        output_fc = tf.layers.dense(hidden2, n_outputs, None, init_xavier)
        sigma = tf.layers.dense(hidden2, n_outputs, None, init_xavier)
        sigma = tf.nn.softplus(sigma) + 1e-5
        

        a = tf.convert_to_tensor(A, dtype=tf.float32)
        

        s, u, v = tf.linalg.svd(a, full_matrices=False)
        #tf_a_approx = tf.matmul(u_tf, tf.matmul(tf.linalg.diag(s_tf), v_tf, adjoint_b=True))

        zeros = tf.zeros_like(s, dtype=tf.int64)
        ones = tf.ones_like(s, dtype=tf.int64)

        tol_indx = tf.where(tf.greater(s, 1e-14), ones, zeros)
        r = tf.reduce_sum(tol_indx)

        mu = tf.matmul(v[:,:r], tf.divide( tf.matmul(u[:,:r], tf.reshape(output_fc, [-1,1]), adjoint_a=True), tf.reshape(s[:r],[-1,1])))
        norm_dist = tf.contrib.distributions.Normal(tf.reshape(mu, [3]), sigma)
        action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
        #assert tf.reduce_sum(action_tf_var)

        action_tf_var = tf.clip_by_value(action_tf_var, -2.0*pi, 2.0*pi)

    return action_tf_var, norm_dist

def scale_state(state):
    return np.reshape(state,(1,input_dims))

lr_actor = 0.00002  #set learning rates. Defautl: 0.00002
lr_critic = 0.001


# define required placeholders
action_placeholder = tf.placeholder(tf.float32,[n_outputs])
delta_placeholder = tf.placeholder(tf.float32,[n_outputs])
target_placeholder = tf.placeholder(tf.float32)

action_tf_var, norm_dist = policy_network(state_placeholder)
V = value_function(state_placeholder)

# define actor (policy) loss function
loss_actor = -tf.log(norm_dist.prob(action_placeholder) + 1e-5) * delta_placeholder
training_op_actor = tf.train.AdamOptimizer(
    lr_actor, name='actor_optimizer').minimize(loss_actor)

# define critic (state-value) loss function
loss_critic = tf.reduce_mean(tf.squared_difference(
                             tf.squeeze(V), target_placeholder))
training_op_critic = tf.train.AdamOptimizer(
        lr_critic, name='critic_optimizer').minimize(loss_critic)

################################################################
#Training loop
gamma = 0.99        #discount factor
num_episodes = 400000 #300

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    episode_history = []
    baseline = 0

    reward_total = 0 
    steps = 0
    done = False

    for episode in range(num_episodes):
        #receive initial state from E
        state = env.reset()   # state.shape -> (2,)

        #Sample action according to current policy
        #action.shape = (1,1)
        action = sess.run([action_tf_var], feed_dict={state_placeholder: scale_state(state)})
        #Execute action and observe reward & next state from E
        # next_state shape=(2,)    
        #env.step() requires input shape = (1,)
        action = np.reshape(action, [1,-1])
 

        next_state, reward = env.step(np.squeeze(action)) 
        steps +=1
        reward_total += reward


        #V_of_next_state.shape=(1,1)
        V_of_next_state = sess.run(V, feed_dict = {state_placeholder: scale_state(next_state)})  
        #Set TD Target
        #target = r + gamma * V(next_state)     
        target = reward + gamma * np.squeeze(V_of_next_state) 
            
        # td_error = target - V(s)
        #needed to feed delta_placeholder in actor training
        td_error = target - np.squeeze(sess.run(V, feed_dict = {state_placeholder: scale_state(state)})) 
            
        #Update actor by minimizing loss (Actor training)
        _, loss_actor_val  = sess.run([training_op_actor, loss_actor], 
            feed_dict={action_placeholder: np.squeeze(action), 
            state_placeholder: scale_state(state), 
            delta_placeholder: td_error})
        #Update critic by minimizinf loss  (Critic training)
        _, loss_critic_val  = sess.run([training_op_critic, loss_critic], 
            feed_dict={state_placeholder: scale_state(state), 
            target_placeholder: target})
            
        state = next_state
        if steps % 500 == 0:
            print(reward)

