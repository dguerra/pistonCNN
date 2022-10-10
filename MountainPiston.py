import tensorflow as tf
import numpy as np
import scipy.stats as st
from foo import environment
from math import *


'''
cap = 5
class environment():
    def __init__(self):
        self.state = np.array( [np.random.uniform(-cap, cap), np.random.uniform(-cap, cap), np.random.uniform(-cap, cap)] ) #Returns a random state

    def reset(self):
        self.state = np.array( [np.random.uniform(-cap, cap), np.random.uniform(-cap, cap), np.random.uniform(-cap, cap)] ) #Returns a random state
        self.dp = np.array([self.state[0]-self.state[1], self.state[1]-self.state[2], self.state[2]-self.state[0]])
        return self.dp
        
    def step(self,action):
        self.state[0] = self.state[0] + action[0]
        self.state[1] = self.state[1] + action[1]
        #reward = -1.0*10.0*((self.state)-0.87)**2.0+5.0
        self.dp = np.array([self.state[0]-self.state[1], self.state[1]-self.state[2], self.state[2]-self.state[0]])
        #reward = 5.0 - np.square( np.sum( np.abs(self.dp) ) )
        reward = 5.0 - np.square( np.sum( np.abs(self.dp) ) )
        return self.dp, reward
'''


env = environment()

tf.reset_default_graph()
CHANNELS = 1
action_dims = 2
input_dims = 24   #49 #24  #3
state_placeholder = tf.placeholder(tf.float32, [None, input_dims, input_dims, CHANNELS]) 
#state_placeholder = tf.placeholder(tf.float32, [None, input_dims]) 

#relation between outputs
A = np.array([[1.0,-1.0, 0.0],[0.0,1.0,-1.0],[-1.0,0.0,1.0]])

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
    n_hidden1 = 128 #40
    n_hidden2 = 128 #40
    
    with tf.variable_scope("policy_network"):
        init_xavier = tf.contrib.layers.xavier_initializer()
        
        hidden1 = tf.layers.dense(state, n_hidden1, tf.nn.elu, init_xavier)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.elu, init_xavier)
        mu = tf.layers.dense(hidden2, action_dims, None, init_xavier)
       
        '''
        a = tf.convert_to_tensor(A, dtype=tf.float32)
        s, u, v = tf.linalg.svd(a, full_matrices=False)
        #tf_a_approx = tf.matmul(u_tf, tf.matmul(tf.linalg.diag(s_tf), v_tf, adjoint_b=True))
        zeros = tf.zeros_like(s, dtype=tf.int64)
        ones = tf.ones_like(s, dtype=tf.int64)
        tol_indx = tf.where(tf.greater(s, 1e-14), ones, zeros)
        r = tf.reduce_sum(tol_indx)
        mu_svd = tf.matmul(v[:,:r], tf.divide( tf.matmul(u[:,:r], tf.reshape(mu, [-1,1]), adjoint_a=True), tf.reshape(s[:r],[-1,1])))
        mu_svd = tf.reshape(mu_svd, [action_dims])
        '''

        sigma = tf.layers.dense(hidden2, action_dims, None, init_xavier)
        sigma = tf.nn.softplus(sigma) + 1e-5
        norm_dist = tf.contrib.distributions.Normal(mu, sigma)
        action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
        #action_tf_var = tf.clip_by_value( action_tf_var, -2.0*cap, 2.0*cap)
    return action_tf_var, norm_dist


def value_function_cnn(input_cnn, branch_name, num_filters=32):  
  conv1 = tf.layers.conv2d(inputs=input_cnn, filters=num_filters, kernel_size=7, activation=tf.nn.relu, use_bias=True, \
    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer(), name="conv1"+branch_name)
  
  conv2 = tf.layers.conv2d(inputs=conv1, filters=num_filters, kernel_size=5, activation=tf.nn.relu, use_bias=True, \
    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer(), name="conv2"+branch_name)

  conv3 = tf.layers.conv2d(inputs=conv2, filters=num_filters, kernel_size=3, activation=tf.nn.relu, use_bias=True, \
    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer(), name="conv3"+branch_name)

  output_cnn = tf.layers.max_pooling2d(conv3, 2, 2)
  conv_rgr = tf.contrib.layers.flatten(output_cnn)

  v_rgr = tf.layers.dense(inputs=conv_rgr, units=1, name="v_rgr", activation=None, use_bias=True, \
    kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer() )

  return v_rgr



def inference_cnn(input_cnn, branch_name, num_filters=32):  
  with tf.variable_scope("inference_cnn"):
      conv1 = tf.layers.conv2d(inputs=input_cnn, filters=num_filters, kernel_size=7, activation=tf.nn.relu, use_bias=True, \
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer(), name="conv1"+branch_name)
  
      conv2 = tf.layers.conv2d(inputs=conv1, filters=num_filters, kernel_size=5, activation=tf.nn.relu, use_bias=True, \
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer(), name="conv2"+branch_name)

      conv3 = tf.layers.conv2d(inputs=conv2, filters=num_filters, kernel_size=3, activation=tf.nn.relu, use_bias=True, \
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer(), name="conv3"+branch_name)

      output_cnn = tf.layers.max_pooling2d(conv3, 2, 2)
      conv_rgr = tf.contrib.layers.flatten(output_cnn)

      mu_rgr = tf.layers.dense(inputs=conv_rgr, units=action_dims, name="mu_rgr"+branch_name, activation=None, use_bias=True, \
        kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer() )

      #sigma_rgr = tf.layers.dense(inputs=conv_rgr, units=action_dims, name="sigma_rgr"+branch_name, activation=None, use_bias=True, \
      #  kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer() )
      #sigma_rgr = tf.nn.softplus(sigma_rgr) + 1e-5

      norm_dist = tf.contrib.distributions.Normal(mu_rgr, 0.1 )
      action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
  
  return action_tf_var, norm_dist

################################################################
#sample from state space for state normalization
#import sklearn
#import sklearn.preprocessing
                                    
#state_space_samples = np.array(
#    [env.observation_space.sample() for x in range(10000)])
#scaler = sklearn.preprocessing.StandardScaler()
#scaler.fit(state_space_samples)

#function to normalize states
#def scale_state(state):                 #requires input shape=(2,)
#    scaled = scaler.transform([state])
#    return scaled                       #returns shape =(1,2)   
###################################################################

def scale_state(state):
    return np.reshape(state,(1,input_dims, input_dims, CHANNELS))

rval = np.array([np.random.uniform(-0.5 * pi, 0.5 * pi), np.random.uniform(-0.5 * pi, 0.5 * pi)])

def the_func(act, delt):
    rrr = (5.0 - np.sum(np.absolute(act - d)))
    return np.array(rrr, ndmin=1, dtype=np.float32)

lr_actor = 0.00002  #set learning rates
lr_critic = 0.001

# define required placeholders
action_placeholder = tf.placeholder(tf.float32, [action_dims])
delta_placeholder = tf.placeholder(tf.float32)
target_placeholder = tf.placeholder(tf.float32)
reward_placeholder = tf.placeholder(tf.float32)

#action_tf_var, norm_dist = policy_network(state_placeholder)
action_tf_var, norm_dist = inference_cnn(state_placeholder, "regression", num_filters=64) #64

#V = value_function(state_placeholder)
#V = value_function_cnn(state_placeholder, "regressionV", num_filters=64)  #64 
norm_dist_prob =   39894.228040143265    #norm_dist.prob(action_tf_var)
#loss_actor = tf.losses.mean_squared_error(tf.reshape(action_tf_var, [action_dims]), tf.reshape(delta_placeholder, [action_dims]))
# define actor (policy) loss function

#rew = tf.py_function(func=the_func, inp=[action_tf_var], Tout=tf.float32)

act1 = tf.constant(0.0)  #tf.reduce_sum(tf.abs(action_tf_var - action_placeholder))

#rew = (5.0 - tf.reduce_sum(tf.abs(tf.reshape(action_tf_var,[action_dims]) - delta_placeholder)))

#asrt = tf.Assert(tf.equal(rew,reward_placeholder), [rew,reward_placeholder])
asrt = tf.constant(0.0)   #tf.Assert(tf.equal(act1,0.0), [action_tf_var, action_placeholder])
#loss_actor = tf.losses.mean_squared_error(action_tf_var, delta_placeholder)

#loss_actor = tf.reduce_mean(-tf.log(norm_dist.prob(action_tf_var) + 1e-5) * rew ) #delta_placeholder
loss_actor = tf.reduce_mean(-tf.log(norm_dist.prob(action_placeholder) + 1e-5) *  reward_placeholder)
training_op_actor = tf.train.AdamOptimizer(lr_actor, name='actor_optimizer').minimize(loss_actor)

# define critic (state-value) loss function
#loss_critic = tf.reduce_mean(tf.squared_difference(tf.squeeze(V), target_placeholder))
#training_op_critic = tf.train.AdamOptimizer(lr_critic, name='critic_optimizer').minimize(loss_critic)
################################################################
#Training loop
gamma = 0.99        #discount factor
num_episodes = 1000000# 300

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    episode_history = []
    reward_total = 0
    display_rate = 100
    for episode in range(num_episodes):
        #receive initial state from E
        rval = np.array([np.random.uniform(-0.5 * pi, 0.5 * pi), np.random.uniform(-0.5 * pi, 0.5 * pi)])
        state = env.reset(rval)   # state.shape -> (2,) 
        steps = 0
        done = False
        if True:
        #if episode % 200 != 0:
                
            #Sample action according to current policy
            #action.shape = (1,1)
            action  = sess.run([action_tf_var], feed_dict={state_placeholder: scale_state(state)})
            #Execute action and observe reward & next state from E
            # next_state shape=(2,)    
            #env.step() requires input shape = (1,)
         
            #action = np.random.uniform(np.full([action_dims], -2.0*cap), np.full([action_dims], 2.0*cap))
            #action = target_dist.rvs()
            #next_state, reward = env.step(np.reshape(action, [action_dims]))

            #imp_ratio = 4.0 * cap * st.norm(np.resize(m,[action_dims]) ,np.resize(s,[action_dims])).pdf(action)
            #reward = (reward * imp_ratio)

            #steps +=1
            #reward_total += reward
            '''
            #Shape of V_of_next_state is (1,1)
            V_of_next_state = sess.run(V, feed_dict = {state_placeholder: scale_state(next_state)})  
            #Set TD Target     
            target = reward + gamma * np.squeeze(V_of_next_state) 
            
            #needed to feed delta_placeholder in actor training
            td_error = target - np.squeeze(sess.run(V, feed_dict = {state_placeholder: scale_state(state)})) 
            '''
            rwrd = 5.0 - np.sum( np.absolute(np.reshape(action, [action_dims]) - rval) )
            #Update actor by minimizing loss (Actor training)
            _, loss_actor_val, _ = sess.run([training_op_actor, loss_actor, asrt], 
                feed_dict={
                action_placeholder: np.reshape(action, [action_dims]), 
                state_placeholder: scale_state(state), 
                delta_placeholder: rval, 
                reward_placeholder: rwrd })  #  td_error })

            '''
            #Update critic by minimizinf loss  (Critic training)
            _, loss_critic_val  = sess.run([training_op_critic, loss_critic], 
                feed_dict={state_placeholder: scale_state(state), 
                target_placeholder: target})
            '''
            
            #state = next_state
            #end while
            episode_history.append(reward_total)
            if episode % display_rate == 0:
                print( episode, steps, loss_actor_val, action, rval, action, rwrd) #reward_total/display_rate )
                reward_total = 0
        '''
        else:
            action  = sess.run([action_tf_var], feed_dict={state_placeholder: scale_state(state)})
            next_state, reward = env.step(np.reshape(action, [action_dims]))
            if episode % 100 == 0:
                print( reward )

        '''
