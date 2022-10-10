import tensorflow as tf
import numpy as np
from math import *
#from foo import environment
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import triang
import PistonTools as pt
from Mirror import Mirror
from foo import environment
from simulacion_piston import axial_to_matrix


#LOGDIR = "/output/phasedivcnn/1"

nrings = 1
cap = pi/2.0
HEIGHT = 256  #1024 #number of samples in each direction in the pupil plane [pixels]
CHANNELS = 4
LAMBDA_RANGES = 1

tf.reset_default_graph()

input_dims = 24   #3   #2    #24  #49  #12 
action_dims = 2
batch_size = 6*(nrings**2)    #8 # 16  #1024  #8192  #4096

#state_placeholder = tf.placeholder(tf.float32, [batch_size, input_dims]) 
state_placeholder = tf.placeholder(tf.float32, [batch_size, input_dims, input_dims, CHANNELS])
#state_placeholder = tf.placeholder(tf.float32, [input_dims, input_dims, CHANNELS])

#mirror = Mirror(nrings, HEIGHT, CHANNELS, LAMBDA_RANGES)
env = environment(CHANNELS)

def policy_network(state):
    n_hidden1 = 800
    n_hidden2 = 800
    with tf.variable_scope("policy_network"):
        init_xavier = tf.contrib.layers.xavier_initializer()
        reg = tf.contrib.layers.l2_regularizer(scale=0.001)
        #reg = tf.contrib.layers.l2_regularizer(scale=0.9999)   #to get linear regression, regularization should be very high

        '''        
        h1_d = tf.layers.conv2d(state, filters=64, kernel_size=5, activation=tf.nn.relu, kernel_initializer=init_xavier, kernel_regularizer=reg, padding='same')
        h2_d = tf.layers.conv2d(h1_d, filters=64, kernel_size=3, activation=tf.nn.relu, kernel_initializer=init_xavier, kernel_regularizer=reg, padding='same')
        h3_d = tf.layers.conv2d(h2_d, filters=64, kernel_size=3, activation=tf.nn.relu, kernel_initializer=init_xavier, kernel_regularizer=reg, padding='same')
        output_d = tf.layers.flatten(h3_d)
        softmax_input = tf.layers.dense(output_d, int((LAMBDA_RANGES*4) * action_dims), None, init_xavier)
        action_probs = tf.nn.softmax( tf.reshape(softmax_input,[batch_size * action_dims, int(LAMBDA_RANGES*4)]) ) 
        '''


        h1_c = tf.layers.conv2d(state, filters=64, kernel_size=5, activation=tf.nn.relu, kernel_initializer=init_xavier, kernel_regularizer=reg, padding='same')
        h2_c = tf.layers.conv2d(h1_c, filters=64, kernel_size=3, activation=tf.nn.relu, kernel_initializer=init_xavier, kernel_regularizer=reg, padding='same')
        h3_c = tf.layers.conv2d(h2_c, filters=64, kernel_size=3, activation=tf.nn.relu, kernel_initializer=init_xavier, kernel_regularizer=reg, padding='same')
        output_c = tf.layers.flatten(h3_c)
        
        mu = tf.layers.dense(output_c, action_dims, None, init_xavier)
        #mu = tf.multiply( tf.nn.sigmoid(mu), pi )
        sigma = tf.layers.dense(output_c, 1, None, init_xavier)
        sigma = tf.add( tf.nn.softplus(sigma), 1e-5 )
        
        norm_dist = tf.contrib.distributions.Normal(mu, sigma)
        action_tf_var = tf.squeeze(norm_dist.sample([1]), axis=0)
        action_tf_var = tf.clip_by_value(action_tf_var, -cap, cap)
    return action_tf_var, norm_dist, mu, sigma  #, action_probs


def value_function(state):
    n_hidden1 = 400  
    n_hidden2 = 400
    n_outputs = 1
    
    with tf.variable_scope("value_network"):
        init_xavier = tf.contrib.layers.xavier_initializer()
        reg = tf.contrib.layers.l2_regularizer(scale=0.001)
        #reg = tf.contrib.layers.l2_regularizer(scale=0.9999)   #to get linear regression, regularization should be very high
        '''
        hidden1 = tf.layers.dense(state, n_hidden1, tf.nn.relu, init_xavier, kernel_regularizer=reg)
        hidden1 = tf.layers.batch_normalization(hidden1)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.relu, init_xavier, kernel_regularizer=reg)
        hidden2 = tf.layers.batch_normalization(hidden2)        
        '''
        hidden1 = tf.layers.conv2d(state, filters=128, kernel_size=5, activation=tf.nn.elu, kernel_initializer=init_xavier, kernel_regularizer=reg, padding='same')
        hidden1n = tf.layers.batch_normalization(hidden1)
        hidden2 = tf.layers.conv2d(hidden1n, filters=128, kernel_size=3, activation=tf.nn.elu, kernel_initializer=init_xavier, kernel_regularizer=reg, padding='same')
        hidden2n = tf.layers.batch_normalization(hidden2)
        hidden3 = tf.layers.conv2d(hidden2n, filters=128, kernel_size=3, activation=tf.nn.elu, kernel_initializer=init_xavier, kernel_regularizer=reg, padding='same')
        hidden3n = tf.layers.batch_normalization(hidden3)
        hidden3 = tf.layers.flatten(hidden3n)
        

        V = tf.layers.dense(hidden3, n_outputs, None, init_xavier)
    return V

################################################################
#function to normalize states
def scale_state(state):                 #requires input shape=(2,)
    #return np.reshape(state,[input_dims, input_dims, CHANNELS])
    return np.reshape(state,[batch_size, input_dims, input_dims, CHANNELS])                       #returns shape =(1,2)  
#return np.reshape(state,[batch_size,input_dims]) 
###################################################################
#lr_actor = 0.000059 (action_dim = 1) #0.00002  #set learning rates

lr_actor = 0.000008
lr_critic = 0.0008

#lr_actor = 0.000008
#lr_critic = 0.008

# define required placeholders
action_placeholder = tf.placeholder(tf.float32, [batch_size, action_dims])
delta_placeholder = tf.placeholder(tf.float32, [batch_size])
target_placeholder = tf.placeholder(tf.float32, [batch_size])
#picked_action_placeholder = tf.placeholder(tf.int32, [batch_size * action_dims, 1])

action_tf_var, norm_dist, mu, sigma = policy_network(state_placeholder )
V = value_function(state_placeholder)

#picked_action_prob = tf.batch_gather(action_probs, picked_action_placeholder)

a_continuos = norm_dist.prob(action_placeholder)
#a_discrete = tf.reshape(picked_action_prob, [batch_size, action_dims])

#a_all = tf.concat([a_discrete, a_continuos],axis=1)

# define actor (policy) loss function 
loss_actor = -tf.log(tf.reduce_prod(a_continuos, axis=1) + 1e-5 ) * delta_placeholder

#loss_actor = tf.reduce_mean( log_prob )
#loss_actor = tf.losses.mean_squared_error(action_tf_var, delta_placeholder)
training_op_actor = tf.train.AdamOptimizer(lr_actor, name='actor_optimizer').minimize(loss_actor)
# define critic (state-value) loss function
loss_critic = tf.reduce_mean(tf.squared_difference(tf.squeeze(V), target_placeholder))
training_op_critic = tf.train.AdamOptimizer(lr_critic, name='critic_optimizer').minimize(loss_critic)
################################################################
#Training loop
gamma = 0.99        #discount factor
num_episodes = 1000000 #300
display_rate = 100
reward_total = 0.0
watch_total = 0.0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    episode_history = []
    rs = np.random.uniform(-cap, cap, [batch_size, action_dims])
    for episode in range(num_episodes):
        #if episode % display_rate != 0:
        if True:
            #receive initial state from E
            #pstn = np.random.uniform(-LAMBDA_RANGES * pi, LAMBDA_RANGES * pi, [(2*nrings)+1, (2*nrings)+1])
            #pstn = np.random.uniform(-cap, cap, [(2*nrings)+1, (2*nrings)+1])
            #pstn = (np.random.randint(3, size= ((2*nrings)+1, (2*nrings)+1)) - 1 ) * pi * 0.9
            #pstn[axial_to_matrix(0, 0, nrings)] = 0.0
            #rs = pt.stack_piston_steps(pstn, nrings)
            #state = mirror.create_data(pstn)


            #rs = (np.random.randint(2, size=(batch_size, action_dims)) - 1) * 2.0*pi
             
            #rs = triang(0.5, loc=-2.0*cap, scale=4.0*cap).rvs([batch_size, action_dims])

            
            
            state_batch = []
            for i_s in rs:
                state_i, _ = env.reset(i_s)
                state_batch.append(state_i)

            state = np.stack(state_batch, axis=0)
            

            #print(np.sum(state))
            #Sample action according to current policy
            action, m, s  = sess.run( [action_tf_var, mu, sigma], feed_dict={state_placeholder: scale_state(state)})
            #act_probs  = sess.run( [action_tf_var, mu, sigma, action_probs], feed_dict={state_placeholder: scale_state(state)})
            #Ignore current policy and sample action from behaviour policy
            #behav_dist = uniform(-cap, 2.0 * cap)
            #pstn_act = np.random.uniform(-cap, cap, [(2*nrings)+1, (2*nrings)+1])
            #pstn_act[axial_to_matrix(0, 0, nrings)] = 0.0
            #rs_act = pt.stack_piston_steps(pstn_act, nrings)

            #behav_dist = triang(0.5, loc=-2.0*cap, scale=4.0*cap)
            #action = behav_dist.rvs([batch_size, action_dims])


            '''
            picked_action_batch = []
            for da in act_probs:
                picked_action_batch.append( np.random.choice(np.arange(len(da)), p=da) )

            picked_action = np.stack(picked_action_batch, axis=0)
            '''
            
            #Build complete action:
            '''
            lambda_range = np.reshape(picked_action, [batch_size, action_dims])
            even = (np.mod(lambda_range, 2) == 0)
            lambda_val = action
            lambda_val[np.where(even == False)] = pi - action[np.where(even == False)]
            final_pstn = ((lambda_range * pi) + lambda_val) - ((LAMBDA_RANGES*2.0) * pi)
            final_pstn = np.reshape(final_pstn, [batch_size, action_dims])
            '''

            #Execute action and observe reward & next state from E
            baseline = 0.5
            next_rs = rs - action
            watch = -np.mean(np.square(next_rs), axis=1) + baseline   #reward shape is (batch_size,)

            
            next_state_batch = []
            reward_batch = []
            for i_ns in next_rs:
                next_state_i, reward_i = env.reset(i_ns)
                next_state_batch.append(next_state_i)
                reward_batch.append(reward_i)

            next_state = np.stack(next_state_batch, axis=0)
            reward = np.stack(reward_batch, axis=0)
            
            V_of_next_state = sess.run(V, feed_dict = {state_placeholder: scale_state(next_state)})  
            #Set TD Target
            #target = r + gamma * V(next_state)     
            target = reward + gamma * np.squeeze(V_of_next_state) 
            
            # td_error = target - V(s)
            #needed to feed delta_placeholder in actor training
            td_error = target - np.squeeze(sess.run(V, feed_dict = {state_placeholder: scale_state(state)})) 
            

            #target_dist = norm(m,s)
            #watch = watch * np.divide( np.prod(target_dist.pdf( action ), axis=1), np.prod(behav_dist.pdf( action ), axis=1) )
            
            reward_total += reward
            watch_total += watch
            #Update actor by minimizing loss (Actor training)
            _, loss_actor_val  = sess.run([training_op_actor, loss_actor], 
                    feed_dict={action_placeholder: action,
                    state_placeholder: scale_state(state), 
                    #picked_action_placeholder: np.reshape(picked_action,[batch_size * action_dims, 1]),
                    delta_placeholder: td_error })  # td_error})   # reward })
            
            #Update critic by minimizinf loss  (Critic training)
            _, loss_critic_val  = sess.run(
                [training_op_critic, loss_critic], 
                feed_dict={state_placeholder: scale_state(state), 
                target_placeholder: target})
            
            
            if episode % display_rate == 0:
                print("*"+str(episode).replace(".", ",")+"*"+str(np.mean(reward_total)/display_rate).replace(".", ",")+"*"+str(np.mean(watch_total)/display_rate)).replace(".", ",")+"*")
                      
                reward_total = 0.0
                watch_total = 0.0
          
        '''
        else:
            #receive initial state from E
            pstn = np.random.uniform(-cap, cap, [(2*nrings)+1, (2*nrings)+1])
            rs = pt.stack_piston_steps(pstn, nrings)
            state = rs  #mirror.create_data(pstn)
            
            #rs = np.random.uniform(-cap, cap, [batch_size, action_dims])
            #state = rs

            #Sample action according to current policy
            action  = sess.run(action_tf_var, feed_dict={state_placeholder: scale_state(state)})
            action = np.reshape(action, [batch_size, action_dims])
            #Execute action and observe reward & next state from E
            reward = -np.mean(np.absolute(rs - action), axis=1)
            #reward = env.step(np.reshape(action, [action_dims])) 
            reward_total += reward
            #print(reward)
            if episode % display_rate == 0:
                print(episode, np.mean(reward_total)	)
                reward_total = 0.0

        
        '''
