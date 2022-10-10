from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import numpy as np
from math import *
#from foo import environment
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import triang
from simulacion_piston import mirror_wavefront
from simulacion_piston import introduce_atmosphere
from FourierOptics import propTF
from FourierOptics import focus
from FourierOptics import propFF

from simulacion_piston import axial_to_matrix
from simulacion_piston import axial_to_matrix
from simulacion_piston import hex_to_pixel


class environment():
    def __init__(self, nrings, HEIGHT, CHANNELS):
        self.nrings = nrings
        self.physical_hex_size = 0.95
        self.LAMBDA_RANGES = 0.5
        self.CHANNELS = CHANNELS
        self.pstn = np.zeros([(2*self.nrings)+1, (2*self.nrings)+1])     # np.zeros([(2*nrings)+1, (2*nrings)+1])
        dq = 0
        dr = 0

        self.L = float ( (self.physical_hex_size / 2.0) * 2 * ((3 * self.nrings) + 2) )    # physical side length of pupil plane in [m]
        self.HEIGHT = HEIGHT

        self.hex_size = min( floor( (self.HEIGHT/2.0)/(1.0 + (self.nrings * 3.0/2.0)) ), floor( (self.HEIGHT / ((self.nrings*2.0) + 1.0))*(2.0/sqrt(3.0))*(1.0/2.0) ))

        lambda0 = 700.e-9
        self.wlambda = [lambda0, lambda0 * 0.930, lambda0 * 0.860, lambda0 * 0.790]

        xi_factor = 2.5     #proportion of difraction signal width over image blur due to the turbulence
        self.r0 = 0.2            #worst atmospheric r0 considered
        self.z0  = 0.8*(self.r0**2.0)*(xi_factor**(-2.0))*(self.wlambda[3]**(7.0/5.0))/(500.e-9**(12.0/5.0))    #propagation distance (previously 15875.0)

        Ax, Ay = hex_to_pixel(dq, dr, self.hex_size)
        Bx, By = hex_to_pixel(dq,   dr-1, self.hex_size)
        Cx, Cy = hex_to_pixel(dq+1, dr-1, self.hex_size)
        self.pupscale = (self.physical_hex_size * 100.0) / self.hex_size     #[cm/pixel]
        
        self.crop_size = (2.0/4.0) * self.hex_size    #(2.0/4.0)*hex_size   #(3.0/4.0)*hex_size   #(3.0/2.0)*hex_size   #hex_height   #glimpse square crop
        print("crop size.", self.crop_size)

        x = np.linspace(-(self.pupscale * self.crop_size * 0.5), (self.pupscale * self.crop_size * 0.5), num=int(self.crop_size), endpoint=False)
        X, Y =  np.meshgrid(x, x, indexing='xy')
        self.foc_mask = np.sqrt((X * X) + (Y * Y))
        mask_radious = 10.0    #in [cm] if r0 = 0.2 it means a diameter of 20 cm and radiuous 10cm
        self.foc_mask[np.where(self.foc_mask<mask_radious)] = 1.0
        self.foc_mask[np.where(self.foc_mask>mask_radious)] = np.nan

        #Compute location and size of the crop window
        center_width = (Ax + Bx + Cx)/3.0   #the origin is the center of the image
        center_height  = (Ay + By + Cy)/3.0  #the origin is the center of the image
        self.corner_height = center_height - (self.crop_size/2.0) + (self.HEIGHT/2.0)
        self.corner_width = center_width - (self.crop_size/2.0) + (self.HEIGHT/2.0)
        print(self.pstn)


    def reset(self, vv, imageI=True, PSF=True):
        dq = 0
        dr = 0
        self.pstn[axial_to_matrix(dq,   dr-1, self.nrings)] = vv[0]  #upper hexagon
        self.pstn[axial_to_matrix(dq+1, dr-1, self.nrings)] = vv[1]  #upper RIGHT hexagon

        wf = mirror_wavefront(self.L, self.HEIGHT, self.nrings, self.pstn, np.zeros_like(self.pstn), np.zeros_like(self.pstn))
        wf_foc = wf[int(self.corner_height):int(self.corner_height+self.crop_size),int(self.corner_width):int(self.corner_width+self.crop_size)]
        
        r0_500nm = np.random.uniform(self.r0, 0.2)
        zf = 10000

        ax = np.random.randint(2)
        act_mask = self.foc_mask  #np.roll(self.foc_mask, np.random.randint(3)-1, axis=ax)

        oi = []
        strehl = 0.0

        if imageI==True:
          for indx in range(self.CHANNELS):
              u = propTF(np.nan_to_num(np.exp(1j* (self.wlambda[0] / self.wlambda[indx]) * wf)), self.L, self.wlambda[indx], self.z0)
              i = np.absolute(u**2.0)
              i = introduce_atmosphere(i, r0_500nm, self.HEIGHT, self.L, self.wlambda[indx], self.z0)
              i = i[int(self.corner_height):int(self.corner_height+self.crop_size),int(self.corner_width):int(self.corner_width+self.crop_size)]
              oi.append(i)

          ii = np.stack(oi, axis=-1)
          ii = np.array(ii, ndmin=3, dtype=np.float32)
        else:
          ii = _

        if PSF==True:
          for indx in range(self.CHANNELS):
              uin = np.nan_to_num(act_mask * np.exp(1j* (self.wlambda[0] / self.wlambda[indx]) * wf_foc))
              u2 = propFF(uin, (self.pupscale * self.crop_size) / 100, self.wlambda[indx], zf)    # find psf by fraunhofer pattern
              I_foc = np.absolute(u2**2.0)
              strehl += np.amax(I_foc)
       
        else:
          strehl = 0.0

        return ii, strehl


#LOGDIR = "/output/phasedivcnn/1"

nrings = 1
cap = (pi/2.0)
HEIGHT = 256 #512 #256  #1024 #number of samples in each direction in the pupil plane [pixels]
CHANNELS = 4
LAMBDA_RANGES = 1

tf.reset_default_graph()

input_dims = 24   #49 #24   #3   #2    #24  #49  #12 
action_dims = 2
batch_size = 12 #6*(nrings**2)    #8 # 16  #1024  #8192  #4096

#state_placeholder = tf.placeholder(tf.float32, [batch_size, input_dims]) 
state_placeholder = tf.placeholder(tf.float32, [batch_size, input_dims, input_dims, CHANNELS])
#state_placeholder = tf.placeholder(tf.float32, [input_dims, input_dims, CHANNELS])

#mirror = Mirror(nrings, HEIGHT, CHANNELS, LAMBDA_RANGES)
env = environment(nrings, HEIGHT, CHANNELS)

def policy_network(state):
    n_hidden1 = 800
    n_hidden2 = 800
    with tf.variable_scope("policy_network"):
        init_xavier = tf.contrib.layers.xavier_initializer()
        #reg = tf.contrib.layers.l2_regularizer(scale=0.001)
        #reg = tf.contrib.layers.l2_regularizer(scale=0.9999)   #to get linear regression, regularization should be very high

        
        h1_d = tf.layers.conv2d(state, filters=8, kernel_size=7, activation=tf.nn.relu, kernel_initializer=init_xavier)
        h2_d = tf.layers.conv2d(h1_d, filters=8, kernel_size=5, activation=tf.nn.relu, kernel_initializer=init_xavier)
        h3_d = tf.layers.conv2d(h2_d, filters=8, kernel_size=3, activation=tf.nn.relu, kernel_initializer=init_xavier)
        output_d = tf.layers.flatten(h3_d)
        #softmax_input = tf.layers.dense(output_d, int((LAMBDA_RANGES*4) * action_dims), None, init_xavier)
        unnorm_logprob = tf.layers.dense(output_d, int(2 * action_dims), None, init_xavier)
        #action_probs = tf.nn.softmax( tf.reshape(unnorm_logprob,[batch_size * action_dims, int(LAMBDA_RANGES*4)]) )
        action_probs = tf.nn.softmax( tf.reshape(unnorm_logprob,[batch_size * action_dims, int(2)]) ) 

        action_disc_var = tf.random.categorical(tf.math.log(action_probs), 1)
        

        h1_c = tf.layers.conv2d(inputs=state, filters=16, kernel_size=7, activation=tf.nn.relu, use_bias=True, \
          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer())
        h2_c = tf.layers.conv2d(inputs=h1_c, filters=16, kernel_size=5, activation=tf.nn.relu, use_bias=True, \
          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer())
        h3_c = tf.layers.conv2d(inputs=h2_c, filters=16, kernel_size=3, activation=tf.nn.relu, use_bias=True, \
          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias_initializer=tf.zeros_initializer())
        #output_cnn_hc = tf.layers.max_pooling2d(h3_c, 2, 2)
        output_c = tf.layers.flatten(h3_c)
        
        mu = tf.layers.dense(output_c, action_dims, None, init_xavier)

        #mu = tf.multiply( tf.nn.sigmoid(mu), pi )

        sh1_c = tf.layers.conv2d(state, filters=8, kernel_size=7, activation=tf.nn.relu, kernel_initializer=init_xavier)
        sh2_c = tf.layers.conv2d(sh1_c, filters=8, kernel_size=5, activation=tf.nn.relu, kernel_initializer=init_xavier)
        sh3_c = tf.layers.conv2d(sh2_c, filters=8, kernel_size=3, activation=tf.nn.relu, kernel_initializer=init_xavier)
        s_output_c = tf.layers.flatten(sh3_c)

        sigma = tf.layers.dense(s_output_c, 1, None, init_xavier)
        sigma = tf.clip_by_value(sigma, 0.15, 0.15)  #tf.add( tf.nn.softplus(sigma), 1e-5 )  #tf.clip_by_value(sigma, 0.3, 0.3) 

        
        norm_dist = tf.contrib.distributions.Normal(mu, sigma)
        action_tf_var = tf.squeeze(norm_dist.sample([1]), axis=0)
        #action_tf_var = tf.clip_by_value(action_tf_var, -cap, cap)
        action_tf_var = tf.clip_by_value(action_tf_var, -pi, pi)
    return action_tf_var, norm_dist, mu, sigma, action_disc_var, action_probs


def value_function(state):
    n_hidden1 = 400  
    n_hidden2 = 400
    n_outputs = 1
    
    with tf.variable_scope("value_network"):
        init_xavier = tf.contrib.layers.xavier_initializer()
        #reg = tf.contrib.layers.l2_regularizer(scale=0.001)
        #reg = tf.contrib.layers.l2_regularizer(scale=0.9999)   #to get linear regression, regularization should be very high
        '''
        hidden1 = tf.layers.dense(state, n_hidden1, tf.nn.relu, init_xavier, kernel_regularizer=reg)
        hidden1 = tf.layers.batch_normalization(hidden1)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.relu, init_xavier, kernel_regularizer=reg)
        hidden2 = tf.layers.batch_normalization(hidden2)        
        '''
        hidden1 = tf.layers.conv2d(state, filters=8, kernel_size=7, activation=tf.nn.elu, kernel_initializer=init_xavier)
        hidden1n = tf.layers.batch_normalization(hidden1)
        hidden2 = tf.layers.conv2d(hidden1n, filters=8, kernel_size=5, activation=tf.nn.elu, kernel_initializer=init_xavier)
        hidden2n = tf.layers.batch_normalization(hidden2)
        hidden3 = tf.layers.conv2d(hidden2n, filters=8, kernel_size=3, activation=tf.nn.elu, kernel_initializer=init_xavier)
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

lr_actor = 0.000663   #0.000663 #0.005   #0.000008
lr_critic = 0.000663  # 0.005

#lr_actor = 0.000008
#lr_critic = 0.008

# define required placeholders
action_placeholder = tf.placeholder(tf.float32, [batch_size, action_dims])
delta_placeholder = tf.placeholder(tf.float32, [batch_size])
target_placeholder = tf.placeholder(tf.float32, [batch_size])
picked_action_placeholder = tf.placeholder(tf.int32, [batch_size * action_dims, 1])

action_tf_var, norm_dist, mu, sigma, action_disc_var, action_probs = policy_network(state_placeholder )
V = value_function(state_placeholder)

picked_act = tf.gather(action_probs, picked_action_placeholder, validate_indices=True,batch_dims=-1)


a_continuos = norm_dist.prob(action_placeholder)
a_discrete = tf.reshape(picked_act, [batch_size, action_dims])

a_all = tf.concat([a_discrete, a_continuos],axis=1)

# define actor (policy) loss function 
loss_actor = -tf.log(tf.reduce_prod(a_all, axis=1) + 1e-5 ) * delta_placeholder

#loss_actor = tf.reduce_mean( log_prob )
#loss_actor = tf.losses.mean_squared_error(action_tf_var, delta_placeholder)
training_op_actor = tf.train.AdamOptimizer(lr_actor, name='actor_optimizer').minimize(loss_actor)
# define critic (state-value) loss function
loss_critic = tf.reduce_mean(tf.squared_difference(tf.squeeze(V), target_placeholder))
training_op_critic = tf.train.AdamOptimizer(lr_critic, name='critic_optimizer').minimize(loss_critic)
################################################################
#Training loop
gamma = 0.0    #0.99        #discount factor: 0.0 for inmediate reward only
num_episodes = 1000000 #300
display_rate = 10
reward_total = 0.0
watch_total = 0.0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    episode_history = []
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
            #rs = np.random.uniform(-cap, cap, [batch_size, action_dims])

            rsA = np.random.uniform(-cap/2.0, cap/2.0, [batch_size, 1])
            rsB = np.random.uniform(-cap/2.0, cap/2.0, [batch_size, 1])
            rsC = np.random.uniform(-cap/2.0, cap/2.0, [batch_size, 1])
            rs = np.concatenate([rsA-rsB,rsA-rsC],axis=1)
            
            #bb1 = (cap/2.0)*(np.random.randint(2,size=(batch_size,1)) - 1)
            #bb2 = (cap/2.0)*(np.random.randint(2,size=(batch_size,1)) - 1)
            #bb3 = (cap/2.0)*(np.random.randint(2,size=(batch_size,1)) - 1)
            #rs = np.concatenate([bb1-bb2,bb1-bb3],axis=1)
            
            #rs = np.abs(rs)
            
            state_batch = []
            for i_s in rs:
                state_i, _ = env.reset(i_s, imageI=True, PSF=False)
                state_batch.append(state_i)

            state = np.stack(state_batch, axis=0)
            

            #print(np.sum(state))
            #Sample action according to current policy
            action, m, s, action_direc  = sess.run( [action_tf_var, mu, sigma, action_disc_var], feed_dict={state_placeholder: scale_state(state)})
            #act_probs  = sess.run( [action_tf_var, mu, sigma, action_probs], feed_dict={state_placeholder: scale_state(state)})
            #Ignore current policy and sample action from behaviour policy
            #behav_dist = uniform(-cap, 2.0 * cap)
            #pstn_act = np.random.uniform(-cap, cap, [(2*nrings)+1, (2*nrings)+1])
            #pstn_act[axial_to_matrix(0, 0, nrings)] = 0.0
            #rs_act = pt.stack_piston_steps(pstn_act, nrings)

            #behav_dist = triang(0.5, loc=-2.0*cap, scale=4.0*cap)
            #action = behav_dist.rvs([batch_size, action_dims])


            direc = action_direc + (action_direc - 1)
            #Execute action and observe reward & next state from E
            next_rs = rs - (action * np.reshape(direc,[batch_size, action_dims]))
            watch = np.mean(np.square(next_rs), axis=1)   #reward shape is (batch_size,)

            
            next_state_batch = []
            reward_batch = []
            for i_ns in next_rs:
                _, reward_i = env.reset(i_ns, imageI=False, PSF=True)
                reward_batch.append(reward_i)
                #reward_batch.append( -np.mean(np.square(next_rs)) )
            


            #next_state = np.stack(next_state_batch, axis=0)
            reward = np.stack(reward_batch, axis=0)
            
            #V_of_next_state = sess.run(V, feed_dict = {state_placeholder: scale_state(next_state)})  
            #Set TD Target
            #target = r + gamma * V(next_state)     
            target = reward #+ gamma * np.squeeze(V_of_next_state) 
            
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
                    picked_action_placeholder: action_direc,
                    delta_placeholder: td_error })  # td_error})   # reward })
            
            #Update critic by minimizinf loss  (Critic training)
            _, loss_critic_val  = sess.run(
                [training_op_critic, loss_critic], 
                feed_dict={state_placeholder: scale_state(state), 
                target_placeholder: target})
            
            
            if episode % display_rate == 0:
                #print(str(episode) + '#' + str(np.mean(reward_total)/display_rate) + '#' + str(np.mean(watch_total)/display_rate))
                print("*"+str(episode)+"*"+str(np.mean(reward_total)/display_rate).replace(".", ",")+"*"+str(np.mean(watch_total)/display_rate).replace(".", ",")+"*")
                reward_total = 0.0
                watch_total = 0.0
          

