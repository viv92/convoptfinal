# #
#  * CS519 - Convex Optimization
#  * Implementation Code for Final Report
#  *
#  * Author: Vivswan Shitole
#  * Email: shitolev@oregonstate.edu
#  #

import simpy
import numpy as np
import math
import tensorflow as tf
import scipy
from collections import defaultdict
import sys
import os
import itertools
import matplotlib
matplotlib.use('Agg')
from lib import plotting
matplotlib.style.use('ggplot')

#DEFINING CUSTOM ENVIRONMENT - EARTHMOVING OPERATION

#define all time delays for each operation in environment

def EnterAreaA_time(TruckSpdRatio):
    return (TruckSpdRatio * np.random.uniform(0.3, 0.34))
def EnterAreaB_time(TruckSpdRatio):
    return (TruckSpdRatio * np.random.uniform(0.22, 0.26))
def DumpBucketA_time():
    return np.random.uniform(0.32, 0.34)
def DumpBucketB_time():
    return np.random.uniform(0.16, 0.17)
def ExcavateA_time():
    return np.random.uniform(0.33, 0.37)
def ExcavateB_time():
    return np.random.uniform(0.33, 0.37)
def Haul_time(TruckSpdRatio):
    return (TruckSpdRatio * np.random.triangular(4.4, 5.7, 6.6))
def EnterDump_time(TruckSpdRatio):
    return (TruckSpdRatio * np.random.uniform(0.6, 0.8))
def Dump_time():
    return np.random.triangular(2, 2.1, 2.2)
def Return0_time(TruckSpdRatio):
    return (TruckSpdRatio * np.random.triangular(2.5, 2.9, 3.4))
def Return1A_time(TruckSpdRatio):
    return (TruckSpdRatio * np.random.triangular(1, 1.5, 2.0))
def Return1B_time(TruckSpdRatio):
    return (TruckSpdRatio * np.random.triangular(1.8, 2.3, 2.8))

#define all processes in the environment

#define all processes

def DumpBucketA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity):
    global state
    while TrkUndrExcA.level == 0:
        yield env.timeout(1)
    yield ExcWtDmpA.get(1)
    yield env.timeout(DumpBucketA_time())
    yield SlInTrkA.put(BucketA_capacity)
    if state[8] > 0:
        state[8] -= (BucketA_capacity/6)
    else:
        state[9] -= (BucketA_capacity/3)
    env.process(ExcavateA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity))

def DumpBucketB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity):
    global state
    while TrkUndrExcB.level == 0:
        yield env.timeout(1)
    yield ExcWtDmpB.get(1)
    yield env.timeout(DumpBucketB_time())
    yield SlInTrkB.put(BucketB_capacity)
    if state[10] > 0:
        state[10] -= (BucketB_capacity/6)
    else:
        state[11] -= (BucketB_capacity/3)
    env.process(ExcavateB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity))

def ExcavateA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity):
    yield env.timeout(ExcavateA_time())
    yield ExcWtDmpA.put(1)
    env.process(DumpBucketA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity))

def ExcavateB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity):
    yield env.timeout(ExcavateB_time())
    yield ExcWtDmpB.put(1)
    env.process(DumpBucketB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity))

def EnterAreaA(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    global state
    yield TrkUndrExcA.put(1)
    yield ManeuvSpcA.get(1)
    yield TrkWtLdA.get(1)
    if TruckCap == 6:
        state[0] -= 1
        state[4] = 1
    else:
        state[1] -= 1
        state[5] = 1
    yield env.timeout(EnterAreaA_time(TruckSpdRatio))
    if TruckCap == 6:
        state[4] = 0
        state[8] = 1
    else:
        state[5] = 0
        state[9] = 1
    yield ManeuvSpcA.put(1)
    env.process(HaulA(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))

def EnterAreaB(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    global state
    yield TrkUndrExcB.put(1)
    yield ManeuvSpcB.get(1)
    yield TrkWtLdB.get(1)
    if TruckCap == 6:
        state[2] -= 1
        state[6] = 1
    else:
        state[3] -= 1
        state[7] = 1
    yield env.timeout(EnterAreaB_time(TruckSpdRatio))
    if TruckCap == 6:
        state[6] = 0
        state[10] = 1
    else:
        state[7] = 0
        state[11] = 1
    yield ManeuvSpcB.put(1)
    env.process(HaulB(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))

def HaulA(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    global state
    global num_of_load

    yield SlInTrkA.get(TruckCap)
    yield TrkUndrExcA.get(1)
    num_of_load += 1
    if TruckCap == 6:
        state[8] = 0
    else:
        state[9] = 0
    yield env.timeout(Haul_time(TruckSpdRatio))
    yield WtEnterDump.put(1)
    env.process(EnterDump(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))

def HaulB(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    global state
    global num_of_load

    yield SlInTrkB.get(TruckCap)
    yield TrkUndrExcB.get(1)
    num_of_load += 1
    if TruckCap == 6:
        state[10] = 0
    else:
        state[11] = 0
    yield env.timeout(Haul_time(TruckSpdRatio))
    yield WtEnterDump.put(1)
    env.process(EnterDump(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))


def EnterDump(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    yield DumpSpots.get(1)
    yield WtEnterDump.get(1)
    yield env.timeout(EnterDump_time(TruckSpdRatio))
    env.process(Dump(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))


def Dump(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    global num_of_dump
    yield env.timeout(Dump_time())
    yield DmpdSoil.put(TruckCap)
    yield DumpSpots.put(1)
    env.process(Return0(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))
    num_of_dump += 1


def Return0(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    global num_of_return
    yield env.timeout(Return0_time(TruckSpdRatio))
    action = agent(name, env.now)
    if action == 0:
        env.process(Return1A(env, name, TrkWtLdA, TrkWtLdB,
        ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
        ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))
    else:
        env.process(Return1B(env, name, TrkWtLdA, TrkWtLdB,
        ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
        ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))
    num_of_return += 1


def Return1A(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    global state
    yield env.timeout(Return1A_time(TruckSpdRatio))
    yield TrkWtLdA.put(1)
    if TruckCap == 6:
        state[0] += 1
    else:
        state[1] += 1
    env.process(EnterAreaA(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))

def Return1B(env, name, TrkWtLdA, TrkWtLdB,
ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio):
    global state
    yield env.timeout(Return1B_time(TruckSpdRatio))
    yield TrkWtLdB.put(1)
    if TruckCap == 6:
        state[2] += 1
    else:
        state[3] += 1
    env.process(EnterAreaB(env, name, TrkWtLdA, TrkWtLdB,
    ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
    ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))

#define monitoring process
def monitor(env, DmpdSoil, TrkWtLdA, TrkWtLdB, SoilAmt, nTrucks, TruckCap):
    global num_of_load
    global num_of_dump
    global num_of_return

    #global cost params to be updated
    global TrckCst
    global ExcCst
    global OHCst

    global HourlyCst
    global Hrs
    global ProdRate
    global UnitCst
    #num of monitoring observations
    obs_num = 0
    #run monitoring loop
    while True:
        obs_num += 1

        #calculate outputs
        L_hrs = env.now/60.0
        L_hourlyCst = OHCst+ExcCst+(TrckCst*nTrucks) #duh constant
        if L_hrs > 0:
            L_prodRate = DmpdSoil.level/L_hrs
            if L_prodRate > 0:
                L_unitCst = L_hourlyCst/L_prodRate

        #terminate condition
        if (DmpdSoil.level > (SoilAmt-TruckCap)):

            #update global stats
            Hrs.append(L_hrs)
            HourlyCst.append(L_hourlyCst)
            ProdRate.append(L_prodRate)
            UnitCst.append(L_unitCst)

            print """
            nTrucks = %d\n
            num_of_load = %d \n
            num_of_dump = %d \n
            num_of_return = %d \n
            Hrs = %.4f \n
            HourlyCst = %.4f \n
            ProdRate = %.4f \n
            UnitCst = %.4f
            """ % (nTrucks, num_of_load, num_of_dump, num_of_return,
            L_hrs, L_hourlyCst, L_prodRate, L_unitCst)
            return

        yield env.timeout(1)


#seed
np.random.seed(0)

#global variables
num_of_load = 0
num_of_dump = 0
num_of_return = 0
nTrucks = 10

''' global state vector
state[0]: num of Trk1 WtLdA
state[1]: num of Trk2 WtLdA
state[2]: num of Trk1 WtLdB
state[3]: num of Trk2 WtLdB
state[4]: is Trk1 in ManeuvSpcA (0/1)
state[5]: is Trk2 in ManeuvSpcA (0/1)
state[6]: is Trk1 in ManeuvSpcB (0/1)
state[7]: is Trk2 in ManeuvSpcB (0/1)
state[8]: % empty Trk1 UndrExcA
state[9]: % empty Trk2 UndrExcA
state[10]: % empty Trk1 UndrExcB
state[11]: % empty Trk2 UndrExcB '''
state = np.zeros(12)
old_state = np.zeros((nTrucks,12))
old_time = np.zeros(nTrucks)
old_action = np.zeros(nTrucks).astype(int)
nA = 2 #number of actions
old_action_probs = np.zeros(nA)
discount_factor = 0.99
alpha_vf = 1e-3 #learning_rate
alpha_policy = 1e-4 #learning_rate
kl_target = 0.003 #max KL divergence allowed

#get interactive session
sess = tf.InteractiveSession()
#placeholder for observation
obs = tf.placeholder(tf.float32, shape=[12])

#value function (vf) neural network
fc_shared = tf.contrib.layers.fully_connected(inputs=tf.expand_dims(obs,0), num_outputs=24, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
vf_output = tf.contrib.layers.fully_connected(inputs=fc_shared, num_outputs=1, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())

#state value
state_value = tf.squeeze(vf_output)

#vf target
vf_target = tf.placeholder(tf.float32, shape=[])

#vf loss
vf_loss = tf.squared_difference(vf_target, state_value)

#optimizer
vf_optimizer = tf.train.AdamOptimizer(learning_rate=alpha_vf)

#training op
vf_train_op = vf_optimizer.minimize(vf_loss)

#policy neural network
policy_hidden = tf.contrib.layers.fully_connected(inputs=fc_shared, num_outputs=12, scope='policy_hidden', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
policy_output = tf.contrib.layers.fully_connected(inputs=policy_hidden, num_outputs=nA, scope='policy_output', activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())

#action probabilities
action_probs = tf.squeeze(tf.nn.softmax(policy_output))

#old policy distribution
action_probs_old = tf.placeholder(tf.float32, shape=[2])

#placeholder for chosen action
chosen_action = tf.placeholder(tf.int32, shape=[])

#probability of chosen action
prob_chosen_action = tf.gather(action_probs, chosen_action)

#probability of chosen action in old policy
oldprob_chosen_action = tf.gather(action_probs_old, chosen_action)

#placeholder for advantage
advantage = tf.placeholder(tf.float32, shape=[])

#surrogate loss function - TODO stop gradient on advantage ?
surrogate = tf.reduce_mean(advantage * (prob_chosen_action / oldprob_chosen_action))

#trainable weights of neural network
theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy_hidden')
theta_w, theta_b = theta

#first derivative of surrogate (g)
grad_surrogate_w, grad_surrogate_b = tf.gradients(ys=surrogate, xs=theta)

#KL divergence between old policy and current policy
kldiv = tf.reduce_sum(action_probs_old * tf.log((action_probs_old + 1e-10) / (action_probs + 1e-10)))

#kl gradient - used for hessian vector product
kl_grad_w, kl_grad_b = tf.gradients(ys=kldiv, xs=theta)

#placeholder for vector passed in hessian vector product
v = tf.placeholder(tf.float32, shape=[288])

#kl gradient vector product
kl_grad_vector_prod = tf.multiply(tf.reshape(kl_grad_w, [-1]), v)

#kl hessian vector product - used as functional for finding x using conjugate gradient
kl_hessian_vector_prod_w, kl_hessian_vector_prod_b = tf.gradients(kl_grad_vector_prod, theta)

#global vars init
initi = tf.global_variables_initializer()

#run session (initialise tf global vars)
sess.run(initi)


#conjugate gradient
def conjugate_gradient(ob, action, advntg, old_action_probs, cg_iters=10, tolerance=1e-10):

    #evaluate gradient of surrogate
    g = tf.convert_to_tensor(grad_surrogate_w)
    b = g.eval(feed_dict={obs: ob, chosen_action: action, advantage: advntg, action_probs_old: old_action_probs})
    b = b.ravel()

    #just for initialization
    x = 0
    r = b.copy() #remaining error in b space
    d = b.copy() #conjugate direction
    rdotr = r.dot(r)

    #begin iterations
    for i in range(cg_iters):

        #evaluate KL hessian vector product
        tmp = tf.convert_to_tensor(kl_hessian_vector_prod_w)
        z = tmp.eval(feed_dict={obs: ob, action_probs_old: old_action_probs, chosen_action: action, advantage: advntg, v: d})
        z = z.ravel()

        if d.dot(z) == 0: #corner case
            return (x,b)

        alpha = rdotr/(d.dot(z)) #step size
        x += alpha * d #new point
        new_r = r - alpha * z #new error
        new_rdotr = new_r.dot(new_r)

        if rdotr == 0: #corner case
            return (x,b)

        beta = new_rdotr/rdotr #correction for new conjugate direction
        d = new_r + beta * d #new direction
        rdotr = new_rdotr #for next iteration
        if rdotr < tolerance: #stopping condition
            break
    return (x,b)


#line search for step size
def line_search(x, g, delta, ob, action, advntg, old_action_probs, max_attempts=10):

    if isinstance(x, (int, long)) or isinstance(g, (int, long)) or x.dot(g) <= 0: #corner case
        return False

    beta = math.sqrt(2*delta/(x.dot(g))) #maximum step size advised
    #evaluate surrogate
    surrogate_value = surrogate.eval(feed_dict={obs: ob, action_probs_old: old_action_probs, chosen_action: action, advantage: advntg})
    #get current weights of neural network
    thet_w = tf.convert_to_tensor(theta_w)
    #begin line search
    for shrink_factor in (0.8 ** np.arange(max_attempts)):
        #new step size proposed
        beta *= shrink_factor
        #new weight params for new step size proposed
        new_theta = tf.reshape(thet_w, [-1]) + beta * x
        sess.run(tf.assign(theta_w, tf.reshape(new_theta, theta_w.shape)))
        #new surrogate function and kl div value
        surrogate_value_new, kl_value_new = sess.run([surrogate, kldiv], feed_dict={obs: ob, action_probs_old: old_action_probs, chosen_action: action, advantage: advntg})
        if kl_value_new > delta:
            surrogate_value_new = np.inf #infinite penalty on disrespecting the constrain
        improvement = surrogate_value_new - surrogate_value
        if improvement > 0:
            return True #proposed step is a valid step
    return False #proposed step is invalid step

#RL Agent - Routing Fork
def agent(truckName, time):
    global state #current state treated as next state
    global old_state #state for which we are learning
    global old_time
    global old_action
    global old_action_probs
    global Mean_TD_Error
    global Iterations
    global kl_target
    global max_policy_epochs
    truckIndex = int(truckName[len('truck')])
    reward = -1 * (time - old_time[truckIndex]) #time of cycle for this truck

    if old_time[truckIndex] > 0:  #not the first ever decision - learn for the old_state
        cur_state_value = state_value.eval(feed_dict={obs: state})
        old_state_value = state_value.eval(feed_dict={obs: old_state[truckIndex]})
        td_target = reward + discount_factor * cur_state_value
        td_error = td_target - old_state_value
        Iterations += 1
        Mean_TD_Error = ((Iterations-1)*Mean_TD_Error + td_error)/Iterations #moving estimate
        #update
        sess.run(vf_train_op, feed_dict={obs: old_state[truckIndex], vf_target: td_target})
        policy_epochs = 0

        #train (update) policy
        x, g = conjugate_gradient(old_state[truckIndex], old_action[truckIndex], td_error, old_action_probs)
        success_flag = line_search(x, g, kl_target, old_state[truckIndex], old_action[truckIndex], td_error, old_action_probs)

        #get action for current state
        cur_action_probs = action_probs.eval(feed_dict={obs: state})
        action = np.random.choice(np.arange(nA), p=cur_action_probs)
    else: #first ever decision - no learning since no old_action
        cur_action_probs = action_probs.eval(feed_dict={obs: state})
        action = np.random.choice(np.arange(nA), p=cur_action_probs)
        Iterations = 1
    #set up for next decision call
    np.copyto(old_state[truckIndex], state)
    old_action[truckIndex] = action
    np.copyto(old_action_probs, cur_action_probs)
    old_time[truckIndex] = time

    return action


#input global params
SoilAmt = 100
TrckCst = 48
ExcCst = 65
OHCst = 75

#output global params (results / costs)
HourlyCst = []
Hrs = []
ProdRate = []
UnitCst = []
Mean_TD_Error = 0
Iterations = 0 #number of decision iterations in an episode

#function to run the simulation
def run_sim(nTrucks, BucketA_capacity, BucketB_capacity, Truck1_capacity, Truck2_capacity, Truck1_speedRatio, Truck2_speedRatio):
    global state
    #simulation environment
    env = simpy.Environment()
    #resources
    TrkWtLdA = simpy.Container(env, init=(nTrucks/2), capacity=nTrucks)
    TrkWtLdB = simpy.Container(env, init=nTrucks-(nTrucks/2), capacity=nTrucks)
    ManeuvSpcA = simpy.Container(env, init=1, capacity=1)
    ManeuvSpcB = simpy.Container(env, init=1, capacity=1)
    TrkUndrExcA = simpy.Container(env, init=0, capacity=1)
    TrkUndrExcB = simpy.Container(env, init=0, capacity=1)
    SlInTrkA = simpy.Container(env, init=0, capacity=(max(Truck1_capacity,Truck2_capacity) + BucketA_capacity))
    SlInTrkB = simpy.Container(env, init=0, capacity=(max(Truck1_capacity,Truck2_capacity) + BucketB_capacity))
    ExcWtDmpA = simpy.Container(env, init=1, capacity=1)
    ExcWtDmpB = simpy.Container(env, init=1, capacity=1)
    WtEnterDump = simpy.Container(env, init=0, capacity=nTrucks)
    DumpSpots = simpy.Container(env, init=3, capacity=3)
    DmpdSoil = simpy.Container(env, init=0, capacity=SoilAmt)
    #dump bucket processes
    env.process(DumpBucketA(env, TrkUndrExcA, ExcWtDmpA, SlInTrkA, BucketA_capacity))
    env.process(DumpBucketB(env, TrkUndrExcB, ExcWtDmpB, SlInTrkB, BucketB_capacity))

    #initial state
    state[0] = (nTrucks/4) + 1
    state[1] = nTrucks/4
    state[2] = nTrucks/4
    state[3] = (nTrucks/4) + 1

    #initial truck arrangement - half of type1 and half of type2
    for i in range(nTrucks):
        if i < (nTrucks/2):
            TruckCap = Truck1_capacity
            TruckSpdRatio = Truck1_speedRatio
        else:
            TruckCap = Truck2_capacity
            TruckSpdRatio = Truck2_speedRatio
        if i % 2 == 0:
            #truck moving process for each truck
            env.process(EnterAreaA(env, 'truck%d' % i, TrkWtLdA, TrkWtLdB,
            ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
            ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))
        else:
            #truck moving process for each truck
            env.process(EnterAreaB(env, 'truck%d' % i, TrkWtLdA, TrkWtLdB,
            ManeuvSpcA, ManeuvSpcB, TrkUndrExcA, TrkUndrExcB, SlInTrkA, SlInTrkB,
            ExcWtDmpA, ExcWtDmpB, WtEnterDump, DumpSpots, DmpdSoil, TruckCap, TruckSpdRatio))

    #monitoring process
    proc = env.process(monitor(env, DmpdSoil, TrkWtLdA, TrkWtLdB, SoilAmt, nTrucks, max(Truck1_capacity, Truck2_capacity)))
    #run all processes
    env.run(until=proc)

#main
def main():
    global num_of_load
    global num_of_dump
    global num_of_return
    global state
    global old_state
    global old_time
    global Mean_TD_Error
    global Iterations
    global nTrucks

    #defining environment params
    BucketA_capacity = 1.5
    BucketB_capacity = 6.0
    Truck1_capacity = 9
    Truck2_capacity = 3
    Truck1_speed = 35.0
    Truck2_speed = 20.0
    Truck1_speedRatio = Truck1_speed / (Truck1_speed + Truck2_speed)
    Truck2_speedRatio = Truck2_speed / (Truck1_speed + Truck2_speed)

    #number of episodes
    num_episodes = 200
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_loss=np.zeros(num_episodes))
    for i_episode in range(num_episodes):
        #reset global vars
        num_of_load = 0
        num_of_dump = 0
        num_of_return = 0
        state = np.zeros(12)
        old_state = np.zeros((nTrucks,12))
        old_time = np.zeros(nTrucks)
        Mean_TD_Error = 0
        Iterations = 0
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1 == 0:
            print "\rEpisode: ", i_episode + 1, " / ", num_episodes
        #run simulation
        run_sim(nTrucks, BucketA_capacity, BucketB_capacity, Truck1_capacity, Truck2_capacity, Truck1_speedRatio, Truck2_speedRatio)
        stats.episode_lengths[i_episode] = Hrs[i_episode]
        stats.episode_rewards[i_episode] = ProdRate[i_episode]
        stats.episode_loss[i_episode] = abs(Mean_TD_Error)
    plotting.plot_episode_stats(stats, name="Online_TRPO", smoothing_window=20)


if __name__ == '__main__':
    main()
