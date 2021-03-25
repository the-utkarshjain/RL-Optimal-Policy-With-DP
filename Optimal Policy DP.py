#!/usr/bin/env python
# coding: utf-8

# # Optimal Policies with Dynamic Programming
# 
# - Policy Evaluation and Policy Improvement.
# - Value and Policy Iteration.
# - Bellman Equations.

# In[1]:
get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport pickle')

# In[2]:
num_spaces = 3
num_prices = 3
env = tools.ParkingWorld(num_spaces, num_prices)
V = np.zeros(num_spaces + 1)
pi = np.ones((num_spaces + 1, num_prices)) / num_prices

# In[3]:
V
# In[4]:
state = 0
V[state]

# In[5]:
state = 0
value = 10
V[state] = value
V

# In[6]:
for s, v in enumerate(V):
    print(f'State {s} has value {v}')

# In[7]:
pi

# In[8]:
state = 0
pi[state]

# In[9]:
state = 0
action = 1
pi[state, action]

# In[10]:
pi[state] = np.array([0.75, 0.21, 0.04])
pi

# In[11]:
for s, pi_s in enumerate(pi):
    print(f''.join(f'pi(A={a}|S={s}) = {p.round(2)}' + 4 * ' ' for a, p in enumerate(pi_s)))

# In[12]:
tools.plot(V, pi)

# In[13]:
env.S

# In[14]:
env.A

# In[15]:
state = 3
action = 1
transitions = env.transitions(state, action)
transitions

# In[16]:
for s_, (r, p) in enumerate(transitions):
    print(f'p(S\'={s_}, R={r} | S={state}, A={action}) = {p.round(2)}')


# ##Policy Evaluation 
# $$\large v(s) \leftarrow \sum_a \pi(a | s) \sum_{s', r} p(s', r | s, a)[r + \gamma v(s')]$$
# In[17]:
def evaluate_policy(env, V, pi, gamma, theta):
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_update(env, V, pi, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

# In[18]:
def bellman_update(env, V, pi, s, gamma):
    value = 0
    for a in env.A:
        expected_val = np.dot(env.transitions(s, a)[:, 0] + gamma*V, env.transitions(s, a)[:, 1])
        value += pi[s, a] * expected_val
    
    V[s] = value

# In[19]:
get_ipython().run_line_magic('reset_selective', '-f "^num_spaces$|^num_prices$|^env$|^V$|^pi$|^gamma$|^theta$"')
num_spaces = 10
num_prices = 4
env = tools.ParkingWorld(num_spaces, num_prices)
V = np.zeros(num_spaces + 1)
city_policy = np.zeros((num_spaces + 1, num_prices))
city_policy[:, 1] = 1
gamma = 0.9
theta = 0.1
V = evaluate_policy(env, V, city_policy, gamma, theta)

# In[20]:
tools.plot(V, city_policy)

# In[21]:
with open('section1', 'rb') as handle:
    V_correct = pickle.load(handle)
np.testing.assert_array_almost_equal(V, V_correct)

# ## Policy Iteration
# In[22]:
def improve_policy(env, V, pi, gamma):
    policy_stable = True
    for s in env.S:
        old = pi[s].copy()
        q_greedify_policy(env, V, pi, s, gamma)
        if not np.array_equal(pi[s], old):
            policy_stable = False
    return pi, policy_stable

def policy_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    policy_stable = False
    while not policy_stable:
        V = evaluate_policy(env, V, pi, gamma, theta)
        pi, policy_stable = improve_policy(env, V, pi, gamma)
    return V, pi
# In[23]:
def q_greedify_policy(env, V, pi, s, gamma):
    value = np.zeros((1, len(env.A)))
    for a in env.A:
        expected_val = np.dot(env.transitions(s, a)[:, 0] + gamma*V, env.transitions(s, a)[:, 1])
        value[0,a] = pi[s, a] * expected_val
    
    max_action = np.argmax(value)
    pi[s, :] = 0
    pi[s, max_action] = 1

# In[24]:
get_ipython().run_line_magic('reset_selective', '-f "^num_spaces$|^num_prices$|^env$|^V$|^pi$|^gamma$|^theta$"')
env = tools.ParkingWorld(num_spaces=10, num_prices=4)
gamma = 0.9
theta = 0.1
V, pi = policy_iteration(env, gamma, theta)

# In[25]:
tools.plot(V, pi)

# In[26]:
with open('section2', 'rb') as handle:
    V_correct, pi_correct = pickle.load(handle)
np.testing.assert_array_almost_equal(V, V_correct)
np.testing.assert_array_almost_equal(pi, pi_correct)

# ## Value Iteration
# $$\large v(s) \leftarrow \max_a \sum_{s', r} p(s', r | s, a)[r + \gamma v(s')]$$
# In[27]:
def value_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    for s in env.S:
        q_greedify_policy(env, V, pi, s, gamma)
    return V, pi

# In[28]:
def bellman_optimality_update(env, V, s, gamma):
    value = np.zeros((1, len(env.A)))
    for a in env.A:
        expected_val = np.dot(env.transitions(s, a)[:, 0] + gamma*V, env.transitions(s, a)[:, 1])
        value[0,a] = expected_val
    
    V[s] = np.max(value)

# In[29]:
get_ipython().run_line_magic('reset_selective', '-f "^num_spaces$|^num_prices$|^env$|^V$|^pi$|^gamma$|^theta$"')
env = tools.ParkingWorld(num_spaces=10, num_prices=4)
gamma = 0.9
theta = 0.1
V, pi = value_iteration(env, gamma, theta)

# In[30]:
tools.plot(V, pi)

# In[31]:
with open('section3', 'rb') as handle:
    V_correct, pi_correct = pickle.load(handle)
np.testing.assert_array_almost_equal(V, V_correct)
np.testing.assert_array_almost_equal(pi, pi_correct)

# In[32]:
def value_iteration2(env, gamma, theta):
    V = np.zeros(len(env.S))
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            q_greedify_policy(env, V, pi, s, gamma)
            bellman_update(env, V, pi, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V, pi

# In[33]:
get_ipython().run_line_magic('reset_selective', '-f "^num_spaces$|^num_prices$|^env$|^V$|^pi$|^gamma$|^theta$"')
env = tools.ParkingWorld(num_spaces=10, num_prices=4)
gamma = 0.9
theta = 0.1
V, pi = value_iteration2(env, gamma, theta)
tools.plot(V, pi)