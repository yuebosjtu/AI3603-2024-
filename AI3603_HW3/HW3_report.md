## Report of Homework 3

### 1. Result Visualization of Cliff-walk Game
#### 1.1 episode reward and ϵ-value during the training process

**SARSA**
<img src="images\sarsa.png" width = 70%>

**Q-Learning**
<img src="images\q-learning.png" width = 70%>

**Dynamic Q-Learning**
<img src="images\dyna-q.png" width = 70%>

#### 1.2 Visualize the final paths found by the intelligent agents after training
The final paths found by the agents after training can be found in document "videos". 

**SARSA**
<img src="images\sarsa_path.png" width = 70%>
```
The video of testing results with agent trained by SARSA can be found in videos/sarsa.
```

**Q-Learning**
<img src="images\q-learning_path.png" width = 70%>
```
The video of testing results with agent trained by SARSA can be found in videos/q-learning.
```

**Dynamic Q-Learning**
<img src="images\dyna-q_path.png" width = 70%>
```
The video of testing results with agent trained by SARSA can be found in videos/dyna-q.
```

### 2. Result Analysis of Cliff-walk Game
#### 2.1 the difference between the paths found by Sarsa and Q-learning and the reason
##### 2.1.1 Different paths
The agent trained by Sarsa **goes up three steps**, then turn right, and after reaching the edge, goes down three steps to reach the terminal. The agent trained by Q-learning goes up one step, then turn right(**along the edge of cliff**), and after reaching the edge, goes down one step to reach the terminal.

##### 2.1.2 The underlying logic of two algorithms
SARSA is a method based on an "on-policy" strategy that considers actions and their outcomes in the current state to update the value function. At each step, SARSA uses the current strategy to determine what should be done next. Therefore, in the problem of cliff walking, if SARSA's initial strategy is to fall into the cliff, it may tend to choose a path away from the cliff, as this reduces the risk of immediate fall. Since SARSA updates the values on a complete trajectory under the current policy, it is more likely to converge to a safe and stable path, such as going to the second line.

Q-Learning is an "off-policy" method because it does not rely on any specific strategy to update the value function. On the contrary, Q-Learning always chooses actions with the highest expected return, even if it means taking risks close to a cliff. This is because Q-Learning considers the maximum expected return on all possible follow-up actions, rather than following a specific strategy. In this case, Q-Learning may find that the path closer to the cliff provides higher immediate rewards (such as reaching the target faster), although this involves greater risk. Therefore, the strategy learned by Q-Learning may be to walk along the edge of a cliff, as this path may provide the maximum cumulative reward in the long run.

##### 2.1.3 Summarize the reason
In summary, SARSA places greater emphasis on the safety and stability of current strategies, while Q-Learning seeks to maximize potential long-term benefits, even if it involves assuming certain risks. That's why after training process, SARSA learns the path to be far away from the cliff, while Q-Learning chooses the path to walk along the cliff.

#### 2.2 Analysis of the training efficiency between model-based RL and model-free alorithms
##### 2.1.1 Training Efficiency
The following images show the variation of rewards with the number of training rounds(within 150) for Q-planning steps of 0, 10, and 20, respectively.

<img src="images\step0.png" width = 30%>
<img src="images\step10.png" width = 30%>
<img src="images\step20.png" width = 30%>

As the number of Q-planning steps increases(The Q-planning steps of Q-Learning is 0), the convergence speed of the Dyna-Q algorithm increases. That is to say, the training efficiency getss faster as the number of Q-planning increases.

##### 2.1.2 Reason Analysis
<img src="images\dyna-q_baseline.png" width = 80%>

From the baseline of Dyna-Q algorithm shown above, we can see that it has N steps of interaction with the enviroment based on Q-Learning algorithm, which contributes to higher accuracy of Q-table. Dyna-Q uses  Q-planning to generate simulated data based on the model, and then improves the strategy using both simulated and real data.

### 3. Train and Tune the DQN Agent
#### 3.1 The performance of agent trained with the given parameters


##### 3.1.1 Training process
**learning curve of episode return**
<img src="images\episodic_return_old_params.png" width = 70%>

**learning curve of losses**
<img src="images\losses_old_params.png" width = 70%>

##### 3.1.2 Testing results
**testing results of 50 rounds**
<img src="images\dqn_res_old_params.png" width = 50%>

```
The video of testing results with old parameters can be found in videos/dqn_test_old_params.
```

#### 3.2 The performance of agent trained with the our parameters
##### 3.2.1 Our parameters of DQN model

| Parameter | given Value |  Our Value | how to design |
| ----------- | ----------- | --------- | ------------- |
| total-timesteps | 500000 |500000
| learning-rate |2.5e-4| 1e-3 |Add learning rate to accelerate learning process
|buffer-size|10000|50000| increase the buffer size to learn from more experience
|gamma|0.6|0.99| increase discount factor to decrease time influence
|target-network-frequency|500|250| Reduce the frequency of target network updates to accelerate learning progress
|batch-size|128|256| increase batch size to make learning more comprehensive
|start-e|0.3|0.2|
|end-e|0.05|0.02|
|exploration-fraction|0.1|0.3|Increase the duration of the exploration phase
|learning-starts|10000|10000|
|train-frequency|10|8|

##### 3.2.2 Training process
**learning curve of episode return**
<img src="images\episodic_return_new_params.png" width = 70%>

**learning curve of losses**
<img src="images\losses_new_params.png" width = 70%>

##### 3.2.3 Testing results
**testing results of 50 rounds**
<img src="images\dqn_res_new_params.png" width = 50%>

```
The video of testing results with our parameters can be found in videos/dqn_test_new_params.
```

##### 3.2.4 Comparison of Training Processes for Models Corresponding to New and Old Parameters
<img src="images\comparison_graph.png" width = 100%>

### 4. Improve Exploration Schema: UCB(Upper Confidence Bound) Algorithm
#### 4.1 Introduction of UCB Algorithm

UCB(Upper Confidence Bound) Algorithm is a classic uncertainty based strategy algorithm.

It is based on a famous inequality:
**Hoeffding's inequality**
Suppose $X_1, X_2, \dots, X_n \in [0,1]$ are $n$ independent and identically distributed random variables.
The experience expectation is $\bar{x}_n=\frac{1}{n}\sum_{k=1}^{n}X_k$, 
then we have $P\{E[X] \geq \bar{x}_n+u\} \leq e^{-2nu^2}$.

Let us apply Hoeffding's inequality to exploration problem. 
Let $\bar{x}_n$ be $\hat{Q}_t(a)$(The expectation reward of action $a$ found by us).
Let $u=\hat{U}_t(a)$(The metric of uncertainty of action $a$).
By Hoeffding's inequality, given a probability $p=e^{-2N_t(a)U_t(a)^2}$($N_t(a)$ is the number of times action $a$ is taken), $Q_t(a) < \hat{Q}_t(a)+\hat{U}_t(a)$ holds at least with probability $1-p$.($Q_t(a)$ is the actual expectation reward of action $a$)

When $p$ is sufficiently small, $Q_t(a) < \hat{Q}_t(a)+\hat{U}_t(a)$ holds a high probability of being true.
$\hat{Q}_t(a)+\hat{U}_t(a)$ can be seen as the upper bound of reward expectation.
UCB algorithm take the action with the largest upper bound of reward expectation, that is $a = argmax_{a \in A}[\hat{Q}_t(a)+\hat{U}_t(a)]$.($\hat{U}_t(a) = (\frac{-logp}{2N_t(a)})^{\frac{1}{2}}$)

Therefore, after setting a probability $p$, we can calculate the metric of uncertainty $\hat{U}_t(a)$.
Intuitively, the UCB algorithm first estimates the upper bound of the expected reward for each action before selecting it, so that there is only a small probability that the expected reward for each action will exceed this upper bound. Then, it selects the action with the highest upper bound of the expected reward, and thus chooses the action that is most likely to obtain the maximum expected reward.

#### 4.2 Evaluation of UCB Algorithm
##### 4.2.1 Merits
1. The UCB algorithm effectively balances the relationship between exploration and utilization by calculating the upper bound of the confidence interval for each option. This enables the algorithm to explore the unknown without ignoring the optimal options already known.
2. The algorithm can automatically adapt to changes in the environment without the need for manual intervention to adjust parameters.
3. The UCB algorithm is relatively simple, easy to understand and implement, and does not require complex computing resources.

##### 4.2.2 Drawbacks
1. Although the UCB algorithm can quickly discover the optimal action, in order to pursue the confidence level of each action, the convergence speed to the optimal action is slower than some methods, such as softmax.
2. The UCB algorithm is more difficult to extend to more general reinforcement learning environments than the ϵ-greedy algorithm, and it is difficult to solve non-stationary problems (where the benefits of actions do not come from a stationary probability distribution) and problems with massive state spaces.