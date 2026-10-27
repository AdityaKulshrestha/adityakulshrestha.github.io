+++
title = 'Rl'
date = 2026-07-31T10:35:26Z
draft = true
+++


# RL for Autoregressive People

## Background
    This section is to setup a basic evolution of RL techniques primarily in LLM. Feel free to skip to next section.

Reinforcement Learning or RL in LLMs have recently become a important recipe in training frontier models. Majority of frontier labs are now focusing on letting the model explore and learn from rewards rather than teaching it through a guided learning like SFT or Distillation. 

This change in training methodology was intially started by OpenAI where they trained the OpenAI models with human like response using Reinforcement Learning using Human Feedback or RLHF. Later several other techniques started getting popular like Proximal Policy Optimization and Direct Preference Optimization. But all these techniques required human verified curated data. There were techniques to use LLM-as-Judge but there were high risks of imposing bias and challenges of ensuring fair evaluation. 

This is where DeepSeek introduced a twisted variation of PPO called Group Relative Policy Optimization. Later, this got into the maintstream training regime.
Let's do some intuitive disection of the name - there is group and relative, so something around relative scoring. Chaining it with policy and optimization. You can deduce we are somewhat leveraging relative scoring to optimize the model to get better outputs.  

## Fundamentals
Before proceeding ahead, we have to cover few important concepts from classical RL in order to grasp the detail better. 

Let's take an example of a model playing chess. Our task is to train a model which can play better chess. 

- State (s): The current position of chess board with all the pieces

![Chess](../../assets/grpo/chess.png)

- Policy ($\pi$): The target model which we are tyring to optimize to play chess. 
- Action ($a$): The next move suggested by the model (policy). 
- Reward ($r$): The reward granted by a verifiable mechanism (can be human or other methods). 
- Value ($V$): TODO
- Q ($Q$): TODO




### LLM Analogy
- Policy: Your LLM which are you're trying to optimize. 
- Action: The next token predicted based on the inputs state (input tokens).
- State: Input tokens passed to LLM. 
- Reward: Reward for the action based on a set of rules. 


## Proximal Polity Optimization: A predecessor to GRPO

Consider a scenario where you have trained a model which can follow instructions. But the model doesn't sound very human like. It will use work like `thereof` or `delves` in every sentence [Why ChatGPT delves so much](https://arxiv.org/abs/2412.11385). You don't want the model to lose out on its ability to generate good answer but want to slightly nudge it away towards using common conversation like words and use words like `delves` in professional drafts or texts. 

There is where Proximal Polity Optimization can help. It slightly nudges the model towards generating better answer and also restricts the model to loose out on its previous ability. For this let's think from start what do we need? 
1. Preference Data - Data ranked in order of preference. It usually follows the structure {prompt: text, sample1: score, sample2: score, ...}. 
2. A Reward model - This will be used to first train how to reward the generation by training it on preference data where it learns to predict only the score for a given input response. 
3. Policy - The LLM which we are trying to optimize. 

Now that we have these basic elements, let's see what are the challenges and how do we overcome them? 

### Absolute Scoring: Bradley Terry Model

We don't want the model to be trained on absolute scoring. Absolute scoring means that it will predict fixed scoring for a certain type of answer. If there are three passages written on Transformers, providing absolute marking is very subjective. Hence training a reward model on asbolute ranking might incur bias and noise. 
So to avoid that we approach a similar approach as how people are ranked in chess. The relative scoring using a model called as Bradley Terry Model. Bradlet Terry Model is also used in Elo's rating and we will use it for our purpose now.


$$
P(r_i > r_j) = \frac{e^{R_{\phi}(p, r_i)}}{e^{R_{\phi}(p, r_i)} + e^{R_{\phi}(p, r_j)}} = \sigma(R_{\phi}(p, r_i) - R_{\phi}(p, r_j))
$$

Here, 
$$
\begin{aligned}
P(r_i \succ r_j) &: \text{Probability of choosing the } r_i^{\text{th}} \text{ response over } r_j. \\
r_i &: \text{Selected response} \\
r_j &: \text{Rejected response} \\
R_{\phi}(p, r) &: \text{Reward by reward model for p input propmt and r response} \\
\sigma &: \text{Function used to convert the reward into probalility between 0 to 1.} 
\end{aligned} \\
$$

Now Bradley Terry model only gives us the probability for choosing one response over other. How do we convert this into loss to train our reward model. 
This where we use Binary Cross Entropy. 

$$
\begin{aligned}
\text{BCE} = -[y\log{(p)} + (1 - y)\log{(1 - p)}]
\end{aligned}
$$

One thing to note here, for our purpose, we take the chosen sample as always 1. After plugin, the right side cancels out and we get:

$$
\begin{aligned}
\text{BCE} = -\log(p)
\end{aligned}
$$

Here $\text{p}$ is the probability of the chosen answer. If you plug that you get:

$$
\begin{aligned}
p = \sigma(r_i - r_j) \\
L_{\theta} = -\log(\sigma(r_i - r_j))
\end{aligned}
$$


### Training the policy model: REINFORCE
We have our reward model now. The next obvious step is to sample(generate multiple responses), grade them using reward model and train the policy model to maximize on the rewards. But there is one caveat, we can't use the pretrainig objective of crossentropy since we are optimizing on the rewards here. 
But our policy model generates discrete tokens so we must find a way to bypass the discrete tokens. What do we have here? log probs of the sequence. We can use that to draft a loss function for our model. This is where REINFORCE tric comes. We don't differentiate through the sample, we use the log derivative trick:


$$

\nabla_{\theta}  \mathbb{E}[R] = \mathbb{E} \left[ \sum_t \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \cdot \right]
$$


The trick here is to rewrite the expectation in terms of logprobs and reward, you can learn more on how it is derived [here](https://snawarhussain.com/educational/reinforcement%20learning/Policy-Gradient-Methods-in-ReInforcement-Learning-Log-trick-for-gradient-computation/#derivation-of-the-log-derivative-trick) What this does essentially:
1. **Generate** - The policy model generates the respones to input prompt. 
2. **Score** - The reward model scores the generated response. 
3. **Get log probs** - We calculate the log probs of the model. (We don't store it during the generation to save computation.)
4. **Reward Multiplication** - Multiple the reward with the gradient of LLM log probabilites.
    - If *R* is high: **increase** the probabilites of the generated tokens. Push the model to generate more high reward answer. 
    - If *R* is low: **decrease** the probability of the generated tokens. Punish the model for generating bad answer. 


### Credit Assignment Problem : Advantages

Consider a response where LLM generated good quaity answer and received +10 from reward model. Now, who is actually responsible for the good answer? Specially which tokens should be awarded? 

This is essentially credit assignment problem, which tokens should receive the credits. 

My way to look at it, you want to increase the direction towards tokens which yield good answer but you only have rewards which are received at the end. To mitigate this issue you introduce advantage.

#### What are advantages?
In RL, Advantages are formally defined as:
$$
A_t^{\pi} = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t)
$$

$Q$ - It is basically tells that for a given state (input prompt) + generated prompt (till t), what is the reward given for an action (next token) being taken. 
$V$ - Think of this an uruguay (from kungfu panda) who knows future and can predict the future reward based on current state. 
$A$ - Is our advantage. We try to optimize for advantage.

Calculating Advantage is not linearly straightforward. Let's see why. 


### Calculating Q and V Function: Generalized Advantage Estimate

#### V Function
Calculating Q and V is too complex or as we call it as -  intractable. We only have reward function which is used to predict onal the final reward based on the entire sequence generated. So how can we estimate per token advantage? 
 
For V function w rely on another model called critic (value function). The critic model predicts the V function value (basically we are training uruguay). It tries to predict the final reward based on the current state. 


##### Q Function
To calculate Q function we leverage indirect approach (we don't train another model since it would be commputationally very heavy; thank god). Hence we rely on these technique. Let's discuss the pros and cons of each: 

There are three ways to calculate GAE:
1. **Monte Carlo** - In monte carlo we sum up the individual reward by taking a weighted sum. 

$$
Q_{MC}(s_t, a_t) = \sum_{k=0}^{\infty} \gamma^{k}r_{t+k}
$$
*$\gamma^{k}$ is the discount factor whose value ranges from 0 to 1 and $r_{t}$ is the reward given by reward model*

Monte Carlo is unbiased because it uses per token grond truth reward but suffers from high variance. 
This means for the same action in the same state, the final reward signal can differ vastly based on what the subsequent steps predict. 


2. **Temporal Difference** - Temporal Difference takes the opposite apporach. It doesnt' consider the entire sequence to predict each token level reward. It only focus on a single step in the future to predict the reward. Here it uses Critic model to look into the future and predict the reward.

This can be explained using Bellman's equation:
$$
Q_{TD}(s_t, a_t) = r_t + \gamma V(s_{t+1})
$$

Contrary to Monte Carlo approach, Temporal Difference provides a stabler variance but suffers from extreme bias. Since it relies on the critic model to predict the next state reward. If Critic predicts the rewards incorrectly, it impacts the advantage also. 


3. **Generalized Advantage Estimation** - GAE is the proposed balanced method which balances the shortcomings from both Monte Carlo and Temporal difference (i.e. high bias and high variance). 
Instead of choosing 1 step (like TD) or considering entire steps (like MC). GAE computes the advantage for 1 step, 2 steps, 3 steps all the way to the end and take exponentially weighted average of all of them.



Mathematically,

$$
A_{K}^{GAE} = \delta_0 + \lambda \delta_1 + 
\lambda^{2} \delta_2 + ... + (\lambda)^{K-1}\delta_{K-1} = \sum_{l=0}^{\infty} (\lambda)^t \\
\delta_t = V_{\lambda}(s_{t+1}) - V_{\lambda}(s_t)
$$

**Lambda Function**:
- $\lambda = 0$: The expression collapses to Temporal Difference (High Bias, Low Variance).
- $\lambda = 1$: The expression converts to monte carlo (low bias, high variance). 
- $\lambda = 0.95$: A balance between MC and TD what we call as GAE. 

NOTE - For simplicity I am not introducing discount factor $\gamma$. Feel free to look it up. 

### Catastrophic Destabilization: Clipping
Our target for model training is to make the loss proportional to the advantages. But it just so could happen that the advantages suddenly become to high or too low. In such cases we don't want the weight to shift drastically and hence we need to add a capping mechanism to control the change in weights. For this we define a probability ratio:

$$
c_t = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta}(a_t | s_t)}
$$

- $c_t = 1$ means the new policy assigns the same probability as the previou one to the current action (generated token) $a_t$ being take for given state (input prompt + generated respones till t) $s_t$.
- $c_t = 1.27$, it means the new probability is assigning 27% more likely to the current action.  

I don't know why, this entire loss term is called as **Surrogate Objective Function or Surrogate Loss**. 
$$
L^{clip}(\theta) = \mathbb{E_t}\left[\text{min}(c_t(\pi_{\theta})A_t, \text{clip}(c_t(\pi_{\theta}), 1 - \epsilon, 1 + \epsilon).A_t) \right]
$$

For most of the RLVR objectives, $\epsilon=0.2$. Let's consider the new policy predicted the probability as 0.8 and the previous policy predicted the probability as 0.4. The ratio becomes - $\frac{0.8}{0.4} = 2$. 
- If $A_t>0$ which means the action was good. We don't want to increase the weight suddenly, the clip the loss $\text{min}(2A_t, \text{clip}(2, 0.8, 1.2)A_t)  = 1.2 A_t$
- If $A_t<0$ which means the action is bad. Hence we clip it $\text{min}(2A_t, \text{clip}(2, 0.8, 1.2)A_t) = 2 A_t$. 


### Reward Hacking: KL Penalty
Goodhart's Law states: *When a measure becomes a target, it ceases to be a good measure*. 

Reward hacking basically means when the model finds other trajectory different from the preffered one to maximize the rewards. Kind of cheating in exam to score high but learning nothing meaningful.

<!-- In the paper  -->

KL divergence is added as additional loss component to overcome this issue. It simply tugs back the model from deviating too much from reference model. It calculates the KL divergence between the updated policy model with the original reference model. 

$$
\text{KL}(\theta) = \mathbb{E}\left[\pi(\theta_{orig}) || \pi(\theta_{ref})\right] 
$$

A weightage factor $\beta$ is used to control the weightage of KL loss component in the overall loss.


### Encouraging Exploration: Entropy Bonus
The RL training heavily focuses on exploration. Hence we penalize the model for generating expected answer through entropy regularization. The target is to balance exploration (trying new things) with exploitation (doing what is known to work). We simply leverage Shannon entropy with weightage:

$$
\text{H}(\pi_{\theta}) = - \sum_{a_t}\pi(a_t | s_t) \text{log}(\pi(a_t | s_t)) = - \mathbb{E}\left[\text{log} \pi_{\theta}(a_t | s_t) \right]
$$

Consider a input prompt: what is 2+2? and we are trying to ask the model to put final answer in \math{}. 
- If the model predicts three token 4 with 0.96 prob, \math with 0.01 and four with 0.02. The total entropy: -(0.96 x ln(0.96) + 0.01 x ln(0.01) + 0.02 x ln(0.02)) = 0.1635
- If the model tries to predict three tokens with \math with 0.4 probability, 4 with 0.3 and four with 0.3. The total entropy: -(0.4 x ln(0.4) + 0.3 x ln(0.3) + 0.3 x ln(0.3)) = 1.088


## Sum Up
The final PPO loss term inclues 4 components: 

$$
\mathcal{L}_{\text{PPO}}(\theta, \gamma) = \mathcal{L}_{\text{clip}}(\theta) + w_1 H(\theta) - w_2 \text{KL}(\theta) - w_3 \mathcal{L}(\gamma)
$$
|Term|Name|Purpose|
|----|----|-------|
| $L_{\text{clip}}(\theta)$ | Surrogate Loss | Maximize the actions for high-advantage (clipping done to avoid large updates) | 
| $H(\theta)$ | Entropy | Encourages for exploration (non obvious tokens) |
| $\text{KL}(\theta)$ | KL Divergence | Penalize deviations from reference policy |
| $\text{L}_(\lambda)$ | Critic Loss | Minimize error in value predictions (critic model training) |


Interesting thing to note on the number of models we load during PPO training and why its so heavy on the memory side. 

| Model | Role | Updated | 
| ----- | ---- | ------- |
| Policy (LLM) | Current model being optimized (generates responses) | Yes |
| Reference Policy | Frozen copy of policy model (no gradients; for kl-reference) | No |
| Reward Model | For rewarding the responses | No |
| Critic | For per-token values (trained along with policy model) | Yes |


## Group Relative Policy Optimization
Deepseek introduced Group Relative Policy optimization through its paper called [Deepseek Maths](https://arxiv.org/abs/2402.03300). This created a major shift in training paragdigm and is being followed till date by most of the frontier labs. Though there are various different flavours available of GRPO like (GSPO , DrGRPO, DSPO.. etc). For our focus we will only focus on vanilla GRPO. 

The PPO training has two major issues:
1. It requires labelled dataset to train the reward models.
2. The training requires large compute since we are loading 4 models at the same time. 

GRPO fixed these challenges with one shot. Instead of relying on value function approximation, it uses the average reward of multiple sampled outputs for a single given input. The advantage is calculated based on relative which served as an proxy for per token reward. 
For reward model, it relied on domains where the final answer can be evaluated through heuristic methods. It's believed that if the model trajectory can reach the answer correctly, then we assume the trajectory is also correct. 
Hence it saved us with two things:

1. No need of reward model: Use heuristics methods to score the final answer only.
2. No need of value model: It used multiple sample to get the average score and use that to relatively score the advantage. 

But, how can the relative score solves credit assignment problem? 
Straight answer is, it does suffer from credit assignment. GRPO assign identical credit to every token. If you are a math geek, you can find the mathematical proof [here](https://arxiv.org/pdf/2606.29238)


### Let's see the Maths
GRPO modifies the surrogate objective and changes the way how the advantage is calculated. Additionally it also removes the entropy penalty for exploration. Mathematically, the objective boils down to:

$$
L_{GRPO}(\theta) = \mathbb{E} \left[q \sim \text{P(Q)}, \ {o_i}_{i=1}^{G} \sim \pi_{\theta_{old}} (O|q) \right] \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{{o_i}} \left\{ \text{min} \left[\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_old}(o_{i,t}|q,o_{i,<t})}\hat{A}_{i,t}, \text{clip} \left(\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_old}(o_{i,t}|q,o_{i,<t})}, 1 - \epsilon, 1 + \epsilon\right) \hat{A}_{i,t} \right] - \beta \mathbb{D}_{KL} \left[\pi_{\theta} || \pi_{ref}\right]\right\}
$$

where:
$$
\begin{align*}
  q &= \text{a question sampled from the question dataset} \\
  P(Q) &= \text{the distribution of the question dataset} \\
  o_{i,t} &= \text{the } i \text{-th output response's } t \text{-th token in a group sampled from the old policy model} \\
  G &= \text{the total number of outputs sampled in a group for each question} \\
  \pi_{\theta} &= \text{the current policy model being optimized} \\
  \pi_{\theta_{old}} &= \text{the old policy model for reference} \\
  \epsilon &= \text{Hyperparameter to control the clipping ratio} \\
  \beta &= \text{Hyperparameter to control the KL Divergence} \\
  r_i &= \text{the raw reward score given to the } i\text{-th output} \\
  \tilde{r}_i &= \frac{r_i - \text{mean}(r)}{\text{std}(r)} \text{ (the group-relative normalized reward)} \\
  \hat{A}_{i,t} &= \tilde{r}_i \text{ (the advantage assigned to each token in the } i\text{-th output)}
\end{align*}
$$

Seems daunting right? Let's unroll each component to understand the maths better.

$$
\begin{align*}
    \sum_{t=1}^{{o_i}} \left\{ \text{min} \left[\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_old}(o_{i,t}|q,o_{i,<t})}\hat{A}_{i,t}, \text{clip} \left(\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_old}(o_{i,t}|q,o_{i,<t})}, 1 - \epsilon, 1 + \epsilon\right)\hat{A}_{i,t} \right] - \beta \mathbb{D}_{KL} \left[\pi_{\theta} || \pi_{ref}\right]\right\}
\end{align*}
$$

 The summation is just a clipped summation of ration * advatages aggregated over all tokens and over all generated samples. This is called surrogate loss. 

$$
\begin{align*}
 \mathbb{D}_{KL} \left[\pi_{\theta} || \pi_{ref}\right]
\end{align*}
$$
The KL Divergence loss component which acts as a leash to restrict the policy model from diverging too far.   


### Let's look at the algorithm (Simplified)

1. For a given input q, sample out G rollouts (responses) from the policy model. 
2. Reward the sampled responses using reward scoring method r. 
3. Calculate the relative scoring and assign advantage A.
4. Update the policy model by maximizing objective.

NOTE - This algorithm is slightly different from the original Deepseek paper where they are doing multiple updates for a single step rollouts. This algorithm follows huggingface implementation. 


### Implementation: Step by Step




### Ablation Studies




## Conclusion