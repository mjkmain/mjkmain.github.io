---
title: RL basic and MDP
author: mjkmain
date: 2024-01-02 03:54:00 +0900
categories: [RL]
tags: [RL]
pin: true
math: true
render_with_liquid: false
comments: True
---

# RL

## 1. Reinforcement Learning basic concepts

### 1.1 Intro
강화학습은 크게 두 가지 단계로 구분된다. 

- Policy Evaluation : 주어진 policy에서 total reward를 계산하는 방법
- Policy Improvement : Total reward를 maximize하는 쪽으로 policy를 구하는 방법

강화학습은 위 두 단계를 반복하여 policy를 구한다.

강화학습의 목적은 total sum of rewards를 maximize하는 optimal policy를 찾는 것.

- Deterministic : state와 action이 결정되면 noise 없이 모델의 output이 결정됨. $\to$ policy가 정해지면 episode가 정해짐 
- **Stochastic** : state와 action이 정해져도 확률적으로 output이 결정됨. $\to$ policy가 정해져도 여러가지 episode가 나옴.

**Model-based vs Model-free**

- Model-based : known MDP (Transition probability $P$를 알 때)
    + state와 action이 주어지면 next state에 대한 확률을 알 수 있음.
    + Dinamic programming

- Model-free : unknown MDP (Transition probability $P$를 모를 때)
    + Transition probability가 알려져있지 않아서 sample data를 기반으로 policy를 계산함.

### 1.2 Terminology

**Reward** 

Reward $R_t$는 시점 $t$에서 agent가 한 행동이 얼마나 좋은지를 평가하는 scalar feedback으로, agent는 cumulative sum of rewards(=total rewards)의 기댓값이 최대가 되도록 학습한다. 

보통 다음의 세 가지 타입의 reward function이 사용된다.
- $R = R(s)$ : 특정 state $s$에 있을 때 reward가 주어짐
- $R = R(s, a)$ : 특정 state $s$에서 특정 action $a$를 취하면 reward가 주어짐
- $R = R(s, a, s^\prime)$ : 특정 state $s$에서 특정 action $a$를 취해서 다음 state $s^\prime$으로 전이되면 reward가 주어짐 

**Return**

Return $G_t$는 total discounted reward를 의미한다.

$$G_t = R_{t+1} + \gamma R_{t+2} + \cdots = \sum_{k}^{\infty}\gamma^k R_{t+k+1}$$

Discount factor $\gamma \in [0, 1]$. stochastic의 특성 상 현재 시점부터 멀리 떨어진 시점에서 얻은 reward는 불확실하기 때문에 $\gamma$를 곱해서 적은 값을 얻게 한다.

**Policy**

Policy는 각 state마다 취할 action의 확률 분포로, return값을 최대화 하도록 각 state $s$마다 optimal action $a$를 정하는 가이드라인의 역할을 함.

$$\pi(a|s) = P(A_t = a | S_t = s)$$

- Known MDP에서는 deterministic optimal policy $\pi_*(s)$가 존재함.
- Unknown MDP에서는 sample data를 통해 stochastic policy를 구해야함.

### 1.3 Notation summary

**$P(X = x)$** : random variable $X$가 $x$일 확률 $\big(=p(x)\big)$

**$\mathbb{E}[X]$** : random variable $X$의 기댓값 $\big(\mathbb{X} = \sum_{x}xp(x)\big)$

**$\underset{a}{\max}f(a)$** : 집합 {$a$}에 대한 $f(a)$의 최대값

**$\underset{a}{\arg\max}f(a)$** : $f(a)$의 최대값을 갖게 하는 $a$값

**$S_t, A_t, R_t$** : 시점 $t$에서의 state, action, reward

**$G_t$** : 시점 $t$에서 얻는 total discounted reward (= return)

**$\pi$** : 모든 state에 대해서 취할 action의 확률 분포 (= policy)

**$\pi(a|s)$** : state $s$에서 action 
$a$를 취할 확률 (= stochastic policy)

**$v_\pi(s)$** : State를 평가하기 위한 state-value function으로, 특정 state $s$에서 policy 
$\pi$를 통해 얻을 수 있는 expected return 
$\mathbb{E}(G)$값. 해당 값이 크면 $s$는 좋은 state이고, 작으면 $s$는 좋지 않은 state가 됨

**$v_\*(s)$** : optimal state-value function으로, state $s$에서 optimal policy $\pi_*$를 통해 얻을 수 있는 expected return값

**$q_\pi(s,a)$** : Action을 평가하기 위한 action-value function으로, policy $\pi$가 주어져 있을 때, 특정 state $s$에서 action $a$를 취했을 때 얻을 수 있는 expected return 값.









## 2. Markov Decision Process (MDP)
### 2.1 Markov Property
- 과거의 상태가 미래의 상태를 결정하는데 영향을 미치지 않음. 즉, 현재 상태가 미래 상태를 결정하는 데 있어 과거의 상태들과 독립적임.
- 이러한 성질로 인해 Markov property를 memoryless property라고도 함.

> Given the present state $S_t = s,$ the future state $S_{t+1} = s^\prime$ does not depend on the past states
{: .prompt-info }

$$P(S_{t+1} = s^\prime | S_0 = s_0, S_1 = s_1, \cdots, S_t = s) = P(S_{t+1} = s^\prime | S_t = s)$$

### 2.2 Markov Process
Markov property를 만족하는 stochastic process를 Markov process라고 한다.

- $P(S_{t+1} = s^\prime | S_t = s)$ 를 state 
$s$에서 state 
$s^\prime$으로 가는 state transition probability라고도 함.

- Markov process is a tuple $(S, P)$.
    + S : a set of states
    + P : state transition probability matrix $[P_{ij}]$

$$P_{ij} = P_{s_i s_j} = p(s_j | s_i) = P(S_{t+1} = s_j| S_{t} = s_i)$$

Markov process의 장점 : Markov property를 만족하기 때문에, 과거의 state에 의존하지 않아서 모델을 단순하게 표현할 수 있다.

### 2.3 Markov Decision Process

Markov Process가 $(S, P)$의 튜플로 구성된 것과는 달리, MDP는 
$(S, A, P, R, \gamma)$의 튜플로 구성된다.

여기에서 $S$는 state space, $A$는 action space, $P$는 state transition probability를 나타내며, $R$은 reward function, $\gamma\in [0,1]$ 는 discount factor을 의미한다. 

MDP는 모든 state에서 Markov property를 만족한다. MDP에서의 "Decision"은 action을 의미하는 것으로, 현재 state $s$에서 특정 action $a$를 취해서 next state $s^\prime$으로 가는 확률 state transition probability $P$는 다음과 같이 표현된다.

$$P_{ss^\prime}^{a} = p(s^\prime|s, a) = P(S_{t+1}=s^\prime|S_{t}=s, A_t=a)$$

Reward function $R_{ss^\prime}^{a}$의 경우, 현재 state $s$에서 action $a$를 취해서 next state $s^\prime$으로 전이되었을 때 얻는 immediate reward를 의미한다.






















<!-- 
$$\nabla_{\theta} J_\theta \approx \int_{\tau} \sum_{t=0}^{\infty}\left[\nabla_\theta \ln P_\theta (a_t|s_t)G_t \right]P_\theta (\tau) d\tau$$

Where $\tau$ denotes the trajectory $(s_0, a_0, s_1, a_1, \cdots)$.

This proof is essential for finding the policy gradient $\nabla_\theta J_\theta$.

$$G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots $$

$$J_\theta = \mathbb{E}_{\tau} [G_0] = \int_\tau G_0 P_\theta (\tau) d\tau$$

## Solution

$$
\begin{aligned}
J_\theta &= \int_\tau G_0 P_\theta (\tau) d\tau\\
\nabla_\theta J_\theta &= \nabla_\theta \int_\tau G_0 P_\theta (\tau) d\tau = \int_\tau G_0 \nabla_\theta P_\theta (\tau) d\tau
\end{aligned}
$$

[1]
> We want this to appear in probability form, $$\int_x x P(x) dx$$, so we use a little trick.\\
> $$\nabla_\theta P_\theta(\tau) = P_\theta(\tau) \nabla_\theta \ln P_\theta(\tau) $$
{: .prompt-tip }

Then, 

$$
\begin{aligned}
\nabla_\theta J_\theta & = \int_\tau G_0 P_\theta(\tau) \nabla_\theta \ln P_\theta(\tau) d\tau
\end{aligned}
$$

[2]
> by Bayes' rule, 
> $$P(A, B) = P(A)P(B|A)$$
{: .prompt-tip }



$$
\begin{aligned}
    P_\theta(\tau) &= P_\theta(s_0, a_0, s_1, a_1, \cdots)\\
                   &= P_\theta(s_0)P_\theta(a_0, s_1, a_1, \cdots|s_0)\\
                   &= P_\theta(s_0)P_\theta(a_0|s_0)P_\theta(s_1, a_1, s_2, \cdots|s_0, a_0)\\
                   &= P_\theta(s_0)P_\theta(a_0|s_0)P_\theta(s_1 | s_0, a_0)P_\theta(s_2, a_2, s_3, \cdots|s_0, a_0, s_1)\\
                   & \quad \vdots\\
                   & = P_\theta(s_0)P_\theta(a_0|s_0)P_\theta(s_1|s_0, a_0)P_\theta(a_1 | s_0, a_0, s_1)P_\theta(s_2|s_0, a_0, s_1, a_1) \cdots\qquad\qquad\\
\end{aligned}
$$

- Here, 
$P_\theta(a_1|s_0, a_0, s_1) = P_\theta(a_1|s_1)$ since it satisfies the Markov property.

> Markov property : The future state depends solely on the current state and not on past states or actions.
{: .prompt-tip }

- And, because the state $s_t$ does not depend on 
$\theta, P_\theta(s_t | s_{t-1}, a_{t-1}) = P(s_t | s_{t-1}, a_{t-1})$.

Then, we can simpify as following 

$$
\begin{aligned}
    P_\theta(\tau) &= P_\theta(s_0, a_0, s_1, a_1, \cdots)\\
                   &= P(s_0)P_\theta(a_0, s_1, a_1, \cdots|s_0)\\
                   &= P(s_0)P_\theta(a_0|s_0)P_\theta(s_1, a_1, s_2, \cdots|s_0, a_0)\\
                   &= P(s_0)P_\theta(a_0|s_0)P(s_1 | s_0, a_0)P_\theta(s_2, a_2, s_3, \cdots|s_0, a_0, s_1)\\
                   & \quad \vdots\\
                   & = P(s_0)P_\theta(a_0|s_0)P(s_1|s_0, a_0)P_\theta(a_1 | s_1)P(s_2|s_1, a_1) \cdots\\
\end{aligned}
$$


What we want to get is the term of $$\nabla_\theta \ln P_\theta(\tau)$$ 

$$
\begin{aligned}
    \nabla_\theta \ln P_\theta(\tau) & = \nabla_\theta  \big[\ln P(s_0) + \ln P_\theta(a_0|s_0) + \ln P(s_1|s_0, a_0) + \ln P_\theta(a_1 | s_1) + \ln P(s_2|s_1, a_1) \cdots \big]\\

    &= \nabla_\theta \ln P_\theta (a_0|s_0) + \nabla_\theta \ln P_\theta (a_1, s_1) + \nabla_\theta \ln P_\theta (a_2|s_2) + \cdots\\
    &= \nabla_\theta\sum_{t=0}^{\infty} \ln P_\theta (a_t|s_t)
\end{aligned}
$$

Substituting equation 
$\nabla_\theta \ln P_\theta(\tau)= \nabla_\theta\sum_{t=0}^{\infty} \ln P_\theta (a_t|s_t)$ into equation 
$\int_\tau G_0 P_\theta(\tau) \nabla_\theta \ln P_\theta(\tau) d\tau$


$$
\begin{aligned}
\nabla_\theta J_\theta & = \int_\tau G_0 P_\theta(\tau) \nabla_\theta \ln P_\theta(\tau) d\tau\\

&= \int_\tau G_0 P_\theta(\tau)\nabla_\theta \sum_{t=0}^{\infty} \ln P_\theta (a_t|s_t) d\tau
\end{aligned}
$$

[3]

Here, $$G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \gamma^3 R_{t+3} +\cdots$$

$$
\begin{aligned}
\nabla_\theta J_\theta &= \int_\tau G_0 P_\theta(\tau)\nabla_\theta \sum_{t=0}^{\infty} \ln P_\theta (a_t|s_t) d\tau\\
&= \int_\tau \big(R_0 + \gamma R_1 + \gamma^2 R_2 + \cdots\big) \big(\nabla_\theta \ln P_\theta(a_0|s_0) + \nabla_\theta \ln P_\theta(a_1|s_1) + \cdots \big) P_\theta(\tau)d\tau

\end{aligned}
$$

At specific time step $$t^*$$, we intuitively know that R_t occurring at time $$t < t^*$$ does not affect 
$$P_\theta(a_{t}^*|s_{t}^*)$$.

We can then simplify the expression as follows:

$$
\nabla_\theta J_\theta = \int_\tau \sum_{t=0}^{\infty} \big[ \nabla_\theta \ln P_\theta (a_t|s_t) \sum_{k=t}^{\infty}\gamma^{k} R_k \big] P_\theta(\tau) d\tau
$$

$$G_t$$ can be represented as $$\sum_{k=0}^{\infty}\gamma^{k} R_{t+k} = \sum_{k=t}^{\infty}\gamma^{k-t}R_{k}$$

Then 

$$
\begin{aligned}
\nabla_\theta J_\theta &= \int_\tau \sum_{t=0}^{\infty} \big[\nabla_\theta \ln P_\theta(a_t|s_t) \sum_{k=t}^{\infty} \gamma^{t}\gamma^{k-t}R_k \big]P_\theta(\tau)d\tau\\
&= \int_\tau \sum_{t=0}^{\infty} \big[\nabla_\theta \ln P_\theta(a_t|s_t)\gamma^{t} G_t \big]P_\theta(\tau)d\tau



\end{aligned}
$$

<!-- # RL Keywords
- Environment : 에이전트가 액션을 취하는 환경
- State : 에이전트의 상태. 시점 $t$에서의 상태 $s_t \in \mathcal{S}$ ($\mathcal{S}$ : State space)
- Reward : 에이전트가 한 번 학습했을 때 주어지는 보상. 보상함수 $r : \mathcal{S} \to \mathbb{R} $
- $\rho_0 : \mathcal{S} \to [0, 1]$은 초기 상태의 확률 분포
- $\gamma$ : 할인율 
- Action : 에이전트가 취하는 행동. 시점 $t$에서의 행동 $a_t \in \mathcal{A}$ ($\mathcal{A}$ : Action space). Action space는 Environment에 의해 결정됨. 행동 집합에 따라 Discrete action space, continuous action space로 구분됨.
- Policy : 학습을 통해 구하려는 함수. 상태 $s$에서 행동 $a$를 취할 확률. $\pi : \mathcal{S} \times \mathcal{A} \to [0, 1]$
- 전이 확률 분포 : $P : \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]$

# RL
강화학습은 주어진 Environmnet에서 State를 기준으로, 최고의 Action을 학습해 나가는 과정. 
Policy-based 학습은 Action을 결정하는 policy를 학습하는 것을 목적으로 한다.

Discrete한 상태 공간$\mathcal{S}$와 행동 공간$\mathcal{A}$에 대하여, 모든 (s, a) 순서쌍에 대한 정책의 행동 가치함수 Q_{\pi}(s, a)를 계산하고 이를 policy improvement에 사용한다. 

$$\pi_{\text{new}}(s) = \underset{a \in \mathcal{A}}{\arg\max} Q_{\pi_{\text{old}}}(s, a), \forall s \in S$$

위의 식을 통해 정책을 개선하면, monotonic improvement가 보장되고, 결국 최적의 정책에 수렴한다.

![Desktop View](https://github.com/mjkmain/blog-image/assets/72269271/4c2c98fc-210b-49f4-b970-6cc125b8be28){: width="680" } 
_The concept of monotonic_

하지만, 상태 공간과 행동 공간이 continuous 한 경우, 우리는 모든 순서쌍 $(s, a)$를 고려할 수 없을 뿐만 아니라, $\arg\max$ 또한 구할 수 없다. 이 경우 function approximation을 사용하며, 행동가치함수를 추정하거나 정책을 모델링하게 된다. 하지만 이런 approximation은 결국 추정에 대한 오차를 발생시키기 때문에 monotonic improvement를 보장할 수 없다.



# TRPO
Trust Region Policy Optimization(TRPO)

TRPO는 Policy-based 알고리즘을 기반으로 하여, Trust Region에서만 update를 진행한다.


$$\eta(\pi) = \mathbb{E}_{s_0, a_0, \cdots}\left[\sum_{t=0}^{\inf} \gamma^{t}r(s_t)\right]$$

$$ \text{where} s_0 \sim \rho_0(s_0), a_t \sim \pi(a_t|s_t), s_{t+1} \sim P(s_{t+1} | s_t, a_t)$$ --> 