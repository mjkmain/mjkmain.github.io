---
title: Policy based RL
author: mjkmain
date: 2024-01-02 03:54:00 +0900
categories: [NLP]
tags: [NLP, LLM]
pin: true
math: true
render_with_liquid: false
comments: True
---



# Questions

## Q1. Policy-based algorithm 

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