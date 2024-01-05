---
title: Policy based RL
author: mjkmain
date: 2024-01-02 03:54:00 +0900
categories: [NLP]
tags: [NLP, LLM]
pin: true
math: true
render_with_liquid: false
---



# Questions

## Q1. (REINFORCE algorithm) Proof 

$$\nabla J_\theta \approx \int_{\tau} \sum_{t=0}^{\infty}\left[\nabla_\theta \ln P_\theta (a_t|s_t)G_t \right]P_\theta (\tau) d\tau$$

Where $\tau$ denotes the trajectory $(s_0, a_0, s_1, a_1, \cdots)$.

This proof is essential for finding the policy gradient $\nabla_\theta J_\theta$.

$$J_\theta = \mathbb{E}_{\tau} [P_{\theta}(\tau)] = \int_\tau G_0 P_\theta (\tau) d\tau$$

$$G_0 = R_0 + \gamma R_1 + \gamma^2 R_2 + \cdots $$







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