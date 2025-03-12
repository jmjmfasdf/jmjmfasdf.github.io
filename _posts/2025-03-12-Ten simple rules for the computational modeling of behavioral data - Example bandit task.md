---
title: "Ten simple rules for the computational modeling of behavioral data"
date: 2025-03-12
tags:
    - Modeling
    - fMRI
    - EEG
    - Behavioral Data
categories: 
    - Paper Review
toc: true
toc_sticky: true
---

# An illustrative example: the multi-armed bandit task

논문에서 제시하는 10가지 모델링 규칙은 매우 일반적이지만, 이를 보다 구체적으로 설명하기 위해 강화학습(reinforcement learning) 모델을 활용한 선택 과제(choice task) 를 예시로 들고 있다. 이 논문에서 다루는 예제들은 사람들이 어떻게 보상을 극대화하는지를 학습하는 과정을 분석하는 것이 목표이다. 특히, 최적의 선택이 처음에는 알려져 있지 않은 상황에서 학습이 어떻게 진행되는지를 연구한다.

## 멀티 암드 밴딧(Multi-Armed Bandit) 과제란?

이 실험은 여러 개의 슬롯머신(slot machine, 또는 "one-armed bandit") 중 하나를 선택하는 방식으로 진행된다. 실험 참가자는 T번의 선택(trials) 을 수행하며, K개의 슬롯머신 중 하나를 선택해야 한다. 각 슬롯머신 $$k$$ 는 선택(trial $$t$$)될 때, 특정한 확률 $$θ_k$$ 로 보상(reward, $$r_t$$)을 지급한다.

즉, 슬롯머신 $$k$$ 를 선택했을 때, 보상 $$r_t$$ 을 받을 확률은 $$θ_k$$ 이며, $$θ_k$$ 는 슬롯머신마다 다르며, 실험 참가자는 이를 사전에 알지 못한다(unknown to the subject). 가장 단순한 실험에서는 보상 확률 $$θ_k$$ 가 시간에 따라 변하지 않고 고정(fixed)되어 있다.

## 멀티 암드 밴딧 과제의 실험 변수

이 실험에서는 다음과 같은 3가지 주요 변수를 설정할 수 있다.

- 시행 횟수 ($$T$$): 참가자가 선택을 몇 번 반복하는가?  
- 슬롯머신 개수 ($$K$$): 몇 개의 슬롯머신이 있는가?  
- 보상 확률 ($$θ_k$$): 각 슬롯머신이 보상을 줄 확률이 얼마인가?

이러한 변수 설정이 실험의 결과에 중요한 영향을 미치며, 논문에서 사용한 기본 실험 설정은 다음과 같다.

- 시행 횟수: $$T = 1000$$
- 슬롯머신 개수: $$K = 2$$
- 슬롯머신의 보상 확률:
- 슬롯머신 1: $$θ_1 = 0.2$$ (20% 확률로 보상)
- 슬롯머신 2: $$θ_2 = 0.8$$ (80% 확률로 보상)

즉, 한 참가자는 두 개의 슬롯머신 중 하나를 선택해야 하며, 각각의 슬롯머신이 보상을 줄 확률은 미리 정해져 있지만 참가자는 이를 모른다. 실험 참가자는 반복적인 선택을 통해 어느 슬롯머신이 더 높은 보상을 주는지 학습해야 한다.

## 이 과제가 왜 중요한가?

멀티 암드 밴딧 과제는 탐색-활용(exploration-exploitation) 문제를 연구하는 데 매우 유용하다. 탐색(Exploration)은 더 나은 선택을 하기 위해 미지의 선택지를 시도하는 것이며, 활용(Exploitation)은 현재까지 학습한 정보를 바탕으로 가장 보상이 높을 것으로 예상되는 선택지를 선택하는 것이라 할 수 있다.

예를 들어, 초반에는 두 슬롯머신의 보상 확률을 모른 상태이므로 여러 번 선택을 시도하며 탐색(exploration) 을 수행해야 한다. 실험이 진행됨에 따라 더 높은 확률로 보상을 주는 슬롯머신(슬롯머신 2, $$θ_2$$ = 0.8)을 인식하고 이를 점점 더 많이 선택하는 전략(활용, exploitation) 을 사용할 것이다. 이 과제는 강화학습 모델을 테스트하는 데 매우 적합하며, 다양한 학습 알고리즘을 비교하는 데 사용할 수 있다.

<br>

# Box 1. Example: Modeling behavior in the multi-armed  bandit task.

이 논문에서는 Multi-Armed Bandit Task에서 사람들이 어떻게 행동하는지를 설명하는 다섯 가지 모델을 제시하고 있다. 각 모델의 핵심 개념과 수식을 설명하면 다음과 같다.

## Model 1: Random Responding (랜덤 응답 모델)

이 모델은 참가자가 과제에 전혀 몰입하지 않고 단순히 무작위로 버튼을 누르는 경우를 가정한다. 그러나 참가자가 특정 선택지에 대한 선호도(bias)를 가질 수도 있다고 본다. 이 선호도는 파라미터 $$ b $$ 로 표현되며, 다음과 같이 선택 확률을 결정한다.

$$
p_1 = b, \quad p_2 = 1 - b
$$

즉, $$ b $$ 값이 0.5이면 두 선택지를 균등한 확률로 선택하고, $$ b $$ 값이 1에 가까울수록 특정 선택지를 선호하게 된다. 이 모델은 단 하나의 자유 파라미터 $$ b $$ 만을 가진다

## Model 2: Noisy Win-Stay-Lose-Shift (노이즈가 추가된 승리-유지, 패배-전환 모델)

이 모델은 보상을 받으면 같은 선택을 반복하고, 보상을 받지 못하면 선택을 바꾸는 단순한 전략을 따른다. 그러나 이 전략을 항상 적용하는 것이 아니라 확률적으로 적용하는데, 확률 $$ 1 - \epsilon $$ 로 win-stay-lose-shift 규칙을 따르고, 확률 $$ \epsilon $$ 로 랜덤 선택을 한다. 이 모델에서 선택 확률은 다음과 같다.

$$
p_k^t =
\begin{cases}
1 - \frac{\epsilon}{2}, & \text{if } (c_{t-1} = k \text{ and } r_{t-1} = 1) \text{ OR } (c_{t-1} \neq k \text{ and } r_{t-1} = 0) \\
\frac{\epsilon}{2}, & \text{if } (c_{t-1} \neq k \text{ and } r_{t-1} = 1) \text{ OR } (c_{t-1} = k \text{ and } r_{t-1} = 0)
\end{cases}
$$

이 모델은 전체적인 랜덤성 수준을 조절하는 단 하나의 자유 파라미터 $$ \epsilon $$ 을 가진다.

## Model 3: Rescorla-Wagner (레스콜라-와그너 학습 모델)

이 모델은 참가자가 이전 결과를 바탕으로 각 슬롯 머신(옵션)의 기대 가치를 학습한다고 가정한다. 학습 규칙은 Rescorla-Wagner 학습 규칙을 따른다.

$$
Q_k(t+1) = Q_k(t) + \alpha (r_t - Q_k(t))
$$

여기서,

$$ Q_k(t) $$ 는 시간 $$ t $$ 에서의 선택지 $$ k $$ 의 기대 가치,  
$$ r_t $$ 는 시간 $$ t $$ 에서 받은 보상,  
$$ \alpha $$ 는 학습률로, 0과 1 사이의 값을 가지며, 보상이 가치 업데이트에 미치는 영향을 조절한다.  

의사결정 과정은 Softmax 선택 규칙을 따른다.

$$
p_k^t = \frac{\exp(\beta Q_k^t)}{\sum_{i=1}^{K} \exp(\beta Q_i^t)}
$$

여기서, $$ \beta $$ 는 ‘inverse temperature’로, 값이 크면 높은 가치를 가진 옵션을 더 확실하게 선택하고, 값이 작으면 랜덤한 선택을 많이 하게 된다. 이 모델은 두 개의 자유 파라미터 $$ (\alpha, \beta) $$ 를 가진다.

## Model 4: Choice Kernel (선택 경향성 모델)

이 모델은 사람들이 단순히 기대 가치를 기반으로 선택하는 것이 아니라, 과거에 선택했던 행동을 반복하는 경향이 있음을 반영한다. 이를 위해 Choice Kernel이라는 개념을 도입한다. Choice Kernel은 최근 선택했던 행동의 빈도를 추적하며, 다음과 같은 업데이트 방식을 따른다.

$$
CK_k(t+1) = CK_k(t) + \alpha_c (a_k(t) - CK_k(t))
$$

여기서, $$ a_k(t) = 1 $$ 이면 선택한 옵션이고, 그렇지 않으면 0이다. $$ \alpha_c $$ 는 선택 경향성을 업데이트하는 학습률이다. 이후 선택 확률은 Softmax 형태로 결정된다.

$$
p_k = \frac{\exp(\beta_c CK_k)}{\sum_{i=1}^{K} \exp(\beta_c CK_i)}
$$

이 모델은 두 개의 자유 파라미터 $$ (\alpha_c, \beta_c) $$ 를 가진다.

## Model 5: Rescorla-Wagner + Choice Kernel (강화 학습 + 선택 경향성)

이 모델은 강화 학습 모델과 선택 경향성 모델을 결합한 가장 복잡한 모델이다. 이 모델에서는 기대 가치를 업데이트하는 Rescorla-Wagner 모델과 과거 선택 경향성을 반영하는 Choice Kernel 모델이 함께 사용된다.

최종적으로 선택 확률은 다음과 같이 계산된다.

$$
p_k = \frac{\exp(\beta Q_k + \beta_c CK_k)}{\sum_{i=1}^{K} \exp(\beta Q_i + \beta_c CK_i)}
$$

즉, 선택 확률이 기대 가치 $$ Q_k $$ 와 선택 경향성 $$ CK_k $$ 의 합에 기반하여 결정된다. 이 모델은 총 네 개의 자유 파라미터 $$ (\alpha, \beta, \alpha_c, \beta_c) $$ 를 가진다.

이 다섯 개 모델은 참가자가 Multi-Armed Bandit Task에서 어떻게 선택을 하는지를 수학적으로 설명하기 위한 것이다. 모델의 복잡도는 점점 증가하며, 가장 간단한 모델은 Random Responding이고, 가장 복잡한 모델은 Rescorla-Wagner + Choice Kernel 모델이다.

- Random Responding: 무작위 선택 (자유 파라미터: $$ b $$)
- Noisy Win-Stay-Lose-Shift: 보상에 따라 반복 또는 변경 (자유 파라미터: $$ \epsilon $$)
- Rescorla-Wagner: 강화 학습 (자유 파라미터: $$ \alpha, \beta $$)
- Choice Kernel: 과거 선택 반복 경향 반영 (자유 파라미터: $$ \alpha_c, \beta_c $$)
- Rescorla-Wagner + Choice Kernel: 강화 학습 + 선택 경향 (자유 파라미터: $$ \alpha, \beta, \alpha_c, \beta_c $$)

각 모델은 인간의 학습 및 의사결정 과정을 수량화할 수 있도록 하며, 연구자들은 실험 데이터를 이 모델에 적합시켜 인간 행동의 기저 메커니즘을 분석할 수 있다.

<br>

# Box 2. Example: simulating behavior in the bandit task.

## 1. 행동 시뮬레이션을 위한 실험 설정

행동을 시뮬레이션하기 위해 먼저 실험 조건을 정의해야 한다. 여기서 시뮬레이션의 실험 조건은 실제 실험과 동일해야 한다. 실험과 다른 설정을 사용하면 결과를 직접 비교하기 어려워진다.

- 총 시행 횟수 $$ T = 1000 $$  
- 선택지(슬롯 머신)의 개수 $$ K = 2 $$  
- 각 슬롯 머신의 보상 확률:  
- Bandit 1: $$ p = 0.2 $$  
- Bandit 2: $$ p = 0.8 $$  

## 2. 모델 파라미터 설정

각 모델의 자유 파라미터를 정의해야 하며, 이를 설정하는 방법 중 하나는 사전 확률 분포(prior distribution)에서 무작위로 샘플링하는 것이다. 사전 분포(prior distribution)은 가능한 한 넓게(broad) 설정하는 것이 일반적이나, 특정 모델에 대한 기존 연구 결과가 있다면, 문헌에서 보고된 값을 참고하여 설정할 수도 있다.

## 3. 시뮬레이션 과정

시뮬레이션은 다음과 같은 절차를 따른다.

1. 모델에 따라 각 선택지의 선택 확률 $$ p_k(1) $$ 을 계산한 후, 그 확률에 따라 선택 $$ a_1 $$ 를 결정한다.
2. 선택한 슬롯 머신의 보상 확률을 고려하여 $$ r_1 $$ (보상 여부)를 결정한다.
3. Model 2~5는 이전 선택(action) 및 보상(outcome) 정보를 이용하여 다음 시행의 선택 확률을 업데이트한다. 즉, $$ p_k(t) $$ 를 갱신하여 $$ a_t $$ 의 확률을 조정한다.
4. 이 과정을 $$ T $$ 번 반복하여 시뮬레이션을 완료한다. 이를 통해 한 명의 가상 참가자(simulated participant)의 데이터를 생성할 수 있다.
5. 단일 시뮬레이션만으로는 모델의 특성을 충분히 이해하기 어렵기 때무에, 여러 파라미터 값에 대해 반복적인 시뮬레이션을 수행하여 모델이 어떻게 동작하는지 확인한다.

## 4. 시뮬레이션 결과 분석

모델의 행동을 이해하기 위해 모델 독립적인 측정값(model-independent measures)을 정의하고 이를 시각화한다.

이 예시에서 알아볼 측정값은 $$ p(stay) $$ (이전 시행에서 같은 선택을 유지할 확률)이다. "나는 피드백에 따라 행동을 바꿀 것인가?" 라는 질문에 대한 대답으로, 특정 모델이 보상에 어떻게 반응하는지를 보여준다. **Win-Stay-Lose-Shift(WSLS) 모델(Model 2)**에서는 보상에 강하게 의존하지만, **Random Responding(Model 1)**에서는 보상과 무관하였으며, 이는 Box 2—figure 1A에서 확인 가능하다.

두 번째 측정값은 $$ p(correct) $$ (올바른 선택을 할 확률)이다. 이는 "나는 학습을 통해 정답을 찾고 있는가?" 라는 질문의 대답이며, 학습이 진행됨에 따라 $$ p(correct) $$ 가 증가해야 한다. Box 2—figure 1B에서 Model 3(Rescorla-Wagner) 시뮬레이션 결과를 확인할 수 있다.

<figure class='align-center'>
    <img src = "/images/2025-03-12-Ten simple rules for the computational modeling of behavioral data/figure1.jpg" alt="">
    <figcaption>figure 1. Simulating behavior in the two-armed bandit task.</figcaption>
</figure>

## 5. 모델 파라미터에 따른 행동 변화

Box 2—figure 1B에서는 Model 3(Rescorla-Wagner)의 파라미터 값 변화에 따른 학습 성능을 분석한다. 초기 학습(early trials)에서 $$ \alpha $$ 가 크면 학습 속도가 빠르지만, $$ \beta $$ 가 클수록 최적의 $$ \alpha $$ 값은 작아진다. 즉, 높은 $$ \beta $$ 값(결정적 선택일 때)에서는 낮은 $$ \alpha $$ 가 학습을 더 잘 유도한다.

후기 학습(late trials)에서는 높은 $$ \beta $$ 값에서는 $$ \alpha $$ 가 너무 크면 학습이 불안정해지고, $$ p(correct) $$ 가 감소한다. 즉, 높은 $$ \alpha $$ 값이 항상 좋은 것은 아님. 학습이 어느 정도 안정화되면 $$ \alpha $$ 가 너무 크면 불리할 수도 있다.

특정 $$ \beta $$ 값에서 $$ \alpha $$ 가 지나치게 작거나 크면 학습이 비효율적이었으며, 초기에는 높은 $$ \alpha $$ 가 빠른 학습을 유도하지만, 후기에는 낮은 $$ \alpha $$ 가 더 안정적인 성능을 보인다.

## 6. 모델 독립적 측정값을 어떻게 선택할 것인가?

모델 독립적인 측정값을 정하는 것은 쉬운 문제가 아니다.

측정값은 전체적인 행동 특성을 반영해야 하며,  모델 간 차이를 진단할 수 있어야 한다. 실험의 연구 목적에 따라 측정값을 신중히 선택해야 한다. 예를 들어, 특정 연구에서는 보상에 대한 반응성이 중요할 수도 있고, 다른 연구에서는 **탐색 vs. 활용 전략(exploration-exploitation tradeoff)**이 핵심일 수도 있다.






