---
title: "Representational similarity analysis in neuroimaging: proxy vehicles and provisional representations"
date: 2025-03-18
tags:
    - fMRI
    - MVPA
    - RSA
    - Neuroimaging
categories: 
    - Paper Review
toc: true
toc_sticky:  true
---

기능적 신경영상(fMRI)은 종종 "뇌에서 어떤 일이 일어나는지"만 보여줄 뿐, "어떻게 일어나는지"에 대한 정보를 제공하지 못한다는 비판을 받아왔다. 그러나 최근의 분석 기법들은 심리학에서 관심을 가지는 개념들에 대한 접근을 가능하게 만들고 있다. 저자는 뇌의 대규모 신경 표현 구조를 이해하는 데 있어서 neuroimaging이 중요한, 하지만 제한적인 창을 제공할 수 있다고 주장한다.

논문에서는 특히 Representational Similarity Analysis(RSA)에 대해 논의하며, 이를 통해 신경 표현이 어떻게 구성되는지를 설명하려고 한다. 저자는 신경 표현의 일반적인 요구 조건(desiderata)을 설명하고, RSA가 신경 표현에 대해 무엇을 말해줄 수 있으며, 무엇을 말해줄 수 없는지를 탐구한다. 또한, RSA와 fMRI를 심리학에서 대표적으로 사용되는 다른 실험 패러다임과 비교하며, RSA가 이에 비해 어떻게 유리한지를 논의한다.

<br>

# Introduction

심리학과 인지과학에서는 **정신적 표상(mental representation)의 존재**와 실재성 에 대한 지속적인 논쟁이 이어져 왔다. 즉, 정신적 표상이 객관적으로 실재하는가, 단순한 유용한 허구인가, 혹은 존재하지 않는가 에 대한 의문이 제기된다. 신경과학자들 또한, 뇌가 정신적 표상을 실현하는가, 아니면 **단순한 인과적 역동 시스템(dynamic causal system)일 뿐, 의미 있는 방식으로 어떤 것도 표상하지 않는가** 를 두고 논쟁해 왔다. **그렇다면 신경 표상의 존재를 입증하는 증거는 무엇인가?**

과거 Hubel과 Wiesel(1959, 1962, 1998)의 연구는 고양이 및 원숭이 시각 피질의 뉴런 활동을 기록하여, 뉴런이 시각 세계를 어떻게 표상하는지 밝히려는 시도 로 간주되었다. 이후 단일 뉴런(single-cell) 연구 들이 신경 표상의 존재를 입증하는 결정적 증거 로 받아들여지기도 했다. 그러나 기능적 신경영상(fMRI)은 단순히 뇌의 특정 영역이 활성화되는 위치를 알려줄 뿐, 정보가 어떻게 표상되는지를 보여주지 못한다는 비판 을 받아왔다(Coltheart, 2006a, b; Fodor, 1999).

저자는 이러한 비판에 대해 반론을 제기한다. 신경영상 연구는 초기부터 단순한 활성화 위치 탐색 이상의 역할을 수행해 왔으며(Roskies, 2009), 최근 분석 기법의 발전으로 인해 신경영상이 인지과학의 질문을 탐구하는 도구로 더욱 유용해졌다 고 주장한다. 특히, **단변량(univariate) 분석에서 다변량(multivariate) 분석으로의 전환**은 신경영상의 해석 방식을 크게 변화시켰다. 기존의 연구는 주로 인지 과제 수행 시 특정 뇌 영역의 활성화 변화를 탐색하는 데 초점을 맞추었다. 하지만, 최근 연구에서는 신경 신호의 복잡한 패턴과 그것들이 과제 요구(task demands) 및 서로 간의 관계를 통해 어떻게 조직되는지를 분석하는 방향으로 변화하고 있다.

<br>

# Functional MRI

fMRI는 자기공명(MR) 기술을 이용하여 특정 과제를 수행하는 동안 뇌의 신경 활동 패턴 변화를 추론하는 기법 이다. 이 기술은 oxygenated hemoglobin과 (deoxygenated hemoglobin)의 비율을 측정하며, 이는 혈류량과 상관관계가 있으며, 궁극적으로 국소적 신경 활동과 연관된다. 하지만 fMRI의 해상도는 대략 3mm³ 크기의 복셀(voxel) 단위 로 제공되며, 각 복셀에는 수백만 개의 뉴런이 포함되어 있기 때문에 세밀한 뉴런 수준의 정보를 제공할 수는 없다.

과거 fMRI 연구는 양전자 방출 단층촬영(PET)의 방법론을 기반으로 발전했으며, 특정 과제와 기준 조건 간의 활성화 차이를 비교하는 단변량(univariate) 방식으로 진행되었다. dl 이 시기에는 특정 뇌 영역에서 과제 수행과 관련된 기능적 변화의 위치를 찾는 것에 초점을 맞추었으나 이후 다변량(multivariate) 분석 기법의 도입으로 인해 fMRI의 분석 역량이 확장되었다.

MVPA(Multivoxel Pattern Analysis)는 복셀 간의 패턴 정보를 활용하여 뇌의 신경 정보를 분석하는 방법이다(Haxby et al., 2001). 예를 들어, 9개의 연속된 복셀이 세 가지 실험 조건에서 동일한 평균 활성화를 보일 경우, 단변량 분석은 이를 구별하지 못하지만, 다변량 분석은 조건 간 활성화 패턴 차이를 감지할 수 있다. 즉, MVPA는 패턴 정보를 활용하여 신경 신호 내에 포함된 정보를 추출하는 데 유리하다.

MVPA의 한 가지 응용 방식은 다중복셀 패턴 분류(MVPC, Multi-Voxel Pattern Classification)로, 기계 학습(classifiers)을 활용하여 특정 신경 패턴을 학습한 후, 새로운 데이터를 분류하는 방식 이다(Haxby et al., 2001). 예를 들어, 초기 연구에서는 후두엽(occipital cortex)의 다중복셀 데이터를 활용하여, 다양한 시각 자극 간의 구별이 가능함을 보였다. 또한, 안면 처리 영역(face-selective region)인 FFA(fusiform face area) 가 단변량 분석에서는 얼굴 정보만 처리하는 것으로 보였지만, MVPC를 적용한 결과 FFA가 얼굴 외 다른 자극도 구별할 수 있는 충분한 정보를 포함 하고 있음이 밝혀졌다.

MVPC는 현재 fMRI 데이터의 표준으로 간주되지만 neural representation을 분석하는 데 있어서 한계가 존재한다. MVPC에서 linear classifier를 사용할 경우 서로 다른 조건에서의 뇌 활동 패턴이 선형적으로 구분 가능한지는 평가할 수 있다. population coding 이론에 따르면, 이러한 선형적 구분 가능성은 다운스트림(downstream) 뇌 영역이 해당 정보를 읽어낼 수 있음을 시사하지만(Kriegeskorte & Kievit, 2013), 실제로 그렇게 이루어지는지를 입증하는 것은 아니다.(Diedrichsen & Kriegeskorte, 2017; Schalk et al., 2017). 더 나아가, 신경영상 연구는 실험 조건과 행동 결과에 따라 변하는 신경 활동을 측정하지만, 이는 단순한 상관관계일 뿐이며, 특정 신경 활동이 특정 기능을 수행하는 데 필수적인지를 입증하지는 못한다. 따라서 신경 표상의 인과성을 입증하려면 경두개 자기자극(TMS)이나 병변 연구와 같은 개입적(interventional) 방법이 필요하다(Schalk et al., 2017).

MVPC의 또 다른 한계는 분류 결과를 가능하게 한 신경 정보의 본질을 명확히 밝히지 못한다는 점이다. machine learning은 훈련 데이터에서 가장 분류에 유용한 특징(feature)을 학습하지만, 그것이 **실험자가 의도한 신호인지, 심리적으로 중요한 정보인지, 혹은 뇌가 실제로 사용하는 정보인지** 확신할 수 없다(Kriegeskorte & Diedrichsen, 2019). 즉, MVPC는 성공적인 분류 결과가 어떤 신경 정보에 기반하는지를 명확하게 설명하지 못한다(Ritchie et al., 2019). 또한, MVPC는 공간적으로 모호하다(spatial ambiguity). 동일한 분류 성능을 제공하는 신경 패턴이 뇌의 여러 다른 영역에서도 나타날 수 있으며, MVPC 자체는 분류 성능을 결정하는 정보가 특정한 뇌 영역에서 비롯된 것인지, 혹은 여러 영역에서 분산적으로 생성된 것인지에 대한 정보를 제공하지 않는다(Naselaris & Kay, 2015). 

<br>

# Representational similarity analysis (RSA)

## What is RSA?

상술한 한계점에도, 특정 지역의 pattern은 여전히 중요한 정보를 가지고 있다. 이에 따라 단순히 특정 뇌 영역의 활성화 패턴을 분류하는 것 이상의 접근 방식이 필요해졌다. 이후 등장한 분석 기법 중 하나가 **표상 유사성 분석(RSA, Representational Similarity Analysis)**이다. RSA는 뇌 활동 패턴 간의 유사성을 분석하는 방법 으로, 기존의 분류(classification) 방식과는 달리 내부 유사성 관계 에 초점을 맞춘다. 구체적으로, RSA는 실험 조건 간의 신경 활동 패턴을 쌍(pair)으로 비교하여 유사성(similarity)을 계산 하며, 이를 이론적 혹은 실험적으로 도출된 구조화된 모델과 비교 한다(Diedrichsen & Kriegeskorte, 2017). 

RSA의 핵심 개념 중 하나는 **표상 공간(representational space)** 이다. 표상 공간은 데이터셋 내에서 **측정값들 간의 유사성을 정의하는 수학적 공간** 으로, 이 공간 내의 distance는 선택한 metric에 따라 정의된다. 이처럼 유사성 공간 내의 구조를 representational geometry라고 하며, 이는 해당 공간에서 점들 간의 관계를 반영한다. 예를 들어, 특정 사물(object)에 대한 표상 공간에서 동물은 특정한 영역에 집중되어 있고, 무생물과는 명확히 구분될 수 있다. 또한, 동물 내에서도 네 발 달린 동물과 날개 달린 동물이 서로 다른 군집을 형성 할 수 있다.(Figure 1)

<figure class='align-center'>
    <img src = "/images/2025-03-18-Representational similarity analysis in neuroimaging proxy vehicles and provisional representations/figure1.jpg" alt="">
    <figcaption>figure 1. (A) First order RSA: Differences between patterns of activity in a chunk of tissue responding to two objects, here a hand and an umbrella, populate once cell of an RDM in (B). (B) A complete RDM can now be compared using second order RSA with other RDMs constructed from behavior, input measures, or other models.</figcaption>
</figure>

RSA의 가장 중요한 특징은, 유사성 공간을 matrix 형태로 변환하여 다양한 비교를 가능하게 한다는 점 이다. first-order RSA는 하나의 데이터셋에서 유사성 공간을 행렬로 변환하는 방식이며, second-order RSA는 서로 다른 데이터셋의 유사성 공간을 비교하는 방식이다(Kriegeskorte & Kievit, 2013). 이렇게 유사성 행렬을 구성하면, 데이터의 형식에 구애받지 않고 다양한 비교가 가능해진다.

## Procedure

RSA의 절차는 다음과 같다. 먼저, experimental condition별 brain activity pattern을 vector로 변환하고, 각 vector 간의 **distance (e.g., Euclidean, angular)**를 계산한다. 이렇게 계산된 distance 값들은 representational dissimilarity matrix (RDM)에 저장된다. 각 RDM의 개별 값들은 specific condition 간 neural pattern 차이를 수치화한 값으로, 공간적인 배열이나 측정 단위와 무관하게 중립적인 성격을 갖는다. 이후, 다른 RDM 간 second-order comparison을 수행하여 데이터 간 structural relationship을 분석할 수 있다.

예를 들어, human inferotemporal cortex (IT) 의 fMRI data 로 생성한 RDM 과 monkey IT neurons 의 spike train data 로 생성한 RDM 을 비교할 수 있다. (Figure 2) 연구에 따르면, 두 RDM 은 animal stimuli 에 대해 매우 유사한 representational geometry 를 나타내며, 이는 인간과 원숭이가 IT region 에서 animal image 를 유사한 방식으로 처리하고 있음을 시사한다(Kiani et al., 2007; Kriegeskorte et al., 2008a, b).

<figure class='align-center'>
    <img src = "/images/2025-03-18-Representational similarity analysis in neuroimaging proxy vehicles and provisional representations/figure2.jpg" alt="">
    <figcaption>figure 2. Comparing RDMs through second-order RSA can relate representational geometries across very different methods, whereas direct relationships can be difficult to establish</figcaption>
</figure>


하지만, RSA 에서 사용하는 "representation" 이라는 개념은 수학적 의미와 cognitive science 적 의미가 반드시 일치하는 것은 아니다. Neuroimaging 에서 "representational space" 와 "representational geometry" 라는 용어는 mathematical metric distance 를 기반으로 한 기술적 개념이다. 반면, cognitive science 나 philosophy of mind 에서의 "representation" 개념은 보다 심층적인 의미를 가지며, 뇌가 특정 정보를 어떻게 처리하고 사용하는지에 대한 논의와 연결된다.

<br>

# What is representation?

mental representation의 본질과 그것이 뇌에서 어떻게 구현될 수 있는지에 대한 문제는 오랫동안 고민되어 왔다. 현재 cognitive function 을 이해하는 데 가장 유망한 이론적 틀은 computationalism 이며, 이는 복잡한 physical system이 어떻게 복잡한 tasks를 수행할 수 있는지를 설명하는 핵심적인 개념이다. Computationalism은 cognition을 computation의 한 형태로 간주하며, 이 과정이 representation 위에서 수행된다고 본다.

어떤 neural function에 대한 computational explanation을 제공하려면, 먼저 그 representation들의 구조와 내용(interpretation function)을 설명해야 한다. 또한, 해당 representations이 어떻게 physically realized되는지와 이들 위에서 작용하는 causal processes가 어떤 방식으로 적절한 output을 생성하는지도 밝혀야 한다(realization function) (Egan, 2010, 2018).

Computational explanation은 몇 가지 중요한 전제를 포함한다. **첫째, representational vehicle과 representational content를 구분해야 한다.** Representational vehicles은 물리적인 구조나 state로서 특정한 representational content를 운반하는 역할을 한다(Egan, 2018). Computational system내에서 causal processes는 이러한 representational vehicles에 작용하여 physical transformation을 유발하며, 그 결과 representational content도 변화한다. 즉, formal properties와 semantic properties사이의 관계가 intelligent behavior를 가능하게 한다.

또한, brain representation의 후보가 되기 위해 충족해야 할 여러 조건(desiderata)이 있다(Egan, 2020; Ramsey, 2007).

- Function: Representation은 시스템 내에서 특정 기능을 수행해야 한다. Clark(1996)의 minimal representationalism에 따르면, processing system이 특정한 inner states 또는 processes를 **외부 세계나 신체 내부 상태에 대한 정보를 운반하는 역할**을 한다고 묘사할 수 있을 때, 이를 **representational한 과정**으로 간주할 수 있다. 따라서 representation은 단일한 요소가 아니라, system of representations안에서 기능해야 하며, 이를 통해 intelligent behavior를 설명할 수 있어야 한다.

- Usage and Interpretation: Representations은 시스템 내에서 사용되거나 해석되어야 한다. 즉, 단순히 존재하는 것이 아니라, 실제 cognitive processing에서 소비(consume)되고 활용되어야 한다.

- Naturalizability: Content ascription(즉, **어떤 representation이 특정한 의미를 지닌다고 판단**하는 것)은 물리적 과정과 물질 내에서 설명 가능해야 한다. 이는 realization function과 interpretation function 모두 physical matter와 processes로 설명될 수 있어야 함을 의미한다.

- Possibility of Misrepresentation: Representation개념은 필연적으로 misrepresentation을 포함해야 한다. 즉, 어떤 representational state가 actual world와 불일치하는 경우가 발생할 수 있어야 한다. Norms for content ascription(즉, 특정 representation이 올바른 의미를 지니는지에 대한 규칙)이 필요하며, 이러한 규범은 시스템이 설정한 기능적 목표를 달성하지 못할 때 위반된다고 볼 수 있다.

## From mental content to neural representation: a proposal

모든 정신의 철학적 이론은 뇌는 개체가 환경을 추적하고 적응적으로 반응할 수 있도록 해야 한다는 개념을 기반으로 한다. 철학자들은 이러한 tracking relation을 설명하기 위해 여러 이론적 접근을 시도해왔다. 주요한 이론으로는 causal or information-theoretic theories, teleological theories, 그리고 structural theories가 있다. Causal or information-theoretic theories는 representation의 내용을 **causal relation**을 통해 정립하며, teleological theories는 **natural function 또는 evolutionary history**에 기반하여 내용을 부여한다. 반면, structural theories는 **isomorphism이나 similarity**가 referent와 representation 간의 관계를 형성하는 방식에 초점을 둔다.

이러한 각 이론에는 한계점이 존재하지만, 일부 연구자들은 **표상의 내용(content)**을 설명하는 단일한 이론이 반드시 필요하지 않다고 주장하며, 다른 이들은 **과학적 연구(scientific practice)**에서 **내용 귀속(content ascription)**이 가지는 설명적(explanatory) 혹은 실용적(pragmatic) 역할을 강조한다(Egan, 2020; Godfrey-Smith, 2006; Shea, 2018; Shea et al., 2018). **Shea(2013)**는 표상이 중요한 이유는 그것이 **시스템과 환경(system and environment)**을 연결하는 역할을 하기 때문이라고 주장한다. 즉, 표상은 객관적 대상(objective objects) 및 **속성(properties)**과 상호작용하며, 문제 공간(problem space) 속에서 특정한 기능을 수행하는지에 대한 설명이 가능해야 한다.

비록 mental content theories가 완전히 정립된 것은 아니지만, brain activity가 external stimuli와 causal relation을 형성하며, 이러한 과정이 초기 감각적 표상(perceptual representation)이 보다 정교한 형태의 representation으로 변환되는 과정에 기여한다는 점은 명확하다. 최근 연구들은 neural representation을 효과적으로 기술하는 방법을 발전시키고 있으며, 그중에서도 high-dimensional state-space를 이용한 접근 방식이 주목받고 있다.

**high-dimensional state-space model**에서는 **뇌**가 높은 차원의 공간에서 다양한 요소를 조합하여 표상할(represent) 수 있는 능력을 갖추고 있다고 본다. 일부 차원은 **피질의 공간적 구조(spatial topography of the cortex)**에 직접적으로 반영되며, 예를 들어 **체감각 피질(somatotopic representation in sensory cortex)**이나 **시각 피질(visuotopic layout in early visual cortex)**에서 이러한 구조를 볼 수 있다. 또한, **방향 선택성(orientation-selectivity)**과 같은 속성은 피질 기둥(cortical columns) 내부에 반복적으로 인코딩되며, 일부 정보는 순전히 기능적으로(encoding purely functionally) 저장되기도 한다. 단일 뉴런 기록(single-unit recording) 연구에서는 **표상적 특성(representational properties)**이 피질 내에서 위계적으로(hierarchically organized) 구성된다는 점을 보여주었으며, 하위 감각 및 운동 피질은 단순한 속성을 **표상(representing)**하고, 상위 피질 영역에서는 보다 **복잡한 의미적 속성(complex semantic properties)**이 형성된다는 점을 시사한다.

이러한 고차원 공간(high-dimensional space) 개념이 중요한 이유는 **신경 내용**과 **신경영상 분석**에 모두 적용될 수 있기 때문이다(Haxby et al., 2014). 특정 **문제 영역(problem domain)**에서 특징(features)을 정의하고, 이를 바탕으로 **의미 공간(semantic space)**을 형성할 수 있다. 예를 들어, **사물(object representation)**을 정의할 때, 크기(size), 색깔(color), 형태(shape features) 등의 차원이 존재할 수 있으며, 각 사물은 이 공간에서 특정한 **점(point)**으로 표현된다.

이 개념을 **신경 표상(neural representation)**으로 확장하면, **뉴런 활동(neuronal processing)**이 인지(cognition) 및 **행동(behavior)**을 어떻게 형성하는지 이해할 수 있다. **신경 표상 공간(neural representational space)**을 정의할 때, 특정 뇌 영역(brain region) 내에서 각 뉴런을 공간의 개별 **축(axis)**으로 설정할 수 있으며, 특정 시간에서 모든 **뉴런(neurons)**의 활성화 패턴은 해당 공간 내의 특정한 **위치(point)**로 나타난다. 예를 들어, 특정 뇌 영역이 **시각 정보 처리(visual processing)**를 담당한다고 가정하면, 해당 **뉴런 집단(neural population)**의 반응은 **자극(stimuli)**의 변화에 따라 표상 공간(representational space) 내에서 이동하는 **활성 벡터(activity vector)**로 나타날 것이다.

이 개념은 **계산 이론(computation theory)**과도 연결된다. 특정 뇌 영역*에서 **신경 표상(neural representation)**을 이해한다는 것은, 해당 영역에서 표현되는 **의미 공간(semantic space)**의 차원을 분석하는 것이며(즉, **해석 함수(interpretation function)**에 해당), **신경 표상 공간(neural representational space)**과 의미 공간(semantic space) 간의 매핑(mapping)을 규명하는 것이다(즉, **실현 함수(realization function)**에 해당).

fMRI는 뇌 활동을 측정할 수 있는 유용한 도구이지만, 개별 **뉴런(neuron)**의 활성화 데이터를 제공하지는 않는다. 대신 특정한 뇌 조직(brain tissue) 내에서 집계된 총체적 활동을 측정할 수는 있다. 이를 고려하여, 각 voxel의 활동을 하나의 **축(axis)**으로 설정하는 다차원 공간(multidimensional space)을 구성할 수 있다. 물론, 이러한 **공간(space)**의 차원 수는 실제 **신경 표상 공간(neural representational space)**보다 훨씬 적으며, 여러 개의 차원이 축소될 것이다. 예를 들어, Kamitani와 Tong(Kamitani & Tong, 2005; Tong et al., 2012)의 연구에서는 fMRI를 활용하여 **자극의 방향(stimulus orientation)**을 성공적으로 해독할 수 있었다. 비록 개별 voxel이 **피질 기둥(cortical columns)**보다 훨씬 큰 단위임에도 불구하고, **피질 내 국소적 비등방성(local anisotropies in the cortex)**에 의해 특정한 정보가 추출될 수 있었기 때문이다.

<br>

# What does RSA tell us about neural representation?

## Things that RSA tells about neural representation

**표상(Representation)**을 뇌에서 직접적으로 식별할 수 있는가? 이에 대한 답은 부분적으로 그렇다는 것이다. RSA는 뇌의 특정 신경 활동 패턴이 **표상의 내용(content)**을 반영하는지 분석하는 도구로 사용될 수 있다. 기본적으로 RSA는 표상의 매개(vehicle)와 내용(content) 간의 구분을 고려하는데, voxel 단위의 활동 패턴이 매개(vehicle) 역할을 하며, **내용**은 신경 활동의 **외부 원인과의 관계 및 해당 패턴이 비교 대상(RSA의 타겟)과 얼마나 공변하는지**를 통해 밝혀진다. 이 과정에서 **표상 기하(representational geometry)**의 **동형성/준동형성(isomorphism/homomorphism)**이 주요한 비교 지표가 된다. 즉, 신경 패턴이 비교 대상과 구조적으로 유사할수록, 그 패턴이 표상적 의미를 가질 가능성이 커진다.

하지만 현실주의적(realist) 관점에서 보면, voxel의 활성화 값 자체는 **표상의 적절한 매개체(proper vehicle of content)**가 아니다. 왜냐하면 신경 표상의 매개는 **인과적 작용(causal powers)**을 가져야 하지만, voxel 값 자체는 그러한 기능을 수행하지 않기 때문이다. 실제 표상의 매개체는 특정한 **신경 아집단(neural subpopulations)**이며, 이는 voxel 수준에서 관찰되는 활성화 패턴의 **비등방성(anisotropy)**을 초래하는 요인이다. 따라서 fMRI는 표상의 매개체를 직접 측정하는 것이 아니라, 그 존재를 추론하는 방식으로 접근해야 한다. 결국 voxel 단위의 활성화 값은 **표상의 매개에 대한 대리**일 뿐이며, 다른 신경 측정 방법을 통해 보다 정확하게 식별될 필요가 있다.

그럼에도 불구하고 RSA는 표상의 내용(content)에 대한 유용한 정보를 제공할 수 있다. 왜냐하면 신경 활동 패턴이 **구조적 관계(structural relationships)**를 반영할 경우, 그 구조 자체가 표상의 내용을 포함할 가능성이 높기 때문이다. RSA는 대뇌 피질의 거시적 수준에서 표상의 구조가 어떻게 조직되어 있는지를 밝히고, 동일한 신경 조직 내에서 **여러 표상 공간(representational spaces)**을 구분해낼 수 있다. 이는 **고차원 표상 이론(hyperdimensional theories of representation)**과 일치하는 결과이다.

특히 RSA의 결과가 단일 뉴런 기록(single-unit recording) 데이터와 일치하는 경우, RSA를 통해 밝혀진 표상의 신뢰도가 높아진다. 예를 들어, 원숭이(monkey)와 인간(human)의 하측두 피질(inferotemporal cortex, IT)에서 RSA를 수행한 연구에서, 다양한 동물 자극(animals)에 대한 **유사성 행렬(similarity matrix)**이 원숭이 IT에서의 개별 뉴런 발화율(firing rates)을 기반으로 계산된 유사성 행렬과 높은 동형성을 보였다(Kriegeskorte et al., 2008a, b). 이는 신경 표상의 대규모 조직이 개별 뉴런의 미세한 수용장(receptive field properties)을 반영할 가능성이 있음을 시사한다.

이 연구는 또한 **인간과 원숭이가 사물의 범주적 구조(categorical structure of objects)**를 유사한 방식으로 표상한다는 점을 뒷받침한다. 우리가 이미 초기 시각 시스템(early visual systems)이 신경 기능 측면에서 **상동적(homologous)**이라는 독립적인 근거를 가지고 있기 때문에, IT에서 발견된 이러한 구조적 유사성은 해당 영역에서의 표상의 매개(vehicles for object representation) 또한 유사할 것이라는 강한 추론적 근거를 제공한다.

## How fMRI signals can be proxy vehicle of neural representation?

fMRI의 신호는 **혈류 변화**를 기반으로 측정되며, 이는 개별 뉴런의 발화율(firing rates) 또는 **특정 신경 아집단(selective neuronal populations)**의 활동과 비교했을 때 훨씬 더 거친(grainy) 단위이다. 그럼에도 불구하고, RSA는 혈류 신호와 실제 신경 표상 사이의 유사성 관계를 밝혀낼 수 있다. 이는 뇌가 다차원 공간에서(topographical multidimensional space) 정보를 조직하고 처리한다는 가설과 일맥상통한다. 즉, 유사한 정보는 유사한 방식으로 처리된다는 원리가 존재한다면, fMRI 신호도 특정한 **표상적 구조(representational structure)**를 반영할 수 있다.

이를 RSA를 활용해 검증할 수 있는 한 가지 방법은 **신경 정보 처리의 변환(transformation of neural processing)**을 추적하는 것이다. 예를 들어, 어떤 자극의 특정한 특징(feature)이 초기에 신경 반응의 차이를 유발하지만, 이후 처리 단계에서는 변함없이 유지된다면 이는 해당 특징이 **표상적으로 불변(invariant representation)**하게 처리된다는 증거가 된다.

예를 들어, **초기 시각 피질(early visual cortex)**에서는 얼굴(face) 자극이 **표상적 군집(representational cluster)**을 형성하지 않지만, **하측두 피질(IT cortex)**과 같은 **고차원 시각 처리 영역(higher-level visual regions)**에서는 얼굴 자극이 하나의 독립적인 **유사성 군집(similarity cluster)**을 형성한다. 이후 단계에서는 개별 얼굴 간의 유사성 값이 **얼굴 방향(face orientation)이 달라지더라도 변하지 않는 패턴을 보인다(Guntupalli et al., 2017). 이는 IT에서 개별 얼굴의 정체성(identity)이 표상되고 있다는 것을 시사한다.

## Studies on misrepresentions with RSA

신경 표상의 내용(content)은 종종 **자극(stimulus)과의 인과적 관계(causal relation)**로 정의될 수 있지만, 항상 그렇지는 않다. 신경 표상의 내용은 **기능적 역할(functional role)**을 기준으로 정의될 수도 있다. 특히 RSA를 통해 표상의 왜곡(misrepresentation)을 분석할 수 있다.

예를 들어, 하측두 피질 IT에 "포식성(predacity) 정도"에 따라 점진적인 신경 활성 경향이 존재한다면(Connolly et al., 2012), 특정한 자극(예: 쥐)이 높은 포식성을 가진 영역에서 활성화를 유발할 경우, 해당 피험자는 쥐를 "위험한 동물"로 오인(misrepresentation)한 것일 수 있다. 만약 이러한 표상이 피험자의 행동과 불일치하거나(agent’s goals와 대립되거나), 잘못된 판단을 유도한다면, 이는 RSA를 통해 신경 표상의 오류를 탐지할 수 있음을 의미한다(Isaac, 2013).

결국 RSA는 표상의 내용을 직접적으로 측정하는 것이 아니라, 표상의 매개(vehicles)의 대리(proxy vehicle)를 분석하는 기법이다. 이 분석을 통해 대뇌 피질이 의미적 특징(semantic features)을 지도처럼(map-like) 표상할 가능성을 발견할 수 있으며, 이러한 패턴을 비교하여 **신경 정보 처리 과정(neural computational transformations)**을 추론할 수 있다.

흥미롭게도, RSA를 통해 밝혀진 **신경 표상의 구조(representational structure)**는 기존의 **디지털적 표상(digital representation)**이 아니라 **아날로그적 표상(analog representation)**에 가깝다는 점을 시사한다. 이는 Fodor와 같은 고전적 계산주의자들이 주장한 임의적 기호(Arbitrary symbolic representation) 모델과는 상반되는 결과이다. 대신, RSA의 결과는 **대뇌 피질에서 표상이 체계적으로 구조화되어 있으며, 표상 기하(representational geometry)**를 통해 표상의 내용을 유추할 수 있음을 보여준다.

<br>

# Provisional representations: What doesn’t RSA tell us

RSA를 통해 밝혀진 **신경 표상(Neural Representations)**은 우리가 일반적으로 이해하는 **정신적 표상(Mental Representations)**과 유사한 특징을 가지지만, 여전히 몇 가지 중요한 차이점이 존재한다. 이를 **임시적 표상(Provisional Representations)**이라고 부를 수 있다. 즉, RSA는 특정한 정보가 신경 활동 패턴 내에 존재함을 보여주지만, 이 정보가 실제로 어떤 신경 아집단(neural subpopulations)에 의해 운반되는지, 그리고 더 세밀한 해상도에서 어떤 추가 정보를 포함하고 있는지는 밝혀주지 못한다.

우선, RSA는 특정 뇌 영역이 표상의 내용을 담고 있음을 시사하지만, 정확히 어떤 신경 집단이 이 표상을 매개하는지는 알 수 없다. fMRI 등에서 측정의 단위가 되는 Voxel 단위의 신호는 연구자가 실험 설계 과정에서 구성한 측정 단위일 뿐, 실제 신경 표상의 매개가 아니다. 또한 우리는 특정 신경 아집단이 유사성 행렬(similarity matrices)에서 분석된 표상 내용을 담고 있다는 것을 알고 있지만, 해당 신경 집단이 그 외에도 어떤 정보를 포함하고 있는지는 알지 못한다. 또한 RSA를 통해 추론된 표상의 매개(vehicle)는 **대리적(proxy)**일 뿐이며, 실제 신경 표상을 개별 뉴런 수준에서 연구할 필요가 있다.

철학자 Egan의 주장처럼, "표상의 매개(vehicle)가 없다면, 표상도 없다(No vehicles, no representations)"는 입장을 따른다면, RSA의 한계는 더욱 분명해진다. 그러나 Egan도 표상의 매개가 반드시 직접 측정되어야 하는 것은 아니며, 추론될 수도 있다고 인정한다. 따라서, RSA는 최소한 특정 정보가 뇌 영역에서 처리됨을 보여주는 "가정적 매개(presumptive or proxy vehicles)"를 제공할 수 있다.

## RSA는 정보가 실제로 사용되는지를 보여주지 못한다

또한, RSA는 특정 신경 활동 패턴이 특정 구조적 정보를 포함하고 있음을 밝혀내지만, 이 정보가 실제로 뇌의 후속 과정(downstream processes)에 의해 사용되는지 여부는 알 수 없다. 다시 말해, RSA는 해당 정보가 **존재한다는 것(Existence Proof)**은 제공하지만, 이 정보가 기능적으로 활용된다는 **증거(Evidence of Use)**는 제공하지 못한다.
만약 후속 정보 처리 과정이 이 정보를 활용하지 않는다면, 이는 행동(behavior)과는 무관한 후속 결과(epiphenomenal effect)에 불과할 가능성이 있다.
즉 RSA를 통해 어떤 뇌 영역이 특정 정보를 표상한다고 해도, 실제로 그 정보가 다른 신경 과정에서 활용되거나 행동에 영향을 미친다는 보장은 할 수 없기 때문에 이를 검증하려면 행동과의 관계를 명확히 규명하는 추가 연구가 필요하다.

이러한 한계를 극복하기 위해 RSA 결과를 행동 데이터(behavioral data)와 연결하는 연구가 진행되고 있다. 예를 들어, Charest et al.(2014)의 연구에서는 **하측두 피질(IT)**에서 RSA를 수행하여 참가자들이 익숙한(familiar) vs. 낯선(unfamiliar) 사물을 볼 때의 신경 패턴을 비교했다. 연구 결과, 참가자들 간에 **표상 기하(representational geometry)**의 유사성이 높은 것으로 나타났다. 더 흥미로운 점은, 개인 간 신경 표상의 차이가 해당 개인의 유사성 판단(similarity judgments) 차이를 예측했다는 점이다.

연구자들은 이러한 결과가 각 참가자의 개별적인 경험(idiosyncratic experience)이 신경 표상과 행동 모두에 영향을 미쳤기 때문이라고 해석했다.
비록 인과 관계(causal relationship)가 직접적으로 증명된 것은 아니지만, 이러한 연구들은 신경 표상과 행동 간의 밀접한 연관성을 시사하며, 실험적 조작을 통해 추가적인 인과적 분석이 가능할 수 있음을 보여준다.

## RSA의 한계와 인과 관계(Causal Inference) 문제

fMRI 및 RSA의 가장 큰 한계 중 하나는, 상관관계(correlation)만 제공할 뿐 인과관계(causality)를 증명할 수 없다는 점인데, 다시 말해서 특정한 뇌 영역이 특정 자극에 반응하는 것(**"가능한 작동 방식(how possibly)"**)을 보여주지만, 이 활성화가 실제로 정보 처리를 유발(**"실제로 작동하는 방식(how actually)"**)하는지는 알 수 없다. 이 문제를 해결하려면, 신경 조작(neuromodulation) 기법(예: TMS, 전기 자극 등)을 활용하여 특정 신경 영역의 활성화를 조작하고 그에 따른 행동 변화를 분석하는 실험이 필요하다.

## CNN(Convolutional Neural Networks)과 RSA의 유사성

흥미롭게도, RSA의 다차원적 분석 방식은 심층 신경망(Deep Neural Networks, DNNs)과 Convolutional Neural Networks (CNNs)의 분석과 유사한 방식으로 활용될 수 있다. CNN이나 딥러닝 모델에서 인지 과제(cognitive tasks)를 수행하도록 훈련된 신경망의 표상 구조는 fMRI에서 추론된 표상 기하(representational geometry)와 유사한 패턴을 보인다(Khaligh-Razavi et al., 2017). 이러한 결과는 딥러닝 모델에서 표상이 유효한 증거로 간주된다면, RSA를 통해 밝혀진 신경 표상도 유효한 증거로 간주될 수 있음을 시사한다(Poldrack, Yamins et al., 2014; Yildirim et al., 2019). CNN에서는 신경망을 조작하여 특정 뉴런이 인지 과정에서 어떻게 작동하는지를 직접 실험할 수 있기 때문에, CNN에서의 인과적 분석이 RSA를 통한 신경 표상 연구에도 적용될 수 있는 가능성이 있다.

<br>

# Representations in psychology

## 1. RSA의 한계와 비교의 필요성

이전 섹션들에서 다루었듯이, **표상 유사성 분석(Representational Similarity Analysis, RSA)**이 fMRI 데이터에서 드러나는 패턴들이 신경 표상이라고 주장할 수 있는 근거를 어느 정도 제공하지만, 모든 철학적 또는 과학적 기준을 충족하지는 못한다. 특히 voxel 단위의 fMRI 데이터는 표상의 실제 vehicle이 아니며, 어떤 신경 집단이 실제로 그 내용을 전달하고 있는지도 확실히 알 수 없다. 저자는 이러한 한계를 이유로 fMRI 기반 RSA가 표상적 지위를 가질 수 없다고 단정짓기보다는, 심리학에서 이미 표상 개념을 당연하게 전제하고 있는 다른 사례(비슷한 수준의 간접성을 가진 심리학적 방법들—특히 반응 시간 기반 실험들—도 여전히 널리 수용되고 있다)들과 비교해 볼 필요가 있다고 제안한다.

## Shepard & Metzler의 정신적 회전 실험 사례

Shepard와 Metzler는 1971년 수행한 고전적 실험에서 사람들이 시각 자극을 인지하고 판단하는 과정에 시간적 패턴이 어떻게 나타나는지를 분석했다. 이 실험에서 참가자들은 서로 다른 두 개의 2차원 이미지가 동일한 3차원 물체의 회전된 버전인지, 아니면 대칭 반전된 형태인지를 판단해야 했다. 중요한 결과는, 두 이미지 간 회전 각도가 커질수록 반응 시간이 선형적으로 증가했다는 것이다. 이 결과는 사람들이 단순히 모양의 특징을 비교하는 것이 아니라, 머릿속에서 하나의 물체를 회전시키는 **정신적 조작(mental transformation)**을 수행하고 있음을 시사했다. 또한, 이 회전이 시각 평면 내의 단순 회전이든, 깊이 방향으로의 3차원 회전이든 관계없이 반응 시간의 패턴은 유사하게 나타났다.

이러한 반응 시간 패턴은 연구자들에게 참가자가 실제로 **3차원 형태의 심상(imagery)**을 마음속에서 구성하고 회전시키고 있다는 가정을 정당화하게 만들었고, 이후 이 결과는 이미지적(analogical)이고 공간적인(spatial) **정신 표상(mental representation)**의 존재에 대한 증거로 해석되었다. Shepard는 이러한 표상과 실제 물체 간의 관계를 **2차 등형성(second-order isomorphism)**이라고 표현했으며, 이는 두 시스템 간 구조적 유사성이 있다는 의미다.

비록 이 해석은 여러 학자들로부터 비판을 받기도 했지만, 그 비판들은 주로 정신 표상의 성질에 관한 과학적 논의의 일환이었다. 예를 들어, Pylyshyn은 이러한 심상이 실제로 존재하는지보다는, 그것이 기호 기반 기계적 처리로 설명될 수 있는지를 문제 삼았고, Carpenter와 Just는 회전 과정의 특성에 주목해 보다 정밀한 실험 설계를 제안했다. 그러나 이러한 논의는 실험 자체의 과학적 정당성이나 표상 개념의 부적절성을 문제 삼기보다는, 표상의 특성을 더 정확히 밝히기 위한 시도로 진행되었다. 실제로 이후의 많은 연구들도 여전히 반응 시간을 주요 척도로 사용하는 방식으로 진행되었다.

흥미로운 점은, Shepard의 초기 논문에서는 ‘representation’이라는 용어 자체는 명시적으로 사용되지 않았음에도 불구하고, 그 결과는 명백히 표상 개념을 전제로 한 현상들—예컨대 마음속에서 3차원 물체를 상상하고 회전시키는 과정—을 설명하기 위해 해석되었다는 것이다. 이는 후속 연구들과 학계 해석을 통해 점차 심리학적 표상에 대한 강력한 간접 증거로 자리잡게 되었음을 보여준다.

## 표상의 존재와 그것의 증거로서 어떤 자료들이 어떤 방식으로 활용되는가

먼저, Shepard와 Metzler의 정신 회전 실험에서는 **표상의 수단(vehicle)**에 대한 직접적인 접근은 없다는 점을 지적한다.  실험 참가자가 3차원 물체를 마음속에서 회전시키는 것처럼 보이는 반응 시간 패턴을 통해 **3차원 심상(mental imagery)**의 존재를 *추론(abduction)*할 수는 있지만, 이 심상이 실제로 어떤 신경적 기제로 실현되는지는 전혀 알 수 없다. 이러한 반응 시간 기반 분석은 오히려 fMRI의 RSA보다 더 간접적인 방식으로 표상 개념을 다루는 셈이다.

이에 비해 RSA는 fMRI 데이터를 기반으로 하여 신경 활동 패턴 간의 구조적 유사성을 분석하고, 이를 행동적 유사성과 비교함으로써 표상의 구조적 근거를 제공한다. 이 방식도 여전히 상관관계(correlation)에 기반을 두고 있지만, 적어도 fMRI 데이터는 특정 뇌 영역과의 관련성을 통해 표상의 가능성 있는 신경적 위치를 제시할 수 있다는 점에서 Shepard의 방법보다 한 걸음 더 나아간다. 다시 말해, Shepard의 방법은 단지 존재 가능성만을 시사하지만, RSA는 그 표상이 어떤 신경 구조 안에 실현되고 있을 가능성까지 암시한다.

그러나 중요한 차이점은, Shepard의 경우 추론된 표상이 행동의 원인으로서 인과적으로 연결되어 있다는 점이다. **자극 간 회전 각도가 커질수록 반응 시간이 길어지는 결과는 명확히 정신적 조작이 행동 반응을 유발한다는 인과성을 내포하고 있다.** 반면, 대부분의 RSA 연구는 뇌 내 표상 구조와 행동 간 인과성을 직접적으로 연결 짓지 않으며, 단지 표상의 구조적 유사성만을 다룬다. 물론 일부 RSA 연구에서는 뇌 반응 패턴과 행동 반응(예: 유사성 판단)이 밀접하게 상관된다는 것을 보여주기도 하지만, 이는 모든 RSA 연구의 일반적 방식은 아니다. 그러므로 RSA 연구에서 **행동과의 직접적은 연계가 없더라도 RSA에서 나타나는 구조적 유사성은 귀납적이지만 정당화 가능한 방식으로 표상의 존재를 추론하게 해주는 증거로 간주할 수있기 때문에** 행동 지표와 geometry간의 연계성에 집중을 해야 한다고 결론을 짓고 있다.

<br>

# Conclusion

Egan(2018)의 주장에 따르면, 표상 개념은 단지 이론적 설명이 아니라 발견의 도구로서 유용하다. 저자는 RSA가 바로 이러한 역할을 한다고 본다. 예를 들어, 인간 시각 피질에는 얼굴 자극에 선택적으로 반응하는 여러 영역이 있으며, 그중에서도 FFA(Fusiform Face Area)는 손상 시 안면 실인증(prosopagnosia)을 유발하거나, 전기 자극 시 얼굴 지각의 변화만을 유도하는 등 얼굴 표상 처리에 인과적으로 관련됨이 알려져 있다.

RSA는 이처럼 뇌 피질의 특정 수준에서 **시야 불변성(view invariance)**이 어떻게 계산되는지를 밝혀냈고, 이는 표상 처리의 계층 구조와 전환 과정에 대한 가설 설정에 기여했다. 또한 RSA는 **원숭이의 피질 영역과의 상동성(homology)**을 밝히는 데도 활용되어, 종간 비교를 통한 정밀한 후속 탐색 연구의 기반을 마련해 준다.

결론적으로 RSA는 후속 세부 신경과학 연구에 적합한 가설 및 대상 영역을 도출하는 데 강력한 도구다. 다만 RSA만으로는 특정 **표상의 수단(vehicle)**이나 **내용(content)**을 확정할 수 없으며, 그 확인은 반드시 보다 미시적인 수준의 연구 기법을 통해 이루어져야 한다. fMRI 기반 분석은 행동 관련 정보를 매우 정교하게 제공할 수 있는 수단이지만, 그 자체로 해석되기보다는 다른 기법과 통합적으로 활용되어야 하며, 후속 연구를 유도하는 길잡이 역할을 한다.

