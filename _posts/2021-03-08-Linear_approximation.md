---
title: "선형근사 (Linear approximation)"

classes: wide

categories:
  - Calculus1
tags:
  - Linear Approximation
---
# 선형 근사

"**함수의 그래프와 그 그래프의 접선은 인접한 부분에서 그래프가 유사하다.**"

의 아이디어 사용

---

**ex) $$ (1.999)^4 $$**

우선 $$y = x^4$$ 의 그래프와 그 위의 점 $$(2,16)$$에서의 접선 $$y = 32x - 48$$의 그래프를 그린다.

$$ y = x^4 $$을 $$f(x)$$,  $$ y = 32x - 48$$을 $$L(x)$$라고 하자.

![linearapproximation_1](https://user-images.githubusercontent.com/72269271/110303746-1abf3b80-803e-11eb-8c03-1a6f83c92765.JPG)

위에서 사용한 아이디어를 통해 $$f(1.999)$$ 와 $$L(1.999)$$는 유사한 값을 갖는다.

실제로 계산해보면

 $$f(1.999) = (1.999)^4 = 15.968023$$,


 $$L(1.999) = 32(1.999)-48 = 15.968$$ 이다.

---
이렇게 1차(선형)근사를 통해 근사값을 찾는 방법을 **선형근사**라고 한다.

정리해보면 선형근사를 통해 근사값을 찾아가는 과정은

1. _주어진 수_(복잡한 수)와 가장 인접한 _편리한 수_ (계산이 쉬운 수, 주로 정수)를 택한다. 
> 예제에서는 **편리한 수** 로 2를 택함. 

2. _주어진 수_ 를 미지수$$x$$ 로 두고 함수로 만든다.
> 예제에서 1.999 -> $$x$$ , 만들어진 함수 : $$ f(x) = x^4$$

3. 위에서 정한 **편리한 수** 를 $$x$$ 값으로 갖는 함수 $$f(x)$$위의 점을 찾는다.
> 예제에서 $$(2,16)$$이 함수 위의 점.

4. 선택된 좌표에서 접선의 방정식을 찾는다. $$ L(x) = f'(a)(x-a)+f(a)$$ (여기서 a는 **편리한 수**)
> 예제에서 $$L(x) = 4(2)^3(x-2)+2^4 = 16x - 48$$