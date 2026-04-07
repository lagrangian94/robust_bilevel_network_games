# Network Interdiction Game with Private Uncertainty Model

seokwookim July 2024

## 1 Problem Description

- • Maximum flow interdiction in a directed network G = (V,A)
- • Leader decision: interdiction xij, ∀(i,j) ∈ A

1. xij = 1 if leader destroys arc (i,j) and 0 otherwise interdiction budget: x ∈ x ∈ {0,1}|A| : 1⊺x ≤ γ

- • Follower solves a 2-stage stochastic maximum flow problem

- 1. Here-and-now h: arc capacity recovery recovery budget: h ∈ h ∈ R|A|+ : 1⊺h ≤ w, h ≤ w(1 − x)
- 2. Wait-and-see y: flow on network
- 3. Uncertain parameter u(ξ): arc capacity, v(ξ): interdiction effect


- • Interdiction constraint (coupling): yij ≤ uij(ξ)(1 − vij(ξ)xij) + hij ∀(i,j) ∈ A


leader x ↷ follower h = h(x) ↷ uncertainty ξ ↷ follower y = y(x,h,ξ)

### 1.1 Robust Counterpart

Robust Counterpart를 유도하는 건 두 가지 방법이 있음. nominal distribution에서 시작해서 value function reformulation을 만든 뒤 그 formulation에 robust counterpart를 만드는 것과, 처음부터 minimax formulation 으로 시작하는 것. 둘이 같은 지 검토 필요하지만 그건 나중에.

먼저, leader가 자신의 distribution과 follower의 distribution을 알고 있다고 가정하면 아래와 같은 stochastic bilevel program이 나옴. Assumption 1 (Fixed Recourse). We assume the uncertainty is in the right-hand-side of the follower’s second stage constraints

d(ξ) = d, A(ξ) = A,bx(ξ) = T(ξ)x + b(ξ)

Eξ∼Pˆ [d⊺0yˆ(ξ)] (1a)

min

sup

x∈X

h,yˆ∈L2n

c⊺h¯ + Eξ∼P˜ Q(h,x¯ ) (1b) yˆ(ξ) ∈ arg max

s.t. h ∈ arg max

h¯∈H

{d⊺1y¯ : Ay¯ ≤ bx(ξ) − Bh} Pˆ − a.s. (1c)

y¯∈Rn

yˆ(ξ) ≥ 0 Pˆ − a.s., (1d) where Q(h,x¯ ) = max

{d⊺1y : Ay ≤ bx(ξ) Bh} (1e)

y∈Rn

- Remark 1. 여기서 decision chrognology는 leader x ↷ follower h(x) ↷ uncertainty ξ ↷ follower yˆ = ˆy(x,h,ξ)

여기서 follower는 distribution P˜(̸= Pˆ)일 수 있으므로, h를 결정할 때 follower가 계산한 optimal recourse는 yˆ(ξ)가 아닐 수 있음.

- Remark 2. network interdiction같은 zero-sum에선 자연스럽게 supyˆ를 하게 됨. 그런데 h에 관해선 optimistic 인지 pessimistic인지, 그 사이인지 바로 정해지지 않음. 여기선 zero-sum의 목적을 고려하여 h까지 pessimistic 으로 모델링해서 목적함수에 suph,yˆ를 넣어줌.

여기서 두 가지의 연관된 문제점일 발생: (i) arg max operator가 제약식에 존재함, (ii) Q(·,·)라는 value function arg max 제약식에 존재함. 따라서 이 둘을 차례대로 삭제하고 일반적인 최적화 모델로 유도해야 함.

follower가 uncertainty를 observe하기 전 계산한 recourse를 explicit하게 model에 꺼낼 수 있음. 또한 [3]의 Proposition 2.1.을 이용하면 arg max operator를 삭제하고 아래와 같은 equivalent reformulation을 유도할 수 있음.

min

x∈X

sup

h y,ˆ y˜∈L2n,

- Eξ∼Pˆ [d⊺0yˆ(ξ)] (2a)

s.t. c⊺h˜ + Eξ∼P˜ [d⊺1y˜(ξ)] ≥ z∗(x) := max h¯∈H

c⊺h¯ + Eξ∼P˜ Q(h,x¯ ) (···λ) (2b) Ay˜(ξ) + Bh˜ ≤ bx(ξ) P˜ − a.s. (···π˜(ξ)) (2c)

- Eξ∼Pˆ [d⊺1yˆ(ξ)] ≥ Q(h,x˜ ) := Eξ∼Pˆ Q(h,x˜ ) (2d) Ayˆ(ξ) + Bh˜ ≤ bx(ξ) Pˆ − a.s. (···πˆ(ξ)) (2e) h˜ ∈ H := h˜ : Wh˜ ≤ w (···ν) (2f) y˜(ξ),yˆ(ξ),h˜ ≥ 0 (2g)


Proof. [3]의 proposition 2.1을 그대로 쓰지 못하는 이유는 follower가 here-and-now decision h를 가지고 있기 때문. 내 생각엔 이걸 rockafellar의 SVI논문에서처럼 h 역시도 L2m space에 있는 square integrable function이되 non-anticipativity space에 있게 formulate 한뒤 [3]의 증명을 사용하면 유도가 될 수도 있을듯?

| |
|---|


- Remark 3. 문제는 Q가 x뿐만 아니라 h에도 parameterized 되어 있어서 z∗와 같이 바로 LP duality formula를 사용할 수 없음. 따라서 다른 방법을 간구해야 함. 그런데 우리 본래 문제는 network interdiction이라서, 사실

c = 0, d1 = d0이고 (2d) 제약식이 애초에 필요가 없음. 그래서 일반적인 2stage LP에 대한 유도는 나중에 작업하기로 하고 일단 이 점을 이용해서 dualize를 아래부터 진행함.

[3]의 approach를 따라서, inner pessimistic supremum 문제를 dualize하면 min

x∈X

inf

π,˜ πˆ∈L2n λ,ν

Eξ∼Pˆ [bx(ξ)⊺πˆ(ξ)] + Eξ∼P˜ [bx(ξ)⊺π˜(ξ)] λz∗(x) + wν (3a)

s.t. A⊺π˜(ξ) − d0λ ≥ 0 P˜ − a.s., (3b) A⊺πˆ(ξ) ≥ d0 Pˆ − a.s., (3c) EP˜[B⊺π˜(ξ)] + EPˆ [B⊺πˆ(ξ)] + W⊺ν ≥ 0 (3d) π˜(ξ) ≥ 0 P˜ a.s., (3e) πˆ(ξ) ≥ 0 Pˆ − a.s., (3f) λ,ν ≥ 0 (3g)

- Remark 4. 위 formulation에 대한 insight


- • (1b)가 없다면, 원래 문제는 nominal dist에 대한 일반적인 zero-sum network interdiction 모형임


- • 그런데 (1b)가 있음으로써, inner maximization의 feasible region이 축소되는 효과가 존재 (바꿔말해, follower의 상이한 belief를 무시한다면 overly conservative한 의사결정을 내린다는 말)


- 만약 (1b)가 없다면, 문제는

E[bx(ξ)⊺πˆ(ξ)]

min

inf

x

πˆ

s.t. A⊺πˆ(ξ) ≥ d0 Pˆ − a.s. EPˆ[B⊺πˆ(ξ)] + W⊺ν ≥ 0 πˆ(ξ) ≥ 0 Pˆ a.s.

• 따라서 원래 제약식의 EP˜[B⊺π˜] − cλ는 ≥ 0 제약식을 완화시켜주는 slack variable의 역할을 하고, 목적함 수의 Eξ∼P˜ [bx(ξ)⊺π˜(ξ)] − λz∗(x)는 그에 대한 trade-off로서 증가하는 비용으로 해석될 수 있음.

- λ → ∞로 가면 A⊺π˜(ξ) → ∞로 가고, coupling 제약식에도 −cλ → −∞로 가서 slack의 역할을 반대로

(제약식을 더 tight하게 만들어서 비용을 증가시키고 infeasible로 만듬) 이제 각 value function z∗,Q를 explicit하게 쓰면

Eξ∼Pˆ [bx(ξ)⊺πˆ(ξ)] + Eξ∼P˜ [bx(ξ)⊺π˜(ξ) − λd⊺0y˜(ξ)] + wν (4a)

min

inf

x∈X

π,˜ πˆ∈L2n λ,ν

s.t. A⊺π˜(ξ) d0λ ≥ 0 P˜ a.s., (4b) A⊺πˆ(ξ) ≥ d0 Pˆ − a.s., (4c) EP˜[B⊺π˜(ξ)] + EPˆ [B⊺πˆ(ξ)] + W⊺ν ≥ 0 (4d) Ay˜(ξ) + Bh ≤ bx(ξ) P˜ − a.s., (4e) Wh ≤ w, (4f) π˜(ξ) ≥ 0 P˜ − a.s., (4g) πˆ(ξ) ≥ 0 Pˆ − a.s., (4h) λ,ν ≥ 0,h ≥ 0 (4i)

by variable transform ˜y ← λy,h˜ ← λh, min

Eξ∼Pˆ [bx(ξ)⊺πˆ(ξ)] + Eξ∼P˜ [bx(ξ)⊺π˜(ξ) d⊺0y˜(ξ)] + wν (5a)

inf

x∈X

π,˜ πˆ∈L2n λ,ν

s.t. A⊺π˜(ξ) − d0λ ≥ 0 P˜ − a.s., (5b) A⊺πˆ(ξ) ≥ d0 Pˆ − a.s., (5c) EP˜[B⊺π˜(ξ)] + EPˆ [B⊺πˆ(ξ)] + W⊺ν ≥ 0 (5d) Ay˜(ξ) + Bh ≤ λbx(ξ) P˜ a.s., (5e) Wh ≤ λw, (5f) π˜(ξ) ≥ 0 P˜ − a.s., (5g) πˆ(ξ) ≥ 0 Pˆ − a.s., (5h) λ,ν ≥ 0,h ≥ 0 (5i)

여기서 network interdiction 구조를 가지고 위의 vector와 matrix를 정의하면 이렇게 나옴. 먼저, directed graph

G = (V,A)를 정의하고 edge (i,j),i ∈ V,j ∈ V들의 집합 A의 edge들에 적절한 indexing을 붙인다면

- V = {s,1,2,··· ,t}, where s,t denotes source, sink node respectively, A = set of directed arcs in graph G,

y := {yk}|A|k=1 yts

, h := {hk}|A|k=1 N := Ny|Nts ∈ Rm , where m denotes # of flow conservation constraints, A :=

N I0

, B :=

0 −I

, bx(ξ) :=

0 bx(ξ)

I0 := I|0 ∈ R|A| bx(ξ) := {ξk(1 − vkxk)}|A|k=1 ∈ R|A| π :=

π ϕ

c = 0 d0

0 ∈ R|A| 1

- W = e⊺ ∈ R1×|A|


   (zero for off diagonal, and v for diagonal entries) (vector of ones)

  

v1 0···

0 v2 0··· ··· ··· ···

V :=

0··· v|A|

그러면 network interdiction 문제에 DRO approach, taking worstcase expectation 를 취한 구체적인 formulation은 아래와 같다:

min

inf

t + wν (6a)

x∈X

π,˜ πˆ∈L2n λ,ν

A

A

Eξ∼Pˆ

Eξ∼P˜

ξk(1 − vkxk)ϕˆk(ξ) + sup P˜∈B

ξk(1 − vkxk)ϕ˜k(ξ) − y˜ts(ξ) ≤ t (6b)

s.t. sup

Pˆ∈B

k=1

k=1

Ny⊺π˜(ξ) + ϕ˜(ξ) ≥ 0 P˜ a.s., (6c) Nts⊺π˜(ξ) λ ≥ 0 P˜ − a.s., (6d) Ny⊺πˆ(ξ) + ϕˆ(ξ) ≥ 0 Pˆ − a.s., (6e) Nts⊺πˆ(ξ) ≥ 1 Pˆ − a.s., (6f) sup

EP˜[ϕ˜k(ξ)] + sup Pˆ∈B

EPˆ ϕˆk(ξ) − ν ≤ 0 ∀k = 1,··· ,|A| (6g)

P˜∈B

Nyy˜(ξ) + Ntsy˜ts(ξ) ≤ 0 P˜ − a.s., (6h) y˜k(ξ) − hk ≤ λξk(1 − vkxk) P˜ − a.s. ∀k = 1,··· ,|A|, (6i) e⊺h ≤ λw, (6j) π˜(ξ),ϕ˜(ξ) ≥ 0 P˜ a.s., (6k) πˆ(ξ),ϕˆ(ξ) ≥ 0 Pˆ − a.s., (6l) λ,ν ≥ 0 (6m)

where B = Bε is an ambiguity set (a ball equipped with wasserstein metric with radius ε)

### 1.2 Node-arc incidence matrix

그리고 noce-arc incidence matrix N은 다음과 같다. 예를 들어 node가 {s,1,2,t}만 존재하고 임의로 연결되어 있다고 가정하면





|Regular arcs (s,1) (s,2) (1,2) (1,t) (2,t)<br><br>|Dummy (t,s)|
|---|---|
|+1 +1 0 0 0 −1 0 +1 +1 0<br><br>0 −1 −1 0 +1 0 0 0 −1 −1<br><br>|−1 0 0 +1|


- ← s

- ← 1
- ← 2


- ← t


(7)

N

 

 

그런데 잘 알려져있다싶이 N의 가장 첫번째 row는 나머지 row들의 linear combination으로 구성된다. 즉 −Ny = N {yk}|A|k=1

yts ≤ 0 제약식에서 첫번째 row의 제약식은 항상 drop해도 된다. 이제 앞으로 N을 지칭할 땐 가장 첫번째 row는 생략한 matrix로 N ∈ R(|V|−1)×(|A|+1)을 지칭할 것임. 그렇다면 다시 N을





|Regular arcs (s,1) (s,2) (1,2) (1,t) (2,t)|Dummy (t,s)<br><br>|
|---|---|
|−1 0 +1 +1 0<br><br>0 −1 −1 0 +1 0 0 0 −1 −1|0 0 +1<br><br>|


- ← 1
- ← 2 ← t


N =

(8)

 

 

 

 , (9)

 Ny|Nts :=

 

- 0

···

- 1


N :=

Ny⊺ ∈ R|A|×(|V|−1) Nts⊺ := (0···1)

N⊺ =

(10)

### 1.3 Affine decision rules

먼저 uncertainty ξ ∈ R|A|를 한 차원 더 높여서 ξ ← (ξ 1)⊺로 만들자. 그리고 Uncertainty set U를 아래와 같이 정의:

U := ξ ∈ R|A|+1 : ξ ∈ Ξ, e⊺|A|+1ξ = 1 Ξ := (ζ,τ) ∈ R|A| × R+ : Dζ ≥ rτ

이렇게 한 차원을 높여서 만드는 이유는 (i) affine decision rule의 intercept term을 compact하게 처리하면서 동시에 (ii) affine function을 quadratic term으로 표현하기 위해.





   =: Πˆξ =

  

  

  

ξ1 . ξ|A| 1

 

  =

πˆ11 ··· πˆ|A|1 πˆ01

|A| l=1 πˆl1ξl + ˆπ01

πˆ1(ξ) ··· πˆt(ξ)

... . . πˆ1t ··· πˆ|A|t πˆ0t

πˆ(ξ) ∈ R|V|−1 =

···

 

 

.

|A| l=1 πˆltξl + ˆπ0t





  

  

   =: Φˆξ =

  

  

   =

ξ1 . ξ|A| 1

ϕˆ11 ··· ϕˆ1|A| ϕˆ10

|A| l=1

ϕˆ1l ξl + ϕˆ10

ϕˆ1(ξ) . ϕˆ|A|(ξ)

... . . ϕˆ|A|1 ··· ϕˆ|A||A| ϕˆ|A|0

ϕˆ(ξ) =

 

 

.

.

|A| l=1

ϕˆ|A|l ξl + ϕˆ|A|0





  

   =: Y ξˆ =

  

  

  

   =

ξ1 . ξ|A| 1

yˆ11 ··· yˆ|A|1 yˆ01

|A| l=1 yˆl1ξl + ˆy01

yˆ1(ξ) . yˆ|A|(ξ)

... . . yˆ1|A| ··· yˆ|A||A| yˆ0|A|

yˆ(ξ) =

 

 

.

.

|A| l=1 yˆl|A|ξl + ˆy0|A|





ξ1 . ξ|A| 1

yˆts(ξ) = yˆts(ξ) = |A|l=1 yˆltsξl + ˆy0ts =: Yˆtsξ = yˆ1ts ··· yˆ|A|ts yˆ0ts

 

 

  

  ,

yˆ1ts ···

yˆts yˆ0ts

Yˆts⊺ =

=

yˆ|A|ts yˆ0ts

ξs = (ξs 1)⊺ Π =ˆ ΠˆL πˆ0 ,Φ =ˆ ΦˆL ϕˆ0 ,Yˆ = YˆL yˆ0

### 1.4 Robust Counterparts of Linear Uncertainty

worst-case objective를 제외한 나머지 제약식 함수는 전부 uncertainty에 linear 하므로, kuhn의 표기법을 가져 와서 reformulate하면, 먼저 notation을 위해 uncertainty set을 다시 정리하면,

U := ξ ∈ R|A| : ξ = ξ¯+ Dζ, ζ ∈ Ξ Ξ := ζ ∈ R|A| : ∥ζ∥ ≤ ε

그러면 여기서는 notation을 좀 abuse해서 아래와 같은 alternative form로 represent하자 ξ = ξ¯(1 + ζ), ∥ζ 2 ≤ ε

Ξs = ζ ∈ R|A| : ∥ζ∥2 ≤ ε , Ds = diag(ξ¯s)

= ζ ∈ R|A| : Rζ ⪰KSOC

r¯s

0 ∈ R1×|A| I ∈ R|A|×|A|

, r¯s = −ε 0

where R =

,

여기서 ζˆs는 scenario s의 값; ε은 value of robustness (hyperparameter) 그리고 linear constraints들을 다시

matrix form으로 compact하게 formulate하자:

Ny⊺(˜π0s + Π˜sLζ) + ϕ˜s0 + Φ˜sLζ ≥ 0 ∀ζ ∈ Ξs ∀s ∈ [S] Nts⊺(˜π0s + Π˜sLζ) ≥ λ ∀ζ ∈ Ξs,∀s ∈ [S], Ny⊺(ˆπ0s + ΠˆsLζ) + ϕˆs0 + ΦˆsLζ ≥ 0 ∀ξ ∈ Ξs ∀s ∈ [S], Nts⊺(ˆπ0s + Πˆsζ) ≥ 1 ∀ξ ∈ Ξs ∀s ∈ [S],

− Ny(˜y0s + Y˜sζ) − Nts(˜y0ts,s + Y˜Lts,sζ) ≥ 0 ∀ξ ∈ Us ∀s ∈ [S], − y˜0s − Y˜Lsζ + λdiag(1 − v ⊙ x)(ξ¯+ Dζ) ≥ −h ∀ξ ∈ Ξs ∀s ∈ [S],

π˜0s + Π˜sLζ ≥ 0, ϕ˜s0 + Φ˜sLζ ≥ 0, πˆ0s + Πˆsζ ≥ 0, ϕˆs0 + Φˆsζ ≥ 0, ∀ξ ∈ Ξs ∀s ∈ [S], e⊺h ≤ λw,λ,ν ≥ 0

그리고 worst-case expectation coupling constraint들은

S

µ˜sk + ˆµsk ≤ Sν ∀k ∈ A,

s=1

ϕ˜s0,k + (Φ˜sL,k)⊺ξ ≤ µ˜sk ∀ξ ∈ Ξs ∀s ∈ [S],∀k ∈ A, ϕˆs0,k + (ΦˆsL,k)⊺ζ ≤ µˆsk ∀ξ ∈ Ξs ∀s ∈ [S],∀k ∈ A,

−Φ˜sLζ ≥ ϕ˜s0 − µ˜s = (˜µs1 ···µ˜s|A|)⊺ −ΦˆsLζ ≥ ϕˆs0 − µˆs = (ˆµs1 ···µˆs|A|)⊺

⇔

여기서 시나리오 간 coupling이 발생. 이걸 이제 leader와 follower의 decision rule space로 separate해서 보면, Leader의 space에선

N⊺Πˆs + I0⊺Φˆs ζ ≥ d0 N⊺πˆ0s I0⊺ϕˆs0 ∀ξ ∈ Ξs, ∀s ∈ [S]

Πˆs Φˆs

πˆ0s ϕˆs0 ∀ξ ∈ Ξs,∀s ∈ [S],

ξ ≥ −

− Φˆsξ ≥ ϕ˜s0 − µˆs ∀ξ ∈ Us,∀s ∈ [S] 그럼 이제

그럼 이제 robust counterpart를

Qˆ := N⊺ΠˆsL + I0⊺ΦˆsL , Qˆζ ≥ d0 − N⊺πˆ0s − I0⊺ϕˆ0 ∀ξ ∈ Ξs,s ∈ [S]

- Λˆs1R =

 

Qˆ ΠˆsL ΦˆsL

 , Λˆs1r¯s ≥

 

d0 − N⊺πˆ0s − I0⊺ϕˆ0 −πˆ0s −ϕˆs0

 ,[Λˆs1]m ∈ KSOC ∀m, (11a)

- Λˆs2R = −ΦˆsL, Λˆs2r¯s ≥ −µˆs + ϕˆs0, Λˆs2]m ∈ KSOC ∀m (11b)


(왜 이렇게 나오냐면)

a⊺mξ ≥ bm ∀m ⇔ sup

Aξ ≥ b ⇔ inf

inf

Rξ⪰r

ξ

##### λ⊺ar ≥ bm, λ⊺mR = am,λm ∈ K ⇔ Λr ≥ b, ΛR = A,[Λ]m ∈ K

λm

Follower의 space에선

 

 ζ ≥

 

  ∀ζ ∈ Ξs s ∈ [S]

N⊺Π˜s + I0⊺Φ˜s −NyY˜Ls − NtsY˜Lts,s −Y˜Ls + diag(λ − v ⊙ ψ0)D

λd0 − N⊺π˜0s − I0⊺ϕ˜s0 Nyy˜0s + Ntsy˜0ts,s −h + ˜y0s − diag(λ − v ⊙ ψ0)ξ¯s

  ∀ζ ∈ Ξs,

 

 ζ ≥

 

Π˜sL Φ˜sL Y˜Ls

−π˜0s −ϕ˜s0 −y˜0s

− Φ˜sLζ ≥ −µ˜s + ϕ˜s0 ∀ζ ∈ Ξs,s ∈ [S]

*여기서 Y˜tssξ ≥ 0은 굳이 안넣어도 될듯함. 이걸 maximize하는 거잖아. 그럼 이제

 

 

N⊺Π˜s + I0⊺Φ˜s −NyY˜Ls − NtsY˜Lts,s

Q˜ :=

Y˜Ls diag(λ v ⊙ ψ0)D

그럼 이제 robust counterpat를

- Λ˜s1R =

  

Q˜s Π˜sL Φ˜sL Y˜Ls

  , Λ˜s1r¯s ≥



 

 

λd0 0

−h + ˜y0s − diag(λ − v ⊙ ψ0)ξ¯s

 

π˜0s −ϕ˜s0 −y˜0s



 

,[Λ˜s1]m ∈ KSOC, (11c)

- Λ˜s2R = −Φ˜sL, Λ˜s2 ≥ −µ˜s + ϕ˜s0, Λ˜s2]m ∈ KSOC (11d)


the below inequalities represents following mccormick reformulation ψk0 ≤ λUxk, ψk0 − λ ≤ 0, λ − ψk0 ≤ λU(1 − xk) ∀k ∈ A

## 2 Finite Semidefinite reformulation

### 2.1 Uncertainty Quantification

A

A

S

Eξ∼Pˆ

ξk(1 − vkxk)ϕˆk(ζ)

ξk(1 − vkxk)ϕˆk(ζ) = (1/S)

sup

sup

ξ∈Us

Pˆ∈B

s=1

k=1

k=1

A

A

S

Eξ∼P˜

ξk(1 vkxk)ϕ˜k(ζ) y˜ts(ζ) = (1/S)

ξk(1 vkxk)ϕ˜k(ζ) y˜ts(ζ)

sup

sup

ξ∈Us

P˜∈B

s=1

k=1

k=1

where Us := ξ ∈ R|A| : ξ = ζ¯+ Dζ,ζ ∈ Ξs , Ξs := ζ ∈ R|A| : ζ∥ ≤ ε 첫 번째 epigraph 제약식은 아래와 같이 서술:

S

ηˆs + ˜ηs ≤ St

s=1

A

ξk(1 − vkxk)ϕˆk(ζ) ≤ ηˆs ∀s ∈ [S],

sup

ξ∈Us

k=1

A

ξk(1 − vkxk)ϕ˜k(ζ) − y˜ts(ζ) ≤ η˜s ∀s ∈ [S]

sup

ξ∈Us

k=1

여기서 [1]나 [2] 처럼 event-wise adaptation 혹은 multipolicy approximation을 사용하면,

Φˆ,Φ˜ ← Φˆs,Φ˜s

, Y,ˆ Y˜ ← Yˆs,Y˜s

s∈[S]

, Πˆ,Π˜ ← Πˆs,Π˜s

s∈ S

s∈[S]

위의 식은 이제

 

 

 

  ≤ ηˆs ∀s ∈ [S],

|A|

A

(ξ¯k + (Dζ)k)(1 vkxk)

ϕˆk,sl ζl + ϕˆk,s0

sup

ξ∈Us

k=1

l=1

 

 

  −

 

 

  ≤ η˜s ∀s ∈ [S]

|A|

|A|

A

(ξ¯k + (Dζ)k)(1 − vkxk)

ϕ˜k,sl ζl + ϕ˜k,s0

y˜lts,sζl + ˜y0ts,s

sup

ξ∈Us

k=1

l=1

l=1

Let

  , Φ˜L :=

  

  

  , ϕˆ0 :=

  

  , ϕ˜0 :=

  

  

ϕ˜11 ··· ϕ˜1|A|

ϕˆ11 ··· ϕˆ1|A|

ϕ˜10 . ϕ˜|A|0

ϕˆ10 . ϕˆ|A|0

... . ϕ˜|A|1 ··· ϕ˜|A||A|

... . ϕˆ|A|1 ··· ϕˆ|A||A|

ΦˆL :=

.

.

혹은 매트릭스 형태로:

⊺

ζ + (ϕˆs0 v ⊙ ψˆ0s)⊺ξ¯s ≤ ηˆs, ∀s ∈ [S]

ζ⊺D⊺(ΦˆsL diag(v)ΨˆsL)ζ + (ΦˆsL diag(v)ΨˆsL)ξ¯s + D⊺(ϕˆs0 v ⊙ ψˆ0s)

sup

ζ∈Ξs

(12) sup

⊺

ζ⊺D⊺(Φ˜sL − diag(v)Ψ˜sL)ζ + (Φ˜sL − diag(v)Ψ˜sL)ξ¯s + D⊺(ϕ˜s0 − v ⊙ ψ˜0s) − y˜Lts,s

ζ + (ϕ˜s0 − v ⊙ ψ˜0s)⊺ξ¯s − y˜0ts,s ≤ η˜s, ∀

ζ∈Ξs

(13) 여기서 v ⊙ x는 element-wise product (Hadamard product)를 의미.

#### 2.1.1 Leader’s constraints

⊺

ζ⊺D⊺(ΦˆsL − diag(v)ΨˆsL)ζ + (ΦˆsL − diag(v)ΨˆsL)ξ¯s + D⊺(ϕˆs0 − v ⊙ ψˆ0s)

ζ + (ϕˆs0 − v ⊙ ψˆ0s)⊺ξ¯s ≤ ηˆs

sup

ζ

s.t. ζ⊺ζ ≤ ε2 Then, Proposition 1 (Inhomogenious S-Lemma). Assuming the uncertainty set has non-empty relative interior, 0 ≤ inf

⊺

ζ − (ϕˆs0 − v ⊙ ψˆ0s)⊺ξ¯s + ˆηs : −ζ⊺ζ + ε2 ≥ 0 is valid, if and only if

−ζ⊺D⊺(ΦˆsL − diag(v)ΨˆsL)ζ − (ΦˆsL − diag(v)ΨˆsL)ξ¯s + D⊺(ϕˆs0 − v ⊙ ψˆ0s)

ζ

  ⪰ 0

 

ϑˆsI − D⊺(ΦˆsL − diag(v)ΨˆsL) −12 (ΦˆsL diag(v)ΨˆsL)ξ¯s + D⊺(ϕˆs0 v ⊙ ψˆ0s)

∃ϑˆs ≥ 0 s.t.

⊺

- 1

- 2 (ΦˆsL diag(v)ΨˆsL)ξ¯s D⊺(ϕˆs0 v ⊙ ψˆ0s)


ηˆs (ϕˆs0 v ⊙ ψˆ0s)⊺ξ¯s ϑˆsε2

그러면 constraint는 아래로 표현

 

  ⪰KP SD

ϑˆsI D⊺(ΦˆsL diag(v)ΨˆsL) 12 (ΦˆsL − diag(v)ΨˆsL)ξ¯s + D⊺(ϕˆs0 − v ⊙ ψˆ0s) −21 (ΦˆsL − diag(v)ΨˆsL)ξ¯s + D⊺(ϕˆs0 − v ⊙ ψˆ0s)

, ϑˆs ≥ 0

⊺

ηˆs − (ϕˆs0 − v ⊙ ψˆ0s)⊺ξ¯s − ϑˆsε2

(14a) (14b)

ψˆlk,s ≤ ϕUxk, ψˆlk,s ≤ ϕˆk,sl , ϕˆk,sl ≤ ψˆlk,s + ϕU(1 xk) ∀l ∈ [1 : |A| + 1],k ∈ [1 : |A|], (14c)

  

   = ΨˆL ψˆ0 (14d)

ψˆ11 ··· ψˆ|A|1 ψˆ01

... . . ψˆ1|A| ··· ψˆ|A||A| ψˆ0|A|

Ψ =ˆ

.

#### 2.1.2 Follower’s constraints

⊺

ζ⊺D⊺(Φ˜sL − diag(v)Ψ˜sL)ζ + (Φ˜sL − diag(v)Ψ˜sL)ξ¯s + D⊺(ϕ˜s0 − v ⊙ ψ˜0s) − y˜Lts,s

ζ + (ϕ˜s0 − v ⊙ ψ˜0s)⊺ξ¯s − y˜0ts,s ≤ η˜s,

sup

ζ∈Ξs

(15a) s.t. ζ⊺ζ ≤ ε2 (15b) Proposition 2.

0 ≤ inf

⊺

ζ − (ϕ˜s0 − v ⊙ ψ˜0s)⊺ξ¯s + ˜y0ts,s + ˜ηs : −ζ⊺ζ + ε2 ≥ 0}

{−ζ⊺D⊺(Φ˜sL − diag(v)Ψ˜sL)ζ − (Φ˜sL − diag(v)Ψ˜sL)ξ¯s + D⊺(ϕ˜s0 − v ⊙ ψ˜0s) − y˜Lts,s

ζ

is valid, if and only if

 

 

ϑ˜sI − D⊺(Φ˜sL − diag(v)Ψ˜sL) −21 (Φ˜sL − diag(v)Ψ˜sL)ξ¯s + D⊺(ϕ˜s0 − v ⊙ ψ˜0s) − y˜Lts,s −21 (Φ˜sL − diag(v)Ψ˜sL)ξ¯s + D⊺(ϕ˜s0 − v ⊙ ψ˜0s) − y˜Lts,s

∃ϑ˜s ≥ 0 s.t.

⊺

η˜s − (ϕ˜s0 − v ⊙ ψ˜0s)⊺ξ¯s + ˜y0ts,s − ϑ˜sε2)

그러면 constriant는 아래로 표현

  ⪰KP SD

 

ϑ˜sI − D⊺(Φ˜sL − diag(v)Ψ˜sL) −12 (Φ˜sL − diag(v)Ψ˜sL)ξ¯s + D⊺(ϕ˜s0 − v ⊙ ψ˜0s) − y˜Lts,s −12 (Φ˜sL − diag(v)Ψ˜sL)ξ¯s + D⊺(ϕ˜s0 − v ⊙ ψ˜0s) − y˜Lts,s

0,

⊺

η˜s − (ϕ˜s0 − v ⊙ ψ˜0s)⊺ξ¯s + ˜y0ts,s − ϑ˜sε2

(15c) ϑ˜s ≥ 0 (15d) ψ˜lk,s ≤ ϕUxk, ψ˜lk,s ≤ ϕ˜k,sl , ϕ˜k,sl ≤ ψ˜lk,s + ϕU(1 − xk) ∀l ∈ [1 : |A| + 1],k ∈ [1 : |A|] (15e)

  

   (15f)

ψ˜11 ··· ψ˜|A|1 ψ˜01

... . . ψ˜1|A| ··· ψ˜|A||A| ψ˜0|A|

Ψ =˜

.

- 2.2 Full Model 이제 전체 모델을 다시 쓰면


2DRNDP (16a)

= min t + wν (16b) s.t. e⊺h ≤ λw,x ∈ X,h ≥ 0, (16c)

S

ηˆs + ˜ηs ≤ St, (16d)

s=1

 

  ⪰KP SD

ϑˆsI − D⊺(ΦˆsL − diag(v)ΨˆsL) −12 (ΦˆsL − diag(v)ΨˆsL)ξ¯s + D⊺(ϕˆs0 − v ⊙ ψˆ0s) −21 (ΦˆsL − diag(v)ΨˆsL)ξ¯s + D⊺(ϕˆs0 − v ⊙ ψˆ0s)

, ϑˆs ≥ 0

⊺

ηˆs − (ϕˆs0 − v ⊙ ψˆ0s)⊺ξ¯s − ϑˆsε2

(16e)

  ⪰KP SD

 

ϑ˜sI − D⊺(Φ˜sL − diag(v)Ψ˜sL) −12 (Φ˜sL − diag(v)Ψ˜sL)ξ¯s + D⊺(ϕ˜s0 − v ⊙ ψ˜0s) − y˜Lts,s

0,ϑ˜

⊺

η˜s (ϕ˜s0 v ⊙ ψ˜0s)⊺ξ¯s + ˜y0ts,s ϑ˜sε2

- 1

- 2 (Φ˜sL diag(v)Ψ˜sL)ξ¯s + D⊺(ϕ˜s0 v ⊙ ψ˜0s) y˜Lts,s


(16f) ψˆlk,s ≤ ϕUxk, ψˆlk,s ≤ ϕˆk,sl , ϕˆk,sl ≤ ψˆlk,s + ϕU(1 − xk) ∀l ∈ [1 : |A| + 1],k ∈ [1 : |A|], (16g) ψ˜lk,s ≤ ϕUxk, ψ˜lk,s ≤ ϕ˜k,sl , ϕ˜k,sl ≤ ψ˜lk,s + ϕU(1 − xk) ∀l ∈ [1 : |A| + 1],k ∈ [1 : |A|], (16h)

S

µ˜sk + ˆµsk ≤ Sν ∀k ∈ A, (16i)

s=1

- Λˆs1R =

 

N⊺ΠˆsL + I0⊺ΦˆsL ΠˆsL ΦˆsL

 , Λˆs1r¯s ≥

 

d0 − N⊺πˆ0s − I0⊺ϕˆ0 −πˆ0s −ϕˆs0

 ,[Λˆs1]m ∈ KSOC ∀m, (16j)

- Λˆs2R = −ΦˆsL, Λˆs2r¯s ≥ −µˆs + ϕˆs0, Λˆs2]m ∈ KSOC ∀m, (16k)


- Λ˜s1R =



 

 

N⊺Π˜s I0⊺Φ˜s −NyY˜Ls − NtsY˜Lts,s −Y˜Ls + diag(λ v ⊙ ψ0)D

 

Π˜sL Φ˜sL Y˜Ls



 

, Λ˜s1r¯s ≥



 

 

λd0 − N⊺π˜0s − I0⊺ϕ˜s0 Nyy˜0s + Ntsy˜0ts,s −h + ˜y0s − diag(λ − v ⊙ ψ0)ξ¯s

 

−π˜0s −ϕ˜s0 −y˜0s



 

,[Λ˜s1]m ∈ KSOC,

(16l)

- Λ˜s2R = −Φ˜sL, Λ˜s2 ≥ −µ˜s + ϕ˜s0, [Λ˜s2]m ∈ KSOC (16m) ψ0 ≤ λUx, ψ0 − λ(e) ≤ 0, λ(e) − ψ0 ≤ λU(e − x) ∀k ∈ A (16n) Ψˆs = ΨˆsL ψˆ0s Ψ =˜ Ψ˜sL ψ˜0s (16o) Issue들


- 1. h = 0,λ > 0 infeasible나는 이유:

- - y(ξ) − h ≤ λξ(1 − vx)∀ξ에서 worst-case uncertainty가 negative가 나올 수 있음 (epsilon의 크기에 따라)
- - epsilon을 너무 작게잡으면 ϵ2이 너무 작아져서 numerically instable, slow progress 에러가 남
- - 해당 RC에 한해 intersection with {ξ ≥ 0}을 고려할 수 있음
- - SDP part의 경우, worst-case가 negative가 나오지 않음을 lemma로 보이면 해결됨 (근데 맞을지는 해봐야 암)


- 2. λ = 100 같이 크게 잡으면 infeasible이 나는 이유:


- λ ≤ π˜ts(ξ) ≤ ··· + ϕ˜(ξ)를 worst-case로 잡으려다보니 LDR coefficient Φ,Π의 upper bound를 넘는듯?

- 3. Uncertainty set에 scaling이 필요? epsilon의 magnitude가 너무 예민

- Uncertainty를 이렇게 모델링?

ξ = ξ¯s(1 + ζ), ζ ∈ {ζ : ∥ζ∥2 ≤ ε}

⇒ Ξs = ξ : D 1(ξ − ξ¯s) 2 ≤ ε

= ξ : (ξ − ξ¯s)⊺D 2(ξ − ξ¯s) ≤ ε2 where D = diag(ξ¯s), D−2 = diag(1/(ξ¯s)2)

이러면 ε ≤ 1 까지는 negative worst-case가 나오지 않음. ζ의 의미: nominal scenario 대비 몇 퍼센트까지 변화할지 (ε = 0.1이면 worst-case가 최대 하나의 arc capacity의 10% 변함)

- 4. λ,ψ0을 그냥 subproblem으로 내리면 안되나? 그러면 걔네는 다시 follower 문제로 옮겨가게 되고.. 대신 그러면 inner master problem에서 걔네의 constraint들의 dual variable들을 master로 넣어야 됨.


- - 만약 1.이 해결되면 feasibility cut은 순전히 2번 이슈로 인해 발생할 것임.
- - 1을 해결못하고 전부 다 feasibility cut으로 뭉개도 여전히 worst-case가 negative인 이슈는 못피함.


### 2.3 Outer Master Problem

min t0 (17a)

s.t. 1⊺h ≤ w,x ∈ X, (17b) t0 ≥ Z0(x,h,λ,ψ0), (17c) ψ0 ≤ λUx, ψ0 λ(e) ≤ 0, λ(e) ψ0 ≤ λU(e x),ψ0 ≥ 0 ∀k ∈ A (17d)

### 2.4 Outer Subproblem

Z0(x,h,λ,w0) = (18a) min (1/S)

S

ηˆs + ˜ηs + (1/S)wν (18b)

s=1

s.t. (16e) (18c)

- (16f) (18d)
- (16g) (18e)
- (16h) (18f)
- (16i) (18g)
- (16j) (18h)
- (16k) (18i)
- (16l) (18j)
- (16m) (18k)


decision variables:

ηˆs,η˜s,ν,ΦˆsL,ϕˆs0,ΨˆsL,ψˆ0s,Φ˜sL,ϕ˜s0,Ψ˜sL,ψ˜0s,y˜Lts,s,y˜0ts,sµ˜sk,µˆsk,Λˆs1,Λˆs2,Λ˜1,Λ˜2

And below are necessary condition (무시해도 되지만 알고리즘 안정성 위해 넣어줌. 그러면 Dual도 equality가 아니라 inequality가 됨)

ηˆs,ν,µ,ˆ µ˜ ≥ 0

*˜η는 항상 non-negative하지 않음.

- 2.5 Dualized Outer Subproblem 참고: SDP 변수 x에 대해 Linear Operator A : Rn → Sn 로 정의:


A(x) = x가 있는 term Adjoint operator A∗ : Sn → Rn 구하기 위해 inner product 사용: dual이 M이라면 ⟨A(x),M⟩ ⟨x,A∗(M)⟩

그러면 trace 성질을 이용해서 ⟨x,?⟩이 되게 유도하고, ?를 M에 대한 linear operator (trace, inner product)로 나타내면 듀얼을 구할 수 있음.

#### 2.5.1 Dual Variables and Their OriginFirst, for semidefinite constraints,

(Mˆ s ∈ KSDP(R(|A|+1)×(|A|+1))···)

 

  ⪰KP SD

ϑˆsI − D⊺(ΦˆsL − diag(v)ΨˆsL) −21 (ΦˆsL diag(v)ΨˆsL)ξ¯s + D⊺(ϕˆs0 v ⊙ ψˆ0s) −21 (ΦˆsL − diag(v)ΨˆsL)ξ¯s + D⊺(ϕˆs0 − v ⊙ ψˆ0s)

0

⊺

ηˆs − (ϕˆs0 − v ⊙ ψˆ0s)⊺ξ¯s − ϑˆsε2

M˜ s ∈ KSDP(R(|A|+1)×(|A|+1))···

 

ϑ˜sI − D⊺(Φ˜sL − diag(v)Ψ˜sL) −21 (Φ˜sL − diag(v)Ψ˜sL)ξ¯s + D⊺(ϕ˜s0 − v ⊙ ψ˜0s) − y˜Lts,s −12 (Φ˜sL − diag(v)Ψ˜sL)ξ¯s + D⊺(ϕ˜s0 − v ⊙ ψ˜0s) − y˜Lts,s

⊺

η˜s − (ϕ˜s0 − v ⊙ ψ˜0s)⊺ξ¯s + ˜y0ts,s − ϑ˜sε2

나머진 테이블로 정리.

  ⪰KP SD

0

Dual Variable Primal Constraint

- Uˆ1s ≥ 0 [−ΨˆsL − ψˆ0s] ≥ −ϕUdiag(x)E
- Uˆ2s ≥ 0 [−ΨˆsL − ψˆ0s] + [ΦˆsL ϕˆs0 ≥ 0
- Uˆ3s ≥ 0 [−ΦˆsL − ϕˆs0] + [ΨˆsL ψˆ0s ≥ −ϕUdiag(1 − x)E


- U˜1s ≥ 0 [ Ψ˜sL ψ˜0s] ≥ −ϕUdiag(x)E
- U˜2s ≥ 0 [−Ψ˜sL − ψ˜0s] + [Φ˜sL Φ˜s0] ≥ 0
- U˜3s ≥ 0 [−Φ˜sL − ϕ˜s0] + [Ψ˜sL ψ˜0s] ≥ −ϕUdiag(1 − x)E αk ≥ 0,k ∈ A ν − Ss=1(˜µsk + ˆµsk) ≥ 0


- Zˆ1s (free) Λˆs1R

 

N⊺ΠˆsL + I0⊺ΦˆsL ΠˆsL ΦˆsL

  = 0

- Zˆ2s (free) Λˆs2R + ΦˆsL = 0


- Z˜1s ∈ R(.)×(|A|)(free) Λ˜s1R −



 

 

N⊺Π˜sL + I0⊺Φ˜sL −NyY˜Ls − NtsY˜Lts,s −Y˜Ls

 

Π˜sL Φ˜sL Y˜Ls



 

=



 

 

0 0 diag(λ − vψ0)Ds

 

0 0 0



 

- Z˜2s (free) Λ˜s2R + Φ˜sL = 0


- βˆ1s ≥ 0 Λˆs1r¯s +

 

N⊺πˆ0s + I0⊺ϕˆs0 πˆ0s ϕˆs0

  ≥

 

d0 0 0

 

- βˆ2s ≥ 0 Λˆs2r¯s − ϕˆs0 + ˆµs ≥ 0


- β˜1s ≥ 0 Λ˜s1r¯s +



 

N⊺π˜0s + I0⊺ϕ˜s0 −Nyy˜0s − Ntsy˜0ts,s −y˜0s π˜0s ϕ˜s0 y˜0s



 

≥



 

λd0 0 −h−diag(λ − v ⊙ ψ0)ξ¯s 0 0 0



 

- β˜2s ≥ 0 Λ˜s2r¯s − ϕ˜s0 + ˜µs ≥ 0


- Γˆs1 ∈ KSOCm(Λˆ1), (⇔ [Γˆs1]m ⪰KSOC

0) Λˆs1 ∈ Km(Λˆ

s 1)

SOC (⇔ Λˆs1]m ⪰KSOC

0, m ∈ [1,m(Λˆs1)]), where m(Λˆs1) is row dimension of

- Γˆs2 ∈ KSOCm(Λˆ2) Λˆs2 ∈ KSOCm(Λˆ2)


- Γ˜s1 ∈ KSOCm(Λ˜1) Λ˜s1 ∈ KSOCm(Λ˜1)
- Γ˜s2 ∈ KSOCm(Λ˜2) Λ˜s2 ∈ KSOCm(Λ˜2)


- Pˆ1s,ϕ ≥ 0,Pˆ1s,π ≥ 0, [ΦˆsL ϕˆs0] ≥ −ϕUE, [ΠˆsL πˆ0s] ≥ −πUE
- Pˆ2s,ϕ ≥ 0,Pˆ2s,π ≥ 0, −[ΦˆsL ϕˆs0] ≥ −ϕUE, −[ΠˆsL πˆ0s] ≥ −πUE


- P˜1s,ϕ ≥ 0,P˜1s,π ≥ 0,P˜1s,y ≥ 0,P˜s,y

ts

- 1 ≥ 0 [Φ˜sL ϕ˜s0] ≥ −ϕUE,[Π˜sL π˜0s] ≥ −πUE,[Y˜Ls y˜0s ≥ −y0E, [Y˜Lts,s y˜0ts,s] ≥ −y˜ts0 E

P˜2s,ϕ ≥ 0,P˜2s,π ≥ 0,P˜2s,y ≥ 0,P˜s,y

ts

- 2 ≥ 0 −[Φ˜sL ϕ˜s0 ≥ −ϕUE,−[Π˜sL π˜0s] ≥ −πUE,−[Y˜Ls y˜0s] ≥ −y0E, −[Y˜Lts,s y˜0ts,s] ≥ −y˜ts0 E




Rˆ KYˆ = 0

#### 2.5.2 Dual Problem Formulation

S

− ϕU⟨Uˆ1s + U˜1s,diag(x)E⟩ − ϕU⟨Uˆ3s + U˜3s,diag(1 − x)E⟩

max

s=1

+ d⊺0βˆ1s,1 + ⟨Z˜1s,3,diag(λ − vψ0)Ds⟩ + λd⊺0β˜1s,1 − (h+diag(λ − v ⊙ ψ0)ξ¯s)⊺β˜1s,3 − ϕU Pˆ1s,ϕ

+ Pˆ2s,π − ϕU P˜1s,ϕ

− πU Pˆ1s,π

+ Pˆ2s,ϕ

F

F

F

+ P˜2s,π

− πU P˜1s,π

+ P˜2s,ϕ

F − yU P˜1s,y

F

F

F

+ P˜s,y

− ytsU P˜s,y

+ P˜2s,y

ts

ts

(19) Subject to:

2

1

F

F

F

F

#### 2.5.3 Cone Constraints:

Mˆ s,M˜2s ∈ KSDP, s = 1,...,S (20) Uˆ1s,Uˆ2s,Uˆ3s,U˜4s,U˜5s,U˜6s ≥ 0 (element-wise), s = 1,...,S (21) Pˆs,P˜s ≥ 0, s = 1,··· ,S (22) αk ≥ 0, ∀k ∈ A (23) βˆ1s,βˆ2s,β˜1s,β˜2s ≥ 0, s = 1,...,S (24) Γˆs1,Γˆs2,Γ˜s1,Γ˜s2 ∈ KSOCm(···), s = 1,...,S (25)

#### 2.5.4 Scalar Constraints:

From ˆηs(≥ 0):

Mˆ s,22 ≤ 1/S, s = 1,...,S (primal: ˆηs) (26) From ˜ηs (free!):

M˜ s,22 = 1/S, s = 1,...,S (primal: ˜ηs) (27) From ν(≥ 0):

αk ≤ w(1/S) (primal: ν) (28)

k∈A

From ϑˆs(≥ 0):

tr(Mˆ s,11) − Mˆ s,22ε2 ≤ 0 (29) From ϑ˜s(≥ 0):

tr(M˜ s,11) M˜ s,22ε2 ≤ 0 (30)

#### 2.5.5 Matrix Equality Constraints:From Φˆs:

[A∗ΦˆsL(Mˆ s) A∗ϕˆs0(Mˆ s,12)+A∗ϕˆs0(Mˆ s,22)]+Uˆ2s−Uˆ3s+[−(I0Zˆ1s,1+Zˆ1s,3)+Zˆ2s I0βˆ1s,1+βˆ1s,3−βˆ2s]+Pˆ1s,ϕ−Pˆ2s,ϕ = 0

(31) A∗L(Mˆ s) = A∗L(Mˆ s,11) + A∗L(Mˆ s,12) A∗L(Mˆ s,11) = −DsMˆ s,11, A∗L(Mˆ s,12) = −Mˆ s,12(ξ¯s)⊺

A∗ϕˆs0(Mˆ s,12) = DsMˆ s,12, A∗. (Mˆ s,22) = ζ¯sMˆ s,22 From Ψˆs = ΨˆsL ψˆ0s (≥ 0):

A∗L(Mˆ s,11) + A∗L(Mˆ s,12) A∗0(Mˆ s,12) + A∗0(Mˆ s,22) − Uˆ1s − Uˆ2s + Uˆ3s ≤ 0 (primal: Ψˆs) (32)

A∗L(Mˆ s,11) = diag(v)DsMˆ s,11, A∗L(Mˆ s,12) = diag(v)Mˆ12ζ¯⊺ A∗0(Mˆ s,12) = diag(v)DsMˆ s,12, A∗0(Mˆ s,22) = ζ¯sMˆ s,22diag(v)

From Φ˜s = Φ˜sL ϕ˜s0 : [A∗L(M˜ ) A∗0(M˜ s,12)+A∗0(M˜ s,22)]+U˜2s−U˜3s+[−I0Z˜1s,1−Z˜1s,5+Z˜2s I0β˜1s,1+β˜1s,5−β˜2s]+P˜1s,ϕ−P˜2s,ϕ = 0 (33)

A∗L(M˜ ) = A∗L(M˜ s,11) + A∗L(M˜ s,12) A∗L(M˜ s,11) = −DsM˜ s,11, A∗L(M˜ s,12) = −M˜12ζ¯⊺ A∗0(M˜ s,12) = DsM˜ s,12, A∗0(M˜ s,22) = ζ¯sM˜ s,22

From Ψ˜s = Ψ˜sL ψ˜0s (≥ 0): A∗L(M˜ s,11) + A∗L(M˜ s,12) A∗0(M˜ s,12) + A∗0(M˜ s,22) U˜1s U˜2s + U˜3s ≤ 0 (primal: Ψ˜s) (34)

A∗L(M˜ s,11) = diag(v)DsM˜ s,11, A∗L(M˜ s,12) = diag(v)M˜12ζ¯⊺ A∗0(M˜ s,12) = diag(v)DsM˜ s,12, A∗0(M˜ s,22) = ζ¯sM˜ s,22diag(v)

From Y˜tss = Y˜Lts,s(= (˜yts,s)⊺) y˜0ts,s ⊺: A∗L(M˜ s,21) A∗0(M˜ s,22) + Nts⊺Z˜1s,2 −Nts⊺β˜1s,2 + P˜s,y

##### 1 − P˜s,y

2 = 0 (primal: Y˜tss) (35)

ts

ts

A∗L(M˜ s,12) = M˜ s,12 A∗0(M˜ s,22) = M˜ s,22

From ˆµs:

βˆ2s,k = αk, ∀k ∈ A,s = 1,...,S (primal: ˆµsk) (36) From ˜µs:

β˜2s,k = αk, ∀k ∈ A,s = 1,...,S (primal: ˜µsk) (37) From Πˆs = ΠˆsL πˆ0s :

−NZˆ1s,1 − Zˆ1s,2 Nβˆ1s,1 + βˆ1s,2 + Pˆ1s,π − Pˆ2s,π = 0 (primal: Πˆs) (38) From Π˜s = Π˜sL π˜0s :

−NZ˜1s,1 − Z˜1s,4 Nβ˜1s,1 + β˜1s,4 + P˜1s,π − P˜2s,π = 0 (primal: Π˜s) (39) From Y˜s Y˜Ls y˜0s :

- From Λˆs1:

- Zˆ1sR⊺ + βˆ1s(¯rs)⊺ + Γˆs1 = 0 (primal: Λˆs1) (41)

From Λˆs2:

- Zˆ2sR⊺ + βˆ2s(¯rs)⊺ + Γˆs2 = 0 (primal: Λˆs2) (42)




Ny⊺Z˜1s,2 + Z˜1s,3 − Z˜1s,6 −Ny⊺β˜1s,2 − β˜1s,3 + β˜1s,6 + P˜1s,y P˜2s,y = 0 (primal: Y˜s) (40)

- From Λ˜s1:

- Z˜1sR⊺ + β˜1s(¯rs)⊺ + Γ˜s1 = 0 (primal: Λ˜s1) (43)

From Λ˜s2:

- Z˜2sR⊺ + β˜2s(¯rs)⊺ + Γ˜s2 = 0 (primal: Λ˜s2) (44)




#### 2.5.6 Notation

- • M =

M11 ∈ R|A|×(A| M12 ∈ R|A|+1 M12 M22 ∈ R

- • Zis =

  

- Zis,1
- Zis,2


.

   where each block has appropriate dimensions

- • βis,j denotes the j-th block of βis
- • y˜tss 의 contribution 중 변수 Z˜1s,2 계산하기:


⟨Z˜1s,2,−Nts(˜ytss )⊺⟩ = −tr((Z˜1s,2)⊺Nts(˜yts)⊺) tr(ABC)=tr(CAB)=tr(BCA)

= tr(˜yts⊺ (Z˜1s,2)⊺(−Nts))

= −y˜ts⊺ (Z˜1s,2)⊺Nts = −(Nts⊺Z˜1s,2)⊺y˜ts

- 2.6 Inner Master Problem With the dual outer subproblem, we can benders decompose again as follows:

max

S

s=1

ts1 (45a)

s.t.

k∈A

αk = w(1/S), (45b)

ts1 ≤ Z1s(αk) (45c)

- 2.7 Inner Subproblem


Z1s(αk) = Z1L,s(αk) + Z1F,s(αk) (46a) where (46b)

Z1L,s(αk) = max ϕU⟨Uˆ1s,diag(x)E⟩ + ϕU⟨Uˆ3s,diag(1 − x)E⟩ + d⊺0βˆ1s,1 − ϕU Pˆ1s,ϕ

+ Pˆ2s,ϕ

− πU Pˆ1s,π

+ Pˆ2s,π

F

F

F

s.t. Mˆ s ∈ KSDP, (46c) Uˆ1s,Uˆ2s,Uˆ3s,Pˆ1s,ϕ,Pˆ2s,ϕ,Pˆ1s,π,Pˆ2s,π ≥ 0 (element-wise), (46d) βˆ1s,βˆ2s ≥ 0 (46e) Γˆs1 ∈ Km(Γˆ

s 1)

SOC ,Γˆs2 ∈ KSOCm(.) , (46f) (26) (primal: ˆηs), (46g) (29) (primal: ϑˆs ≥ 0) (46h)

- (31) (primal: Φˆs), (46i)
- (32) (primal: Ψˆs), (46j)


(36) (primal: ˆµsk), (46k) (38) (primal: Πˆs), (46l)

- (41) (primal: Λˆs1), (46m)
- (42) (primal: Λˆs2) (46n)


Z1F,s(αk) = max ϕU⟨U˜1s,diag(x)E⟩ + ϕU⟨U˜3s,diag(1 − x)E⟩

+ ⟨Z˜1s,3,diag(λ − vψ0)Ds⟩ + λd⊺0β˜1s,1 − (h + diag(λ − v ⊙ ψ0)ξ¯s)⊺β˜1s,3 − ϕU P˜1s,ϕ

+ P˜2s,π

− πU P˜1s,π

+ P˜2s,ϕ

F yU P˜1s,y

F

F

F

P˜s,y

ytsU P˜s,y

+ P˜2s,y

ts

ts

(47a)

2

1

F

F

F

F

s.t. M˜ s ∈ KSDP, (47b) U˜1s,U˜2s,U˜3s ≥ 0 (element-wise), (47c) P˜1s,ϕ,P˜2s,ϕ,P˜1s,π,P˜2s,π,P˜1s,y,P˜2s,y,P˜s,y

1 ,P˜s,y

2 ≥ 0, (47d) β˜1s,β˜2s ≥ 0, (47e) Γ˜s1 ∈ KSOC,Γ˜s2 ∈ KSOC, (47f) (27) (primal: ˜ηs), (47g) (30) (primal: ϑ˜s ≥ 0), (47h)

ts

ts

(47i)

- (33) (primal: Φ˜s), (47j)
- (34) (primal: Ψ˜s), (47k)
- (35) (primal: ˜ytss ), (47l)


(37) (primal: ˜µsk), (47m)

- (39) (primal: Π˜s), (47n)
- (40) (primal: Y˜s), (47o)


- (43) (primal: Λ˜s1), (47p)
- (44) (primal: Λ˜s2) (47q)


The inner subproblems Z1L,s(αk) and Z1F,s(αk) are SDP. Their duals are SDP, which resemble the original primal problem but with αk as a given parameter instead of a decision variable.

### 2.8 Dual of Z1L,s(αk) (SDP)

- 2.8.1 Decision Variables ηˆs,Φˆs,Wˆ s,µˆs,Πˆs,Λˆs1,Λˆs2
- 2.8.2 Dual Problem

min

1 S

ηˆs +

k∈A

αkµˆsk (48a)

s.t. (16e),ϑˆs ≥ 0 ∀s ∈ [S] (48b) (16g) (48c)

- (16j) (48d)
- (16k) (48e)


- 2.8.3 Key Difference from Original Primal


- • Objective: The term k∈A αkµˆsk is added, where αk is a given parameter (not a decision variable)
- • Missing constraint: The constraint ν ≥ s(ˆµsk + ˜µsk) is now in the master problem


### 2.9 Dual of Z1F,s(αk) (SDP)

- 2.9.1 Decision Variables η˜s,Φ˜s,W˜ s,Y˜tss,µ˜s,Π˜s,Y˜s,Λ˜s1,Λ˜s2
- 2.9.2 Dual Problem

min

1 S

η˜s +

k∈A

αkµ˜sk (49a)

s.t. (16f),ϑ˜s ≥ 0 ∀s ∈ [S] (49b) (16h) (49c) (16l) (49d) (16m) (49e)

- 2.9.3 Key Difference from Original Primal


- • Objective: The term k∈A αkµ˜sk is added, where αk is a given parameter (not a decision variable)
- • Missing constraint: The constraint ν ≥ s(ˆµsk + ˜µsk) is now in the master problem Both dual inner subproblems are Copositive Programs that:


- 1. Have the same structure as the original primal COP
- 2. Include k∈A αkµsk in the objective (with αk as parameters)
- 3. Are parameterized by αk from the master problem
- 4. Can be solved independently for each scenario s


The master problem coordinates the αk values subject to k∈A αk = w, and uses Benders cuts from these subproblems.

## 3 Reference

