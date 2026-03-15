```
Comput Optim Appl (2016) 65:637–
DOI 10.1007/s10589-016-9851-z
```
## Inexact stabilized Benders’ decomposition approaches

## with application to chance-constrained problems with

## finite support

```
W. van Ackooij^1 · A. Frangioni^2 · W. de Oliveira^3
```
```
Received: 20 February 2015 / Published online: 26 May 2016
© Springer Science+Business Media New York 2016
```
```
Abstract We explore modifications of the standard cutting-plane approach for mini-
mizing a convex nondifferentiable function, given by an oracle, over a combinatorial
set, which is the basis of the celebrated (generalized) Benders’ decomposition
approach. Specifically, we combine stabilization—in two ways: via a trust region
in the L 1 norm, or via a level constraint—and inexact function computation (solu-
tion of the subproblems). Managing both features simultaneously requires a nontrivial
convergence analysis; we provide it under very weak assumptions on the handling
of the two parameters (target and accuracy) controlling the informative on-demand
inexact oracle corresponding to the subproblem, strengthening earlier know results.
This yields new versions of Benders’ decomposition, whose numerical performance
are assessed on a class of hybrid robust and chance-constrained problems that involve
a random variable with an underlying discrete distribution, are convex in the decision
variable, but have neither separable nor linear probabilistic constraints. The numerical
results show that the approach has potential, especially for instances that are difficult
to solve with standard techniques.
```
### BW. van Ackooij

```
wim.van-ackooij@edf.fr
A. Frangioni
frangio@di.unipi.it
W. de Oliveira
welington@ime.uerj.br
```
(^1) EDF R&D, OSIRIS, 7, Boulevard Gaspard Monge, 91120 Palaiseau Cedex, France
(^2) Dipartimento di Informatica, Università di Pisa, Largo B. Pontecorvo 3, 56127 Pisa, Italy
(^3) Universidade do Estado do Rio de Janeiro - UERJ, Rua São Francisco Xavier 524,
20550-900 Rio de Janeiro, Brazil


638 W. van Ackooij et al.

**Keywords** Benders’ decomposition·Chance-constrained problems·Mixed-integer
optimization·Nonsmooth optimization·Stabilization·Inexact function computation

**Mathematics Subject Classification** 90C15·90C25·49M27·90C

#### 1 Introduction

Motivated by application of generalized Benders decomposition (GBD) algorithms
to chance-constrained optimization (CCO) problems, we investigate in this work
modifications to the standard cutting-plane method (CPM) for minimizing an oracle-
provided convex nondifferentiable function over a combinatorial set. The main aim
of our investigation is to improve the practical performance of GBD by applying it
(actually, to the CPM that sits at its core) two techniques that have been success-
ful in related but different situations: _stabilization_ and _inexact subproblem solution_.
While these have sometimes been separately employed in the past, their simul-
taneous use involves a nontrivial interaction that had not been fully theoretically
investigated before. Since GBD is the most prominent application of our results we
mainly discuss them in that context. However, it has to be remarked that our the-
ory is completely general, and it applies to any situation where the minimization
of a convex (possibly nondifferentiable) function described by a standard (com-
putationally costly) first-order oracle is required, provided that the oracle can be
modified as to become an _informative on-demand inexact_ one, formally defined
below.
The observation that certain problems can be made much easier by temporarily fix-
ing a subset of variables, and that solving the thusly simplified problem may provide
information for improving the allocations of these fixed variables, was originally made
for linear problems [ 6 ]. It was also soon realized that it was particularly promising
for stochastic ones [ 64 ]. Since it basically relies on duality, it was readily gener-
alized to problems with some underlying convexity [ 35 ]; we refer to [ 11 , 30 ]fora
general overview. In a nutshell, GBD is a variant of Kelley’s cutting-plane method
[ 38 ] applied to the value function of the simplified problem. As it is well-known
that CPM suffers from serious computational issues, it is not surprising that meth-
ods aimed at improving the performances of GBD have been proposed. Several of
these have centered on the appropriate choice for the linearizations (Bender’s cuts)
used to construct the model, as it was quickly recognized that this can have a signif-
icant impact on (practical) convergence [ 39 ]. Different strategies have been proposed
for generating good linearizations [ 23 , 52 , 56 , 65 , 68 ], occasionally with a problem-
dependent flavor, e.g., based on duality theory [ 29 ] or on combinatorial arguments
[ 10 ].
We tackle the issue of efficiency of GBD from a different angle, with a two-
pronged approach. On the one hand we aim at decreasing the number of iterations
by _stabilizing_ the CPM. On the other hand we aim at decreasing the cost of each
iteration by allowing the corresponding subproblem to be solved only inexactly. That
the latter can be beneficial as well as it is intuitive was already recognized as early
as [ 69 ]. However, in the corresponding convergence analysis the sequence of accu-


Inexact stabilized Benders’ decomposition approaches... 639

racy parameters is chosen a priori to converge to zero, so it is not adaptive to the
actual needs of the algorithm. In [ 67 ] the idea is applied to the stochastic context,
developing a specific rule for that application that allows to entirely skip evalua-
tion of some subproblems (scenarios) at properly chosen iterations; this is shown
to combine the advantages of disaggregate and aggregate variants. However, the
approach is proposed for the level method only, and is tailored on the quite spe-
cific sum-function structure of two-stage stochastic programs. Besides, we consider
the interaction between inexact subproblem solution and both known forms of sta-
bilization of the CPM, namely, level and trust region/proximal. The combination of
the two devices requires a technically rather involved theoretical analysis in the con-
vex case; see [ 17 ] for the proximal stabilization, and [ 16 , 25 , 59 ] for the level one. It
may be worth mentioning that, in the convex case, inexact computations can also be
used in the context of fast gradient-like approaches [ 13 , 58 ], but these methods are not
easily adapted to the combinatorial setting. In our case, as it happened e.g., in [ 32 ],
finiteness helps tosimplifythearguments somewhat. Weexploit this tothoroughlyana-
lyze the rules for selecting the two parameters controlling the oracle—the _target_ and
the _accuracy_ —in order to devise methods with the weakest possible requirements.
This allows us to cover a large set of different rules for choosing them. Stabiliza-
tion of combinatorial Benders’ approaches has already been proposed (e.g., [ 1 , 53 ]),
but usually with exact oracles. The only previous work we are aware of where both
approaches are used simultaneously in the combinatorial case is [ 70 ], but only for
the level stabilization. Moreover, our convergence arguments are significantly more
refined.
It is worth remarking that, while we primarily discuss our results in the context
where the subproblems are convex ones, they also apply—and, indeed, are even more
likely to be significant—to the extensions of GBD where convexity is not present.
These require different techniques according to the nature of the non-convexity in the
subproblem: for instance, polyhedral techniques to approximate the convex hull [ 9 , 54 ]
or generalized “logic” duality (inference duality) [ 37 ] for integer linear subproblems,
linearizations techniques to iteratively refine the description of the convex envelope
for nonlinear nonconvex subproblems [ 41 ], or generalized concavity [ 71 ]. In all these
cases the subproblems are significantly harder to solve, which makes their approximate
solution all too relevant.
This work is organized as follows. Section 2 reviews (generalized) Benders’ decom-
position for binary nonlinear optimization problems, in order to set the stage for
our results. In Sect. 3 we revisit the CPM to solve the resulting binary problem,
extending it to handle oracles with on-demand accuracy. Two different stabilized
variants of the CPM are then discussed in Sect. 4. Section 5 discusses the appli-
cation of the developed techniques to a large set of randomly generated hybrid
robust and nonlinear CCO problems, which is in itself new. Those COO problems
can be recast as “monolithic” Mixed-Integer Second-Order Cone Problems (MIS-
OCP), and therefore solved with standard software; our tests show that under many
(although not all) conditions the new approaches significantly outperform the direct
use of general-purpose MINLP tools. Finally, Sect. 6 closes with some concluding
remarks.


640 W. van Ackooij et al.

#### 2 Generalized Benders decomposition

Consider the following Mixed-Integer Non-Linear Problem (MINLP)

```
min
```
##### {

```
f ( x ): G ( x )≤ Tz , z ∈ Z , x ∈ X
```
##### }

##### (1)

where _f_ :R _n_ →Rand _G_ :R _n_ →R _p_ are convex mappings, _X_ ⊆R _n_ is a convex
compact set, _T_ ∈R _p_ × _m_ , and _Z_ is some subset of{ 0 , 1 } _m_ which is “simple to describe”.
In ( 1 ), the binary _z_ act as _complicating variables_ , in that for fixed _z_ the problem reduces
to the convex Non-Linear Problem (NLP)

```
v( z ):=min
```
##### {

```
f ( x ): G ( x )≤ Tz , x ∈ X
```
##### }

##### , (2)

which is much easier to solve. This is the core idea behind generalized Benders’
decomposition (GBD): the problem is restated in terms of the complicating variables
only as
v∗:=min

##### {

```
v( z ): z ∈ Z
```
##### }

##### , (3)

using the _value function_ v:{ 0 , 1 } _m_ →R∪{∞}of the NLP ( 2 ). The optimal values of
( 1 ) and ( 3 ) coincide: if _z_ ∗solves the latter and _x_ ∗solves ( 2 ) with _z_ = _z_ ∗, then( _x_ ∗, _z_ ∗)
solves ( 1 ). We recall in the following lemma some known properties ofv.

**Lemma 1** _The mapping_ v _is proper, convex, and bounded from below. If for a given
z_ ∈ Dom(v)( 2 ) _satisfies some appropriate constraint qualification (e.g., Slater’s
condition) so that the set_ Λ( _z_ )⊂R+ _pof optimal Lagrange multipliers of the constraints_

_G_ ( _x_ )≤ _Tzin_ ( 2 ) _is nonempty, then_ ∂v( _z_ )=− _T_ TΛ( _z_ )_._

Note that other constraint qualification can be invoked in the Lemma apart from
Slater’s condition, such the QNCQ one. What is required is just that solving the convex
NLP ( 2 ) for fixed _z_ be equivalent to solving its dual, i.e., that after having computed
v( _z_ )an optimal dual solutionλ∗( _z_ )∈Λ( _z_ )is available that can be used to construct
a _subgradient_ forvin _z_. This brings us in the classical setting for NonSmooth Opti-
mization (NSO): an _oracle_ is available that provides function values and subgradients
for the function.
The simplest approach for NSO problems, irrespectively of _z_ being continuous or
discrete, is perhaps the _Cutting-Plane Method_ [ 38 ]. It relies on using the first-order
information provided by the oracle to construct a _model_ ofv. At a given iteration _k_ a
set of iterates{ _z_^1 ,..., _zk_ −^1 }has been generated and an index setO _k_ ⊂{ 1 ,..., _k_ − 1 }
gathers the points _zj_ whose oracle information(v _j_ =v( _zj_ ), w _j_ ∈∂v( _zj_ ))is (still)
available. The standard _cutting-plane model_ forvis

```
vˇ k ( z ):=max{v j +〈w j , z − zj 〉: j ∈O k }≤v( z ), (4)
```
wheretheinequalityfollowsfromconvexityofv.Eachoftheinequalitiesin ( 4 )is,inthe
parlance of Benders’ decomposition, an _optimality cut_. Minimizingvˇ _k_ over the feasible
set _Z_ providesalowerboundvlow _k_ forv∗andanewcandidatesolution _zk_ toproblem ( 3 ),
at which the oracle is to be called. Ifvˇ _k_ ( _zk_ )=v( _zk_ )then clearly _zk_ is optimal for ( 3 ),


Inexact stabilized Benders’ decomposition approaches... 641

asvlow _k_ =ˇv _k_ ( _zk_ )≤v∗≤v( _zk_ ); otherwise, the newly obtained function value (and
subgradient) changes the model, and one can iterate. This brief account is, in a nutshell,
the CPM applied to the solution of ( 3 ), i.e., the celebrated Benders’ decomposition
method applied to ( 1 ). Remarkably, this process does not depend on the fact that _Z_
is a continuous or discrete set, provided that one can efficiently minimizevˇ _k_ over _Z_.
Which is not to say that the process is identical for discrete and continuous feasible
sets, as a number of relevant details differ. For instance, if _zk_ +^1 ∈/Dom(v), i.e., if
subproblem ( 2 ) is infeasible, then _zk_ +^1 must be eliminated from the feasible set of ( 3 )
by adding a _feasibility cut_ to the set of constraints, which is usually obtained by means
of the _unbounded ray_ of the dual problem. In the binary setting, one can alternatively
use a simpler _no-good-type_ cut [ 12 ], along the lines of the _combinatorial_ Benders’
cuts proposed in [ 10 ]. In particular, let _S_ ( _z_ )={ _s_ : _zs_ = 0 }and _Sk_ = _S_ ( _zk_ ); then

```
∑
s ∈ Sk
```
```
zs ≥1(5)
```
is a feasibility cut that excludes the point _zk_ from the feasible set of ( 3 ). This corre-
sponds to restricting _Z_ to a subset _Zk_ +^1 ⊇Dom(v), or equivalently to force the model
vˇ _k_ + 1 to agree withvin _zk_ (vˇ _k_ ( _zk_ )<vˇ _k_ + 1 ( _zk_ )=v _k_ =+∞). Hence, together with the
indexsetO _k_ ofoptimalitycuts,oneonlyneedstokeeptheindexsetF _k_ ⊆{ 1 ,..., _k_ − 1 }
of feasibility cuts, and define the _master problem_

```
zk +^1 ∈arg min
```
##### {

```
vˇ k ( z ) : z ∈ Zk
```
##### }

##### ≡

##### ⎧

##### ⎪⎪

##### ⎨

##### ⎪⎪

##### ⎩

min _r_
s.t. _r_ ∑≥v _j_ +〈w _j_ , _z_ − _zj_ 〉 _j_ ∈O _k
s_ ∈ _Sjzs_ ≥^1 _j_ ∈F _k
z_ ∈ _Z_ , _r_ ∈R
(6)
It is easy to show that no optimal solution of ( 3 ) is excluded by feasibility cuts, and
that the CPM solves problem ( 3 ) [and, consequently, problem ( 1 )]. Note that ( 6 )is
unbounded below ifO _k_ =∅, i.e., no optimality cut has been generated yet. This will
surely happen for _k_ =0, and may happen even for a large _k_ (for instance, for all _k_ if
_Z_ ∗is empty). A simple way to avoid this issue is to add to ( 6 ) the single constraint
_r_ ≥0 in caseO _k_ =∅, making ( 6 ) a feasibility problem seeking just any point in _Zk_.
That constraint is removed as soon as an optimality cut is generated.

**Algorithm 1** Combinatorial Cutting-Plane Method (CCPM)

**Step 0.** (Initialization) LetδTol≥0 be the stopping tolerance.v 0 up←∞,O 1 =F 1 ←∅,and _k_ ←1.
**Step 1.** (Master) Find _zk_ by solving problem ( 6 ), and letvlow _k_ be its optimal value.
**Step 2.** (Stopping test)Δ _k_ ←vup _k_ − 1 −v _k_ low.IfΔ _k_ ≤δTol, stop: _z_ upis aδTol-optimal solution.
**Step 3.** (Oracle call) Solve ( 2 ) with _z_ replaced by _zk_.
–If( 2 ) is infeasible thenF _k_ + 1 ←F _k_ ∪{ _k_ },O _k_ + 1 ←O _k_ ,v _k_ up←vup _k_ − 1 andgotoStep4.

- Else, obtainv _k_ =v( _zk_ )and a subgradientw _k_ as in Lemma 1 ,F _k_ + 1 ←F _k_ andO _k_ + 1 ←O _k_ ∪{ _k_ }.
    vup _k_ ←min{v _k_ ,vup _k_ − 1 }.Ifv _k_ =vup _k_ then _z_ up← _zk_.
**Step 4.** (Loop) _k_ ← _k_ +1 and go to Step 1.


642 W. van Ackooij et al.

**Theorem 1** _Algorithm_ 1 _terminates after finitely many steps either returning a_ δTol _-
optimal solution to problem_ ( 3 ) _or proving that it is infeasible._

_Proof_ First, note that the algorithm cannot stop for _k_ =1:vup 0 =∞impliesΔ _k_ =
∞>δTol[vlow 1 =0 becauseO 1 =∅, cf. the added constraint _r_ ≥0in( 6 )]. Once _zk_
is determined in Step 1 andv _k_ is computed in Step 3, we have two cases. Ifv _k_ =∞,a
feasibility cut is added which excludes (at least) _zk_ from the feasible region for good.
As long as no feasible solution is found (i.e.,v _j_ =∞for all _j_ ≤ _k_ ) one keeps having
vlow _k_ = 0 ⇒ Δ _k_ =∞unless ( 6 ) is infeasible, in which casev _k_ low=v
up
_k_ =∞
and henceΔ _k_ =0: the algorithm stops, with _Z_ ⊆ _Zk_ =∅proving that ( 3 )is
empty. Because at each iteration _Zk_ strictly shrinks and _Z_ has a finite cardinality, after
finitely many steps either the algorithm establishes that ( 3 ) is infeasible or it generates
an optimality cut.
In this last case, from ( 6 ) we have thatv _k_ low=ˇv _k_ ( _zk_ )≥v _j_ +〈w _j_ , _zk_ − _zj_ 〉for all
_j_ ∈O _k_. Therefore,‖w _j_ ‖‖ _zj_ − _zk_ ‖≥〈w _j_ , _zj_ − _zk_ 〉≥v _j_ −vlow _k_ ≥v
up
_k_ − 1 −v

low
_k_ =Δ _k_ ,
where the last inequality comes from the fact thatv _k_ up− 1 ≤v _j_ for all _j_ ∈O _k_. Hence,

_zk_ differs from all the previously generated iterates as long as theΔ _k_ >0. Since _Z_
contains only finitely many points, the algorithm finitely stops. 

The CPM can therefore be used to solve ( 3 ) [and thus ( 1 )]; it can also be gener-
alized somewhat, see e.g., [ 66 ]. However, CPM is well-known to suffer from slow
convergence in practice. It is therefore attractive to mimic “more advanced” NSO
approaches, such as one of the several available variants of _bundle methods_ ,devel-
oped for the continuous case (e.g., [ 31 , 36 , 40 ] among many others). Adapting these
methods to the case where _Z_ is _combinatorial_ requires specific work; this is done in
Sect. 4 , whereas in the next one we introduce another mechanism that can improve
the performances of the approach.

#### 3 Extending the cutting-plane method: inexact oracles

Algorithm 1 has two potentially costly steps: Step 1, that requires solving a MI(N)LP,
and Step 3 that requires solving a (convex) NLP. A way to reduce the computational
burden in Step 3 is to only _approximately solve_ the NLP; in the parlance of [ 13 , 16 , 17 ]
this is employing an _inexact oracle_ for the value functionv. While we discuss this
with reference to GBD, the idea of inexact oracles clearly has wider application, as
( 10 ) will make apparent.
To illustrate what an inexact oracle is in the GBD case, we cast the oracle problem
( 2 ) in a Lagrangian setting. That is, for the vectorλof Lagrangian multipliers, the dual
function of ( 2 )

```
θ(λ):=min
```
##### {

```
f ( x )+〈λ, G ( x )〉: x ∈ X
```
##### }

```
−〈λ, Tz 〉 (7)
```
has the property thatθ(λ)≤v( _z_ )for eachλ∈R+ _p_. Hence, the Lagrangian dual

```
v( z )=max
```
##### {

```
θ(λ): λ∈R+ p
```
##### }

##### (8)


Inexact stabilized Benders’ decomposition approaches... 643

is (under appropriate constraint qualification, cf. Lemma 1 ) equivalent to ( 2 ). In fact,
in correspondence to the optimal solutionλ∗to ( 8 ) one can find an optimal solution
to _x_ ∗∈ _X_ to ( 7 ) (withλ=λ∗) such that _G_ ( _x_ ∗)≤ _Tz_ and〈λ∗, _G_ ( _x_ ∗)− _Tz_ 〉= 0
(complementary slackness); one then has _f_ ( _x_ ∗)≥v( _z_ )≥θ(λ∗)= _f_ ( _x_ ∗). _Approx-
imately_ solving ( 2 ) to any given accuracyεreduces to finding a feasible primal-dual
pair( _x_ ̄,λ) ̄ (i.e., _x_ ̄∈ _X_ , _G_ ( _x_ ̄)≤ _Tz_ , ̄λ≥0) such that( 0 ≤) _f_ ( _x_ ̄)−θ(λ) ̄ ≤ε.Every
conceivable algorithm for the problem has to provide both information in order to be
able to stop at a certified (approximately) optimal solution: the Lagrangian setting is
not the only possible one. When the form of _G_ (·)allows for an appropriate algebraic
dual, such as in the case of our experiments (cf. § 5 ), more complex dual feasibility
conditions can take the place of the minimization in ( 7 ) to ensure weak duality. Yet,
such a dual solution would compriseλas well as other dual variables for the other
constraints representing _x_ ∈ _X_. We will therefore not explicitly distinguish the two
cases, and we will stick with the Lagrangian notation for ease of discussion.
Algorithmically speaking, there are two different approaches that can yield useful
approximated oracles:

- _Dual approach_ : directly tackle problem ( 8 ) via some appropriate optimization
    algorithm; asθis most often nonsmooth, usually a NSO approach is required.
    This typically constructs a sequence{λ _h_ }of iterates with nondecreasing objective
    valueθ(λ _h_ )that approximatev( _z_ )from below. Any such algorithm eventually
    constructs a primal feasible (at least, up to some specified numerical tolerance)
    solution _x_ ̄such that _f_ ( _x_ ̄)−θ(λ _h_ )is appropriately small [ 27 ], thereby providing
    the stopping criterion. Although it did not turn out to be computationally effective
    in our specific application, the dual approach has indeed shown to be efficient
    in several cases to solve large-scale continuous [ 33 , 34 ] and combinatorial [ 61 ]
    programs.
- _Primal-dualapproach_ :underappropriateconditions,subproblem ( 2 )canbesolved
    by primal-dual interior-point methods (e.g., [ 5 ] among the many others). These
    typically construct a sequence of primal-dual pairs( _xh_ ,λ _h_ )which track the _central_
    _path_ towards the optimal primal-dual solutions( _x_ ∗,λ∗). Possibly after an initial
    infeasible phase, both solutions are _feasible_. In particular, it is well known (see for
    instance [ 7 , §11.2]) that every central point _xh_ ∈ _X_ yields a dual feasible point
    λ _h_. Hence, once againλ _h_ yields a lower bound on the optimal value, and _xh_ an
    upper bound; the algorithm stops when the two are suitably close.

In both cases, a sequence{λ _h_ }is produced, each one of which can be used to
construct _approximate linearizations_. In fact,v=θ(λ)≤v( _z_ )andw=− _T_ Tλare
such that
v(·)≥v+〈w,·− _z_ 〉. (9)

Note that ( 9 ) only hinges on weak duality, and therefore does not require constraint
qualification. Thus, often a sequence of candidates for the next linearization in Algo-
rithm 1 is available, provided that one allows them not to be tight, i.e.,v<v( _z_ ).
Indeed, _proving_ that a linearization _is_ tight, or at least that the errorε=v( _z_ )−vis
“small”, requires the entirely different information _xh_ that provides an _upper bound_
v ̄≥v( _z_ ). In this context, taking a leaf from [ 16 ] and (separately) [ 17 ], we define an


644 W. van Ackooij et al.

_informative on-demand inexact oracle_ as any procedure that, given _z_ ∈ _Z_ ,a _descent
target_ tar∈R∪{−∞,+∞}and a _desired accuracy_ ε≥0, returns:

```
⎡
⎣
```
```
i) as function information, two valuesvandv ̄such thatv≤v( z )≤ ̄v
ii) as first-order information, a vectorw∈R p such that (9) holds
under the condition that, ifv≤tar thenv ̄−v≤ε
```
##### (10)

Setting tar=∞andε=0 gives the standard exact oracle forv. We will aim at
defining the weakest possible conditions on the way to set the two parameters that still
guarantee convergence of the algorithm. Intuitively, this means that we want to keep
tar “small”, andε“large”. Indeed, whenv>tar (which, for instance, surely holds
if tar=−∞), not much at all is required fromv ̄:v ̄=∞is a perfectly acceptable
answer, corresponding to “no feasible solution to ( 3 ) has been found yet” (which may
be just because no feasible solution exists). Note that the oracle must be able to detect
when ( 2 ) is infeasible, so that a feasibility cut is added to ( 6 ). This is signaled by
returningv=∞(which obviously impliesv ̄=∞), incidentally making the value of
tar irrelevant. Usually, in this casewshould be required to describe a valid inequality
forDom(v)that cuts away the current _z_ , which requires the oracle to find an _unbounded
feasible descent direction_ for ( 8 ). Although doing so may be beneficial in practice, in
our setting we can use ( 5 ) instead whenv=∞, disregardingw; we therefore assume
this, just in order to simplify the notation. Moreover, setting a finite target tar<∞
actually allows us to dispense with feasibility cuts altogether. In fact, suppose that
( 2 ) is infeasible for the fixed _z_ : since _X_ in ( 1 ) is compact, we can argue by using[ 36 ,
Proposition 2.4.1, Chapter XII] that ( 7 ) is unbounded. Thus, any dual or primal-dual
algorithm applied to its solution will typically construct a sequence{λ _h_ }such that
θ(λ _h_ )→∞, while (clearly) never constructing any feasible _xh_. It is therefore easy
to add the simple checkθ(λ _h_ )>tar to the oracle, and stop it immediately when this
happens (or at any moment thereafter). If tar<∞this will typically ensure finite
termination of the oracle even in the unbounded case (which is not trivial for pure
dual approaches), while still ensuring by weak duality thatv=θ(λ _h_ )>tar and
w=− _T_ Tλ _h_ provide the required information (whatever the value ofv ̄, e.g.,v ̄=∞).
This yields the following simple but useful remark:

_Remark 1_ (Valid cuts for functionv)Given _z_ ∈ _Z_ and a finite target tar<∞as
input, an oracle ( 10 ) solving ( 7 ) can provide a _valid cut_ wforvindependently of
whether or not ( 2 ) is feasible for _z_. Hence,F _k_ in ( 6 ) can remain empty for all _k_ even
if _Z_ \Dom(v)=∅.

As we will see, this information is enough for the CPM, making it irrelevant to
“formally prove” if ( 2 ) is or not feasible. This shows that when inexact computations
are allowed for, and a finite target is specified, the separation between feasibility and
optimality cuts blurs somewhat.

**3.1 A cutting-plane method for inexact oracles**

We now adapt Algorithm 1 for dealing with the (informative, on-demand) inexact
oracle ( 10 ). We start by providing the necessary change for defining the next iterate:


Inexact stabilized Benders’ decomposition approaches... 645

having the (inexact) oracle information at hands, the inexact cutting-plane approxi-
mation forvis just

```
vˇ k ( z ):=max
```
##### {

```
v j +〈w j , z − zj 〉: j ∈O k }, (11)
```
whereO _k_ is as in Algorithm 1. Since ( 9 ) yieldsvˇ _k_ ( _z_ )≤v( _z_ )for all _z_ ∈ _Z_ , minimizing
vˇ _k_ still provides a valid global lower bound over the optimal value of ( 3 ). The algorithm
is then given below.

**Algorithm 2** Inexact Combinatorial Cutting-Plane Algorithm (ICCPM)
**Step 0.** (Initialization) As in Step 0 of Algorithm 1. In addition, chooseε 1 ≥0andγ>0.
**Step 1.** (Master) As in Step 1 of Algorithm 1 , but with model given in ( 11 ).
**Step 2.** (Stopping test) As in Step 2 of Algorithm 1.
**Step 3.** (Oracle call) Choose tar _k_ , send the triple( _zk_ ,ε _k_ ,tar _k_ )to oracle ( 10 ), receivev _k_ ,v ̄ _k_ ,andw _k_.

- If the subproblem is infeasible (v _k_ =∞), proceed as in Algorithm 1.
- Otherwise,F _k_ + 1 ←F _k_ ,O _k_ + 1 ←O _k_ ∪{ _k_ },vup _k_ ←min{ ̄v _k_ ,vup _k_ − 1 }.Ifv ̄ _k_ =v _k_ upthen _z_ up← _zk_.
    **Step 3.1** (Accuracy control) Ifv _k_ ≤vlow _k_ +γthen chooseε _k_ + 1 ∈[ 0 ,ε _k_ ), otherwise choose
    ε _k_ + 1 ∈[ 0 ,∞)arbitrarily.
**Step 4.** (Loop) _k_ ← _k_ +1 and go to Step 1.

The algorithm is essentially the same as the original one with the obvious modifi-
cations regarding upper and lower estimates. The only real novel mechanisms are the
ones regarding the handling of the target tar _k_ and of the accuracyε _k_. These are stated
in a general form that allows many different implementations. The crucial requirement
is that, eventually, iterations wherev _k_ ≤tar _kand_ ε _k_ ≤δTolare performed. With this
simple device the algorithm can be proven to converge.

**Theorem 2** _Assume that the choice of tarkin Step 3 and that of_ ε _kin Step 3.1 are
implemented in such a way that the following properties holds: (i) if_ vup _k_ =∞ _, then
tark_ =∞ _, and (ii) if_ ε _kis reduced in a sequence of_ consecutive _iterations, then
tark_ ≥vlow _k_ +γ _and_ ε _k_ ≤δTol _holds for k large enough. Under these conditions,
Algorithm_ 2 _finitely terminates with either a_ δTol _-optimal solution to problem_ ( 3 ) _or a
proof that it is infeasible._

_Proof_ We will begin by showing that the algorithm establishes infeasibility of problem
( 3 ) after finitely many iterations. Sincev( _z_ )=∞for all _z_ ∈ _Z_ , the algorithm can
never producev ̄ _j_ <∞, and as a consequencevup _k_ =∞for all _k_. Then, by assumption
i) tar _k_ =∞, which means that oracle ( 10 ) must necessarily returnv _k_ =∞for all _k_.
The algorithm behaves exactly as the original one and the proof is completed.
Assume now thatv _k_ <∞at least once: the problem admits a feasible solution.
In fact, at the first such _k_ one has tar _k_ =∞, and thereforev _k_ <tar _k_ :by( 10 ),
∞>v _k_ +ε _k_ ≥ ̄v _k_ ≥v( _z_ ), hence at least one feasible point exists andvup _k_ <∞
as well. Since the total number of possible “infeasible” iterations (withv _k_ =∞)is
finite, for the sake of finite convergence arguments we can assume that none happens
after iteration _k_. Note that, due to Remark 1 ,F _k_ may be empty even if some _zk_ really
was infeasible. Hence, we can assume that a feasible _zk_ is determined in Step 1: from
( 6 ) withvˇ _k_ given in ( 11 ) we have thatv _k_ low=ˇv _k_ ( _zk_ )≥v _j_ +〈w _j_ , _zk_ − _zj_ 〉for all


646 W. van Ackooij et al.

```
j ∈O k , whence by the Cauchy-Schwarz inequality
‖w j ‖‖ zj − zk ‖≥〈w j , zj − zk 〉≥v j −v k low. (12)
```
For all the iterations _j_ ∈O _k_ wherev _j_ >vlow _k_ ,( 12 )gives‖ _zj_ − _zk_ ‖>0, i.e.,
_zk_ = _zj_ as in the proof of Theorem 1. If one could prove thatv _j_ >v _k_ lowholds always,
the finiteness of _Z_ would complete the proof.
However, since we have two different valuesv ̄ _j_ ≥v( _zj_ )≥v _j_ , it may well happen
thatv ̄ _j_ −vlow _k_ ≥Δ _k_ >δTolwhile one is performing a _tight_ iteration:T _k_ ={ _j_ ∈
O _k_ :v _j_ ≤vlow _k_ }.For _j_ ∈T _k_ we could have _zk_ = _zj_ , which exposes the algorithm
to the danger of cycling. Note that ( 12 ) cannot be used with _j_ = _k_ because _k_ ∈/O _k_ :
the linearization corresponding to _zk_ was not in the definition ofvˇ _k_ when the master
problem was solved. That is, one can repeat the same iterate (say, _zk_ +^1 = _zk_ ), and in
this casev _k_ >v _k_ lowonly implies that (possibly, but not even necessarily)v _k_ low+ 1 >v _k_ low.
Thus, the lower bound onv∗may increase arbitrarily slowly; and similarly for the
upper bound. In fact, when condition i) is no longer in effect, and until condition
ii) is triggered, tar _k_ can be chosen arbitrarily (possibly, tar _k_ =−∞), which means
that one can havev ̄ _k_ =∞(although, at this point of the argument,v _k_ up<∞).
Thus, we need to distinguish among the iterations that are _in-target_ , i.e., belong to
I _k_ ={ _j_ ≤ _k_ :v _j_ ≤tar _j_ }, and those that are not. The rationale is that only for
_j_ ∈I _k_ the value ofv ̄ _j_ actually is “meaningful”, while for _j_ ∈/I _k_ nothing much can
be said,v ̄ _j_ =∞being possible. The relevant set of iterations is thenT _k_ ′=T _k_ ∩I _k_ ,
and for these iterations one has

```
v
up
k − 1 =min{ ̄v
```
```
j : j ∈O
k }≤min{ ̄v
j : j ∈T′
k }≤
≤min{v j +ε j : j ∈T k ′}≤v k low+min{ε j : j ∈T k ′}, (13)
```
where the first inequality comes fromT _k_ ′⊆O _k_ , the second inequality comes from
the assumption on oracle ( 10 ) when _j_ ∈T _k_ ′⊆I _k_ , and the third one comes from the
definition ofT _k_. Note thatT _k_ ′=∅may happen, making ( 13 ) useless, because either
no tight iteration has been performed, or none of them is an in-target one, the latter
clearly depending on how tar _k_ is chosen.
Let us now assume by contradiction that the algorithm does not finitely terminate.
For iteration _k_ , eitherv _k_ ≤v _k_ low+γ,orv _k_ >v _k_ low+γ. Let us first consider the case
whenv _k_ >v _k_ low+γoccurs infinitely many times (not necessarily consecutively).
By taking subsequences if necessary we can assume that the condition actually holds
at every _k_. Since the set _Z_ is finite and the algorithm is assumed to loop forever,
eventually it necessarily re-generates iterates that have already been found earlier.
Hence, consider two iterates _h_ > _k_ such that _zh_ = _zk_ : one has

```
vlow h =ˇv h ( zh )=max{v j +
```
##### 〈

```
w j , zh − zj
```
##### 〉

```
: j ∈O h }≥v k +
```
##### 〈

```
w j , zh − zk
```
##### 〉

```
=v k >vlow k +γ (14)
```
wherethefirst equalities arejust thefact that _zh_ is optimal forvˇ _h_ , theleftmost inequality
comes from _k_ ∈O _h_ , and the following equality comes from _zh_ = _zk_. Actually, if the
algorithm runs forever then at least one of the iterates _zk_ has to be generated _infinitely_


Inexact stabilized Benders’ decomposition approaches... 647

_many_ times. Sinceγ>0, repeating ( 14 ) on the corresponding sub-sequence proves
thatvlow _k_ →∞as _k_ →∞, which contradictsvlow _k_ ≤v∗≤v _k_ up<∞.
Hence,v _k_ >vlow _k_ +γcan only occur finitely many times: if the algorithm runs
forever, then eventually a long enough sequence of _consecutive_ iterations whereε _k_ is
reduced has to be performed. The assumption ii) on the choice of the oracle parameters
now applies: eventually, tar _k_ ≥v _k_ low+γ(≥v _k_ ⇒ _k_ ∈I _k_ )andε _k_ ≤δTol.This
ensures thatv ̄ _k_ −v _k_ ≤ε _k_ ≤δTol: eventually, iterates become “accurate enough”. Now,
consider two (large enough, hence accurate enough) iterations _k_ > _j_ :if _j_ ∈T _k_ , then by
( 13 ) one would haveΔ _k_ =v
up
_k_ − 1 −v

```
low
k ≤ ̄v
```
```
j −v j ≤ε j ≤δTol(obviously,vup
k − 1 ≤ ̄v
```
```
j ),
```
contradicting the fact that the algorithm does not stop. One should therefore have that
v _j_ >vlow _k_ always holds; however, in this case ( 12 ) shows that _zk_ = _zj_ for all the
infinitely many _k_ > _j_ , which contradicts finiteness of _Z_. Altogether, we have shown
that the algorithm must therefore stop after finitely many iterations. 

We now discuss why, save for a few minor twists, the conditions onε _k_ and tar _k_ in
Theorem 2 appear to be basically the weakest possible ones.

```
–Havingtar k =∞as long asvup k =∞, as required by condition i), may seem a
harsh request, but it appears to be unavoidable in order to account for the case where
( 1 ) is infeasible. In fact, if the algorithm were allowed to set a finite target tar 1 , then
the oracle may return any finitev 1 >tar 1 together withw 1 =0 (and, obviously,
v ̄ 1 =∞). Then, at the next iteration the algorithm would havevlow 2 =v 1 >tar 1 ,
and may well produce z^2 = z^1. Hence, an infinite sequence of iterations may start
where zk = z^1 andv k low→∞, but the algorithm never stops becausevup k =∞
as well (andε k is finite). That is, the algorithm would spend all the time trying
to computev( z^1 )=∞by finitely approximating it from below. Thus, setting
tar k =∞is required until the problem is proven feasible.
```
- A similar observation justifies the need for the constantγ>0, both in the defin-
    ition of the threshold for reducingε _k_ in Step 3.1, and in the condition on tar _k_ .In
    fact, if one would require decrease only ifv _k_ ≤v _k_ low, then a sequence of iterations
    all producing the same _z_ may ensue wherev _k_ >vlow _k_ , but “only very slightly so”.
    Hence, while one may havevlow _k_ + 1 >vlow _k_ , the increment may be vanishingly small,
    and sinceε _k_ would not be decreased, ultimatelyvlow _k_ may never become close
    enough tovup _k_ (a similar argument, in the continuous case, is made in [ 13 , Obser-
    vation2.7]). Analogously, if one would set tar _k_ =vlow _k_ , then again a sequence of
    iterations producing the same _zk_ could be performed wherev _k_ may be fractionally
    larger thanvlow _k_. While this would force the decrease ofε _k_ , none of these iterations
    would be in-target, which would not allow us to use ( 13 ). That is, whilev _k_ lowmay
    indeed be converging tov∗, oracle ( 10 ) may never report a close enough upper
    bound; in fact, one may well havev ̄ _k_ =∞for all _k_.
A few minor improvements could be added:
- It is possible to weaken somewhat condition i) by requiring that tar _k_ =∞holds
    _after finitely many iterations where_ vup _k_ =∞, whatever mechanism be used to
    ensure this.
- Running the algorithm with fixedε _k_ =δToland tar _k_ =∞does not require anyγ
    (which is somehow obvious since that parameter only influences the choice ofε _k_


648 W. van Ackooij et al.

```
and tar k ). An alternative way to get the same result would be to ask thatε k =δTol
and tar k is “large enough” (e.g., tar k =v k up− 1 ) eventually, with some mechanism,
say a fixed number of iterations, to ensure it.
```
- A fixedγ>0 is only the simplest possible option. All one needs is that each
    time when the same iterate is repeated,vlow _k_ has “significantly increased”. As
    customary in other settings (e.g., [ 13 ]), one may obtain this by ensuring that for
    every subsequence of the sequence{γ _k_ }the series diverges (even if, say,γ _k_ →0).
    Note that this allowsγ _k_ =0 to happen, but only finitely many times: eventually
    γ _k_ >0 has to happen, albeit it does not need to be bounded away from zero.
- Alternatively, forδTol >0 the simple choiceγ _k_ =αΔ _k_ for some fixedα> 0
    obviously works.
- A slightly weaker version of Step 3.1 is possible whereε _k_ is only reduced if
    v _k_ ≤v _k_ low+γ _kand zkcoincides with a previously generated tight iterate_ ,for
    this cannot happen infinitely many times. However, the check that _zk_ = _zj_ for all
       _j_ ∈T _k_ could ultimately be costly in both time and memory.

**3.2 Accuracy handling in the ICCPM**

Theorem 2 provides ample scope for constructing different strategies to handle the
oracle parametersε _k_ and tar _k_ , besides the obvious one whereε _k_ =δToland tar _k_ =∞
throughout. In fact, intuitively having high-accuracy computations at the initial phases
of the algorithm is unnecessary, and that starting with a “large”ε _k_ and a “low” tar _k_
would be better. This has in fact been proven true computationally in the continuous
case [ 16 , 62 , 67 ]. There are several possible ways of doing it:

- One may choose (even a-priori) a sequence{ε _k_ }→δTol(finitely), while still
    keeping tar _k_ =∞. This again works, but it is clearly non adaptive.
- Keepingε _k_ “large” and tar _k_ =−∞is possible for most of the time. That is, if the
    condition at Step 3.1 is not satisfied, then one can immediately resetε _k_ + 1 to some
    “large” value (say,ε 1 ), and leave it there until forced to reduce it. This generalizes
    the rule presented in [ 17 , Algorithm5.4] for the continuous case.

All this shows that there is no need to solve the oracle with even a modicum of
accuracy,bothfromtheprimalandthedualviewpoint,unlessthealgorithmhasreached
the global optimum of the current modelvˇ; only then the accuracy has to be increased.
Said otherwise, the function ideally only have to be computed with _provable_ accuracy
δTolonly at the optimal solution _z_ ∗. Insisting thatε _k_ and tar _k_ are kept “coarse” for most
of the iterations is therefore possible, and likely justified in cases where the oracle cost
is a preponderant fraction of the overall computational burden.
However, there can be cases where the master problem cost may be non-negligible,
in particular because it is a combinatorial program. Furthermore, working with a
“coarse” model ofv—even coarser then the ordinary cutting-plane model arguably
already is—may increase the number of iterations, and therefore finally prove not com-
putationally convenient. Hence, more “eager” mechanisms for handling the accuracy
parameters may be preferable, which can be obtained in several ways.


Inexact stabilized Benders’ decomposition approaches... 649

- For instance, tar _k_ ←max{v _k_ up,vlow _k_ +γ _k_ }may make sense. Basically, one is
    trying to improve on the upper bound, although doing so early on has only a
    limited effect on the algorithm behavior (the only role ofvup _k_ being in the stopping
    condition).
- Condition ii) can be ensured by choosingα ∈ ( 0 , 1 )and settingε _k_ + 1 =
    max{δTol,min{αΔ _k_ ,ε _k_ }}. This is for instance the strategy that has been
    employed in [ 67 , 69 ], and it is intuitively attractive becauseε _k_ is nonincreasing
    with the iterations and converges toδTol. A (minor) issue arises whenδTol=0,
    since the algorithm may not finitely terminate: basically, once identified the opti-
    mal solution _z_ ∗one would keep on calling the oracle on _z_ ∗infinitely many times,
    with a sequence ofε _k_ (quickly) converging to zero, but never really reaching it.
    This issue is easily resolved by settingε _k_ ←δTolwhenε _k_ has reached a sufficiently
    small value.
- The analysis in Theorem 2 directly suggests to chooseε _k_ + 1 <min{ε _j_ : _j_ ∈T _k_ ′},
    with some mechanism ensuringε _k_ ≤δToleventually. This has the advantage that
    the right-hand side of the equation is∞whileT _k_ ′=∅, so one starts to decrease
    ε _k_ only when tight iterates are indeed performed.
- The above mechanisms still rigidly alternates master problem and subproblem
    computations. If the master problem cost is significant, when the oracle error is
    found to be excessive one may rather prefer to avoid the computation of _zk_ +^1
    (which may well produce _zk_ once again) and directly re-compute the function
    with increased accuracy. A potential advantage of this eager mechanism is that,
    while the oracle is still called several times on the same iterate _zk_ , (at least, some
    of) the calls happen _consecutively_. This means that _warm starts_ can be used in the
    oracle making a sequence of calls withε _k_ decreasing up to some ̄εnot significantly
    more costly than a single call with accuracy ̄ε.

Thus, several strategies exist for handling the accuracy in the original CPM. In the
next paragraph we will discuss how these can be adapted when stabilized versions of
the approach are employed.

#### 4 Extending the cutting-plane method: stabilization

In this section we explore a different strategy to improve the performances of the CPM:
rather than decreasing the iteration cost, we aim at requiring fewer iterations. For this
we propose two different _stabilization techniques_ which, in analogy with what has
been done for the continuous case, have the potential of improving the quality of the
first-order information by decreasing the _instability_ of the CPM, i.e., the fact that two
subsequent iterates can be “very far apart”. This has been shown to be very beneficial
(e.g.,[ 3 , 32 , 53 ]amongthemanyothers)fortworeasons:fewersubproblemsaresolved,
and therefore master problems of smaller sizes need be solved. Also, stabilization
involves modifications of the master problem, which may (or may not) lead to an even
further reduction of its computational cost. It has also been reported (cf. [ 70 ]inthe
discrete setting and [ 62 ] in the continuous one) that stabilization in GBD improves
feasibility of master iterates, thus reducing the number of feasibility cuts.


650 W. van Ackooij et al.

**4.1 Trust-region stabilization**

The most widespread stabilization technique in the continuous setting is the _proximal_
one, whereby a term is added to the objective function of the master problem to
discourage the next iterate to be “far” from one properly chosen point (say, the best
iterate found so far). In our setting, we find it more appropriate to use trust-regions
[ 45 ]; note that in the continuous case the two approaches are in fact “basically the
same” [ 3 , 31 ]. For this we modify the master problem ( 6 ) by selecting one _stability
centerz_ ˆ _k_ , which can initially be thought of as being the current best iterate, and the
radiusB _k_ ≥1 of the current trust region. Then, we restrict the feasible region of ( 6 )
as
_zk_ ∈arg min

##### {

```
vˇ k ( z ): z ∈ Zk ,‖ z −ˆ zk ‖ 1 ≤B k
```
##### }

##### , (15)

which just amounts at adding the single linear _local branching constraint_ [ 28 ]

##### ∑

```
s :ˆ zks = 1
```
```
( 1 − zs )+
```
##### ∑

```
s :ˆ zks = 0
```
```
zi ≤B k.
```
A similar approach was used in [ 46 , 53 ], both with exact oracles. In addition to consider
inexact oracles, our analysis is significantly more refined. The effect of the trust region
is to force the next iterate to lie in a _neighbourhood_ of the stability center where only
at mostB _k_ variables can change their state w.r.t. the one they have inˆ _zk_. A benefit of
this choice is that the complement of that neighbourhood,‖ _z_ −ˆ _zk_ ‖ 1 ≥B _k_ +1, can
be represented by an analogous linear constraint. Note that in a convex setting any
such set would be non-convex, thus making the master problem significantly harder
to solve if one were to add it; yet, in our case this is just another linear constraint in
a(n already nonconvex) MILP. It is therefore reasonable to add these constraints, and
we will denote byR _k_ the set of iterations at which the _reverse region_ constraints are
added. We remark in passing that, since for _zs_ ∈{ 0 , 1 }one has _z_^2 _s_ = _zs_ ,also‖ _z_ −ˆ _zk_ ‖^22
has a similar linear expression (this is in fact used in §4.2). The ICCPM is modified
as follows:

**Algorithm 3** Trust Region Inexact Combinatorial Cutting-Plane Algorithm (TRIC-
CPM)

**Step 0.** (Initialization) As in Step 0 of Algorithm 2 ,plusR 1 ←∅,vˆ^1 =∞, chooseB 1 ≥1,β>0and
ˆ _z_^1 ∈ _Z_ arbitrarily.
**Step 1.** (Master) As in Step 1 of Algorithm 2 except using ( 15 ).
**Step 2.** (Stopping test)Δ _k_ ←vup _k_ − 1 −vlow _k_ .IfΔ _k_ >δTolthenR _k_ + 1 ←R _k_ ,B _k_ + 1 ←B _k_ and go to
Step 3.
IfB _k_ = _m_ then stop: _z_ upis aδTol-optimal solution. Otherwise,R _k_ + 1 ←R _k_ ∪{ _k_ }and choose
B _k_ + 1 ∈(B _k_ , _m_ ].
**Step 3.** (Oracle call) As in Step 3 of Algorithm 2 , except add the following before Step 3.1:
Ifv ̄ _k_ ≤ˆv _k_ −βthenˆ _zk_ +^1 ← _zk_ ,vˆ _k_ + 1 ← ̄v _k_ , chooseε _k_ + 1 ∈[ 0 ,∞)arbitrarily and go to Step 4.
Otherwiseˆ _zk_ +^1 ←ˆ _zk_ ,vˆ _k_ + 1 ←ˆv _k_ and proceed to Step 3.1.
**Step 4.** (Loop) _k_ ← _k_ +1 and go to Step 1.


Inexact stabilized Benders’ decomposition approaches... 651

A few remarks on the algorithm are useful. The initial stability centerˆ _z_^1 may or
may not be feasible; to be on the safe side, by initializingvˆ 1 =∞we assume it is
not. If a feasible iterate _zk_ is produced, the new mechanism at Step 3 ensures that the
stability center is moved to the feasible point. This is called a “serious step” (SS) in
the parlance of bundle methods. Note that the initial master problem, not really having
an objective function, may well return _z_^1 =ˆ _z_^1. In this case one may be doing a “fake”
SS where the stability center does not really change, while its objective function value
does. Adding the reverse region constraint whenB _k_ is increased in Step 2 is not strictly
necessary, but it is cheap and ensures that iterates in the previous neighbourhood are no
longer produced. Reducing the feasible region of ( 15 ), this hopefully makes it easier
to solve.

**Theorem 3** _Under the assumptions of Theorem_ 2 _, Algorithm_ 3 _terminates after finitely
many steps with either a_ δTol _-optimal solution to problem_ ( 3 ) _or a proof that this
problem is infeasible._

_Proof_ The analysis of Theorem 2 applied to a fixed stability center, and therefore to
the fixed (finite) feasible region _Z_ ∩(‖·−ˆ _z_ ‖ 1 ≤B), shows that if no SS is done,
eventually the stopping condition at Step 2 is triggered. If this happens butB< _m_ ,
then _local_ δTol-optimality of _z_ up(which, as we shall see later on, is _not_ necessarily
the same as _z_ ˆ) in the current neighbourhood has been proven. If the master problem
were convex this would be the same as global optimality, but since this is not the case
we need to increaseBup until eventually the whole _Z_ is covered; obviously, this can
happen only a finite number of times. WhenB _k_ = _m_ , global optimality is ensured;
in particular, if no feasible solution has been found throughout, then ( 3 ) clearly is
infeasible.
Hence,wemustprovethatchanging _z_ ˆ _k_ —performingaSS—canonlyhappenfinitely
many times. Note that a SS do not trigger a forced decrease ofε _k_ (although it is
possible to reduce it), so we cannot rely on that mechanism to prove finite convergence.
However, sinceβ>0, a SS can only be declared when a “significant” decrease is
obtained, which can only happen finitely many times. 

Again, the assumptions on the SS mechanism are (almost) minimal. Indeed, with
β=0 it would be possible that the same iterate _z_ is produced infinitely many times,
eachtimetheoracleprovidingavanishinglysmallervalueforv ̄andthereforetriggering
a“fake”SS(whereˆ _z_ is not actually changed). This in itself would not impair finite
convergence, but an even more devious case exists: the one where _z_ ˆ“cycles through”
a set of iterates (possibly with the very same, and optimal, value ofv( _z_ )), each time
fractionally reducingvupbut never really producing the true value:β>0 rules this
out. Similarly to the remark concerningγin ICCPM: a fixed valueβ>0 is the easiest
choice. A non-summable sequence{β _k_ }or, forδTol>0,β _k_ =αΔ _k_ forα∈( 0 , 1 ]are
other options that work as well. We remark in passing that the rules deciding when a SS
has to be declared are crucial in the continuous case when dealing with inexact oracles
[ 17 ]. It is also apparent that the analysis extends to the “eager” accuracy control of
§3.1.
A relevant practical detail is howBis increased in Step 2 when local optimality
is detected. The simple ruleB _k_ + 1 ←B _k_ +1 would work, but previous experience


652 W. van Ackooij et al.

(in the exact setting) [ 1 ] suggests that this is not computationally effective: once a
local optima for a “small” region has been found, it is usually best to try to prove its
global optimality. Therefore, in our computational experiences we have used (without
any significant amount of tuning) the simple approach whereBis updated by moving
through a restricted set of sizes, as described in §5.3. Similarly to what happens for
ε _k_ and tar _k_ , it is possible to resetBto a “small” value each time a SS is computed,
since this only happens finitely many times. This differs from [ 53 ] that employs the
trust-region constraint only in the initial iterations.

**4.2 Level stabilization**

In this section we explore _level stabilization_ , which is, even in the continuous case,
significantly different to analyze [ 40 ]. Continuous level bundle methods have been
analyzed in the inexact case in [ 16 ], and discrete level bundle methods have been
analyzed in the exact case in [ 15 ]; to the best of our knowledge, apart from the recent
[ 70 ] this is the first work investigating a discrete level bundle method with inexact
oracles. Our results are more general, as we deal with oracles with on-demand (as
opposed to fixed) accuracy, and the hypotheses we require to establish convergence
are significantly weaker.
Level stabilization centers on defining a _level parameter_ v _k_ lev∈(vlow _k_ ,vup _k_ − 1 )and
the _level set_
Z _k_ :=

##### {

```
z ∈ Zk :ˇv k ( z )≤v k lev
```
##### }

##### . (16)

The next iterate _zk_ is then chosen inZ _k_ , whenever it is nonempty. IfZ _k_ =∅, then
vlow _k_ <vlev _k_ ≤v∗, i.e., a lower bound on the optimal value that is _strictly better_ then
vlow _k_ has been generated. As a result, the current lower boundv _k_ lowcan be updated by
the simple rulev _k_ low←vlev _k_.
Choosing a new iterate inZ _k_ can be done in several ways. A good strategy is to
choose _zk_ by satisfying some criterion of proximity with respect to a given stability
center _z_ ˆ _k_ ∈{ 0 , 1 } _m_ , which in this case may not even belong to _Z_. That is, we find
_zk_ by minimizing a convex _stability function_ φ(·; ˆ _zk_ )overZ _k_ ; in the continuous case,
the most common choice forφisφ=‖·‖^22. In our setting it may be natural to

take the 
1 or (^) ∞norms instead, as done in [ 15 ], since this leads to a MILP master
problem. Yet, as already mentioned, for binary variables the Euclidean norm also is
linear:‖ _z_ −ˆ _zk_ ‖^22 =〈^12 **1** −ˆ _zk_ , _z_ 〉+‖ˆ _zk_ ‖^22 , **1** being the all-one vector of appropriate
dimension. Taking the 
2 norm as a stability function leads to
_zk_ ∈arg min

##### {

```
φ( z ;ˆ zk ): z ∈Z k
```
##### }

##### ≡

##### ⎧

##### ⎨

##### ⎩

```
min〈^12 1 −ˆ zk , z 〉
s.t. v j +〈w j , z − zj 〉≤vlev k j ∈O k
z ∈ Zk.
(17)
The level version of the ICCPM then reads as follows:
```

Inexact stabilized Benders’ decomposition approaches... 653

**Algorithm 4** Level Inexact Combinatorial Cutting-Plane Algorithm (LICCPM)

**Step 0.** (Initialization) Run Algorithm 2 with the added condition in Step 2: stop (also) ifv ̄ _k_ =vup _k_ <∞.
IfΔ _k_ ≤δTolthen terminate.
**Step 1.** (Stopping test) As in Step 2 of Algorithm 2.
**Step 2.** (Master) Choose arbitrarilyˆ _zk_ ∈{ 0 , 1 } _m_. Choosevlev _k_ ∈[vlow _k_ ,v _k_ up− 1 −δTol).Solve( 17 ): if it is
infeasible thenv _k_ up←vup _k_ − 1 , choosev _k_ low∈[v _k_ lev,v∗]and go to Step 4, else _zk_ is available.
**Step 3.** (Oracle call) Choose tar _k_. Send the triple( _zk_ ,ε _k_ ,tar _k_ )to oracle ( 10 ), receivev _k_ ,v ̄ _k_ ,andw _k_.
–Ifv _k_ =∞then proceed as in Algorithm 1.

- Otherwise,F _k_ + 1 ←F _k_ ,O _k_ + 1 ←O _k_ ∪{ _k_ },vup _k_ ←min{ ̄v _k_ ,vup _k_ − 1 }.Ifv ̄ _k_ =v _k_ upthen _z_ up← _zk_.
    **Step 3.1** (Accuracy control) Ifv _k_ ≤v _k_ levthen chooseε _k_ + 1 ∈[ 0 ,ε _k_ ), otherwise chooseε _k_ + 1 ∈
    [ 0 ,∞)arbitrarily.
**Step 4.** vlow _k_ + 1 ←vlow _k_ , _k_ ← _k_ +1 and go to Step 1.

```
As usual, a few remarks on the algorithm can be useful:
```
- The level master problem ( 17 ) does not need any trick, as ( 6 ) and ( 15 ) do, when
    O _k_ =∅. In fact, as long as this happens, the value ofvlev _k_ is completely irrelevant:
    Z _k_ = _Zk_ , and one is seeking the nearest feasible point toˆ _zk_ (i.e.,ˆ _zk_ itself if
    ˆ _zk_ ∈ _Zk_ ).
- The algorithm does not actually need an _optimal_ solution to ( 17 ): any _feasible_
    point is enough. This opens the way for applying heuristics to ( 17 ), for instance by
    solving the continuous relaxation and then applying randomized rounding. One
    can also append the formulation ( 17 ) with additional constraints, such as those
    induced by precedence relations [ 50 ], if these can be exhibited.
- Conversely, ( 17 ) does not automatically produce a valid (local) lower boundvlow _k_
    as ( 6 ) [and ( 15 )] do, while—at least ifO _k_ =∅—requiring one for definingvlev _k_.
    Thus, the algorithm requires an initialization phase which essentially uses the
    standard CPM. A different initialization step will be discussed later on.

**Theorem 4** _Assume that Step 2 is implemented in such a way that_ v _k_ lev _changes or
problem_ ( 3 ) _is found to be infeasible_ only finitely many times_. Furthermore, assume
that the choice of tarkin Step 3 and that of_ ε _kin Step 3.1 satisfy the assumptions
of Theorem_ 2 _, only with tark_ ≥vlev _k replacing tark_ ≥vlow _k_ +γ_. Then, Algorithm_ 4
_finitely terminates with either a_ δTol _-optimal solution to problem_ ( 3 ) _or a proof that it
is infeasible._

_Proof_ Step 0 of the algorithm—possibly a complete run of Algorithm 2 —finitely
terminates due to Theorem 2 : eitherv
up
_k_ − 1 =v

low
_k_ =∞⇒Δ _k_ =0, proving that
( 3 ) is infeasible, or a feasible _zk_ is eventually found, at which point Step 0 terminates.
Since this happens at Step 2 of Algorithm 2 ,vlow _k_ is available, which is what the
initialization aimed at. Also, note that one could havev _k_ up− 1 ≤vlow _k_ +δTol<∞, i.e.,
the algorithm stops at Step 0 because the initialization has already found aδTol-optimal
solution.
The crucial assumption on the level management is thatvlev _k_ can change and ( 3 )be
infeasible only finitely many times. Consequently, if the algorithm does not terminate,
there exists an iteration _k_ ̄such thatvlev _k_ =vlev _k_ ̄ andZ _k_ =∅for all _k_ ≥ _k_ ̄. Let us


654 W. van Ackooij et al.

therefore assume _k_ ≥ _k_ ̄:wehavev _j_ +〈w _j_ , _zk_ − _zj_ 〉≤vlevfor all _j_ ∈O _k_ , where
vlev=vlev _k_ =vlev _k_ ̄ , from which we get the analogous of ( 12 )

```
‖w j ‖‖ zj − zk ‖≥〈w j , zj − zk 〉≥v j −vlev. (18)
```
DefiningT _k_ ={ _j_ ∈O _k_ :v _j_ ≤vlev},( 18 ) immediately shows that _zk_ = _zj_ for
_j_ ∈/T _k_. Again, only tight iterations can repeat previous iterates. Therefore,v _k_ >vlev
can only happen finitely many times. In this case one has _k_ ∈/T _k_ + 1 , and more in
general _k_ ∈/T _h_ for all _h_ > _k_. So, each time a non-tight iteration is performed, its
iterate must be different from all these previous non-tight iterations. Thus, finiteness
of _Z_ ensures that either the algorithm stops, or eventually only tight iterates can be
performed. Let us therefore assume that _k_ ̄is large enough so that for all _k_ ≥ _k_ ̄the
iterate is tight.
From the hypotheses on tar _k_ andε _k_ , eventually tar _k_ ≥vlevandε _k_ ≤δTol, i.e,
v ̄ _k_ −v _k_ ≤ε _k_ ≤δTol. Copying ( 13 ) (keepingI _h_ andT _h_ ′unchanged), we similarly
conclude

```
vup k − 1 ≤min{ ̄v j : j ∈T k }≤min{v j +ε j : j ∈T k ′}
≤v k lev+min{ε j : j ∈T k ′}≤vlev+δTol.
```
But the choice ofvlevat Step 2 now requires thatvlev <v _k_ up− 1 −δTol ≤vlev,a
contradiction: hence, the algorithm must terminate finitely. 

The assumptions on Step 2 are not trivial to satisfy. This is because the general
rule in Step 2,v _k_ lev∈[vlow _k_ ,vup _k_ − 1 −δTol), _requires_ changing the value ofvlevfrom

that of the previous iteration whenv _k_ lev− 1 >v
up
_k_ − 1 −δTol, i.e., one has found a better
upper bound at the previous iteration that forcesvlevto be decreased. Furthermore,
whenZ _k_ =∅it is easy to choosevlev _k_ + 1 in a way that causes this to happen again at

the next iteration: just increasevlevof a vanishingly small fraction. Hence, ensuring
that none of this happens infinitely many often requires careful choices in the updat-
ing mechanism. This is especially true ifδTol=0, because it means that eventually
one must havevlow _k_ =v _k_ lev=v _k_ up− 1 =v∗, quite a high call. Indeed, we will show
that this is not, in fact, possible unless one provides the algorithm with a significant
helping hand under the form of a way to compute “tight” lower bounds. Different work-
ing examples ofvlev-selection mechanisms can be developed, though, as we discuss
below.
The analysis should also plainly extend to the case when the feasible set _Z_ is not
finite, but still bounded,Dom(v)⊂ _Z_ (ensuring thus that∂vis locally bounded)
andδTol>0. This case is analyzed in [ 70 ], under stricter assumptions on the ora-
cle. We have not pursued this extension because our analysis is rather focussed on
the handling of the accuracy parameters in the oracle: very similar results could be
expected in the non-finite compact case, but this would make the arguments harder to
follow.


Inexact stabilized Benders’ decomposition approaches... 655

**4.3 Accuracy handling in the LICCPM**

The short proof of Theorem 4 somehow hides the fact that the level parametervlevhas
to be properly managed for the assumptions to be satisfied. We now discuss possible
mechanisms which obtain that.
IfδTol>0, then the assumption of the Theorem can be satisfied by the following
simple mechanism: in Step 3, wheneverZ _k_ =∅we setv _k_ low←v _k_ lev. Furthermore,
denoting by _h_ ( _k_ )< _k_ the iteration wherevlev _k_ has last changed ( _h_ ( 1 )=1), for some
fixedα∈( 0 , 1 )we set

```
v k lev←
```
##### {

```
v
up
k − 1 −max{δTol,αΔ k }ifv
```
```
up
k − 1 <v
```
```
up
h ( k )−δTolorZ k −^1 =∅
vlev h ( k ) otherwise
```
##### . (19)

In plain words,vlev _k_ is updated whenevervlowneeds be revised upwards, orvupis
“significantly” revised downwards. This mechanism ensures thatvlev _k_ cannot change
infinitely many times. In fact, even if the same iterate _zk_ (possibly, the optimal solu-
tion) is generated more than once, updating the upper boundv ̄ _k_ by vanishingly small
amounts,vlev _k_ only changes if the upper bound decreases “significantly”, i.e., by at least
δTol>0. Similarly,Z _k_ =∅cannot happen infinitely many times: in fact, whenever
this happens

```
Δ k + 1 =v
up
k −v
```
```
low
k ≤v
```
```
up
k − 1 −v
```
```
lev
k − 1 =max{δTol,αΔ k }.
```
In other words, the gap shrinks exponentially fast until eventuallyΔ _k_ ≤δTol, triggering
the stopping condition. Note, however, that forδTol=0( 19 ) only givesΔ _k_ →0, but
not finite convergence.
Although not particularly significant in practice, it may be worth remarking that
the fact that ( 19 ) only works withδTol>0, unlike those of Algorithms 2 and 3 ,is
not not due to a weakness of the analysis, but rather to an inherent property of level-
based methods. Indeed, using a level stabilization one does not have a way to prove
that a given valuevlev _k_ is a _sharp_ lower bound onv∗: whenZ _k_ =∅we can conclude
vlev _k_ <v∗. The fact that the inequality is strict shows that a level-based approach
will never be able to prove thatvlev _k_ =v∗. This is why one is not allowed to pick
vlev _k_ =vup _k_ − 1 :ifv _k_ up− 1 =v∗, one would never be able to prove this becauseZ _k_ =∅.
Indeed, consider a stylized problem with _Z_ ={ ̄ _z_ }. At the first iteration (in Step 0),
the oracle may providev ̄ 1 =v( _z_ ̄)=v∗and somev 1 =v∗−ε 1 (withε 1 >0), so
that Step 0 ends withvlow 1 =v∗−ε 1. Even if one setsε _k_ =0, an infinite sequence of
iterations then follows wherebyZ _k_ =∅always happens andvlow _k_ →v∗—say, using
( 19 )—but never quite reaching it. This is a known (minor) drawback of this form of
stabilization.
Actually, LICCPM can finitely converge even ifδTol=0, but only if the initialvlow _k_
provided by Step 1 happens to be preciselyv∗. This is because thatvlow _k_ is produced
by different means, which do allow to prove thatvlev _k_ ≤v∗. This observation suggests
a variant of the algorithm which can work withδTol =0: just prior to setting the


656 W. van Ackooij et al.

level parameter one solves ( 6 ) and updatesv _k_ lowwith its minimum value. This was
proposed when level methods were introduced [ 40 ], in the continuous case. The con-
dition ensures thatZ _k_ =∅will happen at _every_ iteration, thus making the relevant part
of the assumption in Theorem 4 moot. However, this re-introduces the risk thatvlow _k_
increases infinitely many times. Furthermore, ( 19 ) now no longer rule out the risk that
decreases ofv
up
_k_ are vanishing. To avoid these problems, one may for instance intro-
duce some mechanism whereby eventuallyvlev _k_ =v _k_ low: basically, at some point the
algorithm reverts to the standard CPM. Hybrid versions where ( 6 ) is solved “from time
to time” are also possible. While in principle applicable, we do not see this approach
as promising in our specific setting because our master problem is combinatorial, and
hence possibly computationally costly. For this reason we do not pursue its analysis
further.
A somewhat opposite approach would be to dispense the need of finding avlow _k_ that
is a guaranteed lower bound onv∗from the start, and hence the need of solving ( 6 )at
least once. To do that, one can avoid the call to (the modified) Algorithm 2 in Step 0,
and instead initializevlow 1 <∞arbitrarily. Then, the following has to be added right
before Step 3.1:

```
Step 3.0 (vlow k update) IfZ k =∅has not happened yet andv k up−δ k <vlow k , then
vlow k ←min{vup k ,vlow k }−δ k , chooseε k + 1 ∈[ 0 ,∞)arbitrarily and go to Step 4
```
whereδ _k_ ∈(δTol,δ ̄]for someδ< ̄ ∞. With this modification, similar to the one present
in [ 16 ], the algorithm replaces the dependable _lower_ boundvlow _k_ onv∗with a guess,
produced using the best available _upper_ bound and a displacement (this is called a
“target value” approach [ 13 ]). Step 3.0 cannot be executed infinitely many times: each
time it doesvlow _k_ decreases by an amount bounded away from zero, andv∗>−∞.
Hence, eventuallyv _k_ lowwill be a valid lower bound onv∗−δ ̄, andvup _k_ −δ _k_ <vlow _k_
can no longer happen unlessvlow _k_ increases. But the latter only happens whenZ _k_ =∅,
at which point Step 3.0 is disabled: a dependable lower bound has been found, and
the normal course of the algorithm, as analyzed in Theorem 4 , starts. Basically, all the
iterations up to that point take the place of the Step 0 where ( 6 ) is solved. Note that
the algorithm cannot stop before thatZ _k_ =∅at least once, since the target will always
“overrun”v _k_ up− 1 by at leastδ _k_ >δTol. Hence, the analysis of Theorem 4 still applies. In
fact, similarly to §4.1, we can disable the accuracy control when the target decreases.
As a final remark, the analysis clearly extends to the more “eager” accuracy control
versions of §3.1, with the corresponding computational trade-offs.

**4.4 Bundle resets: making master problems easier to solve**

All the master problems considered in this work are combinatorial problems, and
hence in principle difficult to solve. It can be expected that the size of the two bundles
F _k_ andO _k_ , respectively of feasibility and optimality cuts, may have an impact on
the solution time (this may be true also forR _k_ of the reverse region constraints in
TRICCPM). It is therefore possible—although by no means certain—that reducing
the bundle sizes helps in reducing the master problem time.


Inexact stabilized Benders’ decomposition approaches... 657

In the convex case it is sometimes possible to reduce the size toO _k_ all the way down
to|O _k_ |=2 by the so-called _aggregation technique_. However, this cannot be done
forF _k_ , and even forO _k_ this only works for certain stabilizations: the standard CPM
and trust region approaches, for instance, do not have any particularly elegant way of
resetting the bundle [ 31 , §5.3], while proximal (under specific assumptions) [ 31 , §5.2]
and level do. However, the aggregation technique heavily relies on convexity of the
master problem (in particular by using the dual optimal solution), and therefore does
not extend to our discrete case, even for the stabilizations that would allow it in the
continuous one.
There is a practical way in which one can resetO _k_ (and, by the same token,F _k_ ):
it can be done arbitrarily, only provided that this happens _finitely many times_ .The
standard convergence proofs then apply after that the last reset has occurred. This
is not a very satisfactory mechanism, and in particular it does not allow to set any
a-priori bound on|O _k_ |; however, it is, basically, the only mechanism that works in
general even in the convex case, unless strong properties allow otherwise [ 31 , §5.2].
IfδTol>0, for instance, a simple way to resetO _k_ (and, similarly,F _k_ ) is to initialize
_k_ ̄←1, pickα∈( 0 , 1 ), and employ the following rule (e.g., at the beginning of Step
4)

```
IfΔ k ≤αΔ k ̄then chooseO k + 1 ⊇{ k }, k ̄← k ,elseO k + 1 =O k ∪{ k }.
```
That is, the bundle can be reset each time the optimality gap “decreases enough”; this
can happen only a finite number of times. Similar rules could check for “substantial
changes” invlow _k_ orvup _k_ separately.
There is a non-trivial trade-off regarding bundle management. On one hand, keep-
ing the bundle as small as possible may save on master problem time. On the other
hand, accruing information is what drives the algorithm, and therefore discarding
information too quickly may be very detrimental to convergence rates. A fortiori in
the discrete case, only computational experiments can provide guidance on the best
way to perform bundle resets.

#### 5 Application to probabilistically constrained optimization

To test our approaches we will consider chance-constrained optimization (CCO) prob-
lems of the form

```
f ∗:=min
```
##### {

```
f ( x ) :P[ g ( x ,ξ)≤ 0 ]≥ p , x ∈ X
```
##### }

##### (20)

whereξ∈R _r_ is a random variable, _f_ :R _n_ →Ris a convex function, _g_ =[ _gi_ ] _i_ ∈ _I_
is a mapping over a finite index set _I_ such that each _gi_ :R _n_ ×R _r_ →Ris convex
in the first argument, and _X_ =∅is a bounded convex set. The _joint probabilistic
constraints_ require that all the inequalities _gi_ ( _x_ ,ξ)≤0for _i_ ∈ _I_ hold simultaneously
with high enough probability _p_ ∈( 0 , 1 ], and arise in many applications such as water
management, finance and power generation (e.g., [ 51 , 57 , 63 ] and references therein).
For introductory texts on joint probabilistic programming we refer to [ 19 , 48 ].


658 W. van Ackooij et al.

This large class of problems contains cases of different difficulty, even for the
same _n_ , _r_ and| _I_ |, depending on the underlying assumptions on the probabilistic
constraint. For instance, setting _p_ =1 essentially eliminates the difficulty, reducing
( 20 ) to a standard convex nonlinear optimization problem, albeit potentially with many
constraints [ 8 ]. This does not mean that such a problem is trivial, since the functions
_f_ or _g_ can be nonsmooth or/and difficult to evaluate. For such cases, specialized
approaches can be required: the methods of choice currently being constrained bundle
ones [ 26 , 59 , 62 ].
When _p_ ∈( 0 , 1 )instead, one of the fundamental differentiating factors is whether
the distribution ofξis continuous or discrete. In the former case, one can face hard
nonconvex nonlinear optimization problems, and a careful theoretical study of the
properties of the probability function (e.g., differentiability [ 60 ]) is needed. We will
rather consider the case whereξtakes values in a finite setΞ={ξ _s_ : _s_ ∈ _S_ }⊆R _r_
of possible realizations (or _scenarios_ ), with associated weightsπ _s_ (summing to one).
The continuous case can clearly be approximately reduced to the discrete one by
drawing an appropriate finite sample; the key question then concerns the minimal
sample size which allows to assert feasibility for the original problem with a given
confidence level (e.g., [ 44 ] and the references therein).
Numerical methods for problems with discrete distributions are, as mentioned in
[ 18 , §2.5], necessarily based on combinatorial techniques. Indeed, there are now| _S_ |
blocks of constraints _g_ ( _x_ ,ξ _s_ )≤0, one for each _s_ ∈ _S_ , and ( 20 ) requires to minimize
_f_ over the intersection of _X_ and all possible ways to select a set of scenarios _P_ ⊆ _S_
such that

##### ∑

_s_ ∈ _P_ π _s_ ≥ _p_. In other words, while one is allowed not to satisfy all blocks
of constraints _g_ ( _x_ ,ξ _s_ ) ≤0, the set of these that are not satisfied by the chosen
solution _x_ must be a low-probability one [ 55 , Chapter 4]. To develop solution methods,
simplifying assumptions are frequently made. A common one is that all the constraint
functions _gi_ , _i_ ∈ _I_ are separable, i.e., _gi_ ( _x_ ,ξ)=ξ− ̃ _gi_ ( _x_ )for given concave functions
_g_ ̃ _i_. This assumption is crucial for optimization algorithms based on _p-efficient points_ ,
a concept introduced in [ 47 ] and used to obtain equivalent problem formulations,
as well as necessary and sufficient optimality conditions [ 19 , 48 ]. Methods based on
_p_ -efficient points are diverse: see [ 20 , 49 ] for primal and dual CPM, [ 22 ] for cone
generation methods, and [ 21 ] for augmented Lagrangian and bundle methods. All in
all, numerical techniques for this class of problems are well-developed.
When the constraints are not separable, _p_ -efficient approaches are no longer suit-
able. In this case, one frequently encounters the assumption that the constraints are
linear with respect to _x_ , e.g., _g_ ( _x_ ,ξ)= _A_ (ξ ) _x_ − _b_ (ξ ). This allows to reformulate
( 20 ) as a MINLP, which is actually a MILP if _f_ and _X_ are also linear. This is by far
the most widely studied class of CCO problems with finite support and non-separable
constraints. For instance, in the recent [ 43 ] an approach similar in spirit to combina-
torial Benders’ cuts is proposed whereby valid inequalities are derived to strengthen
the MILP formulation of the problem, making it easier to solve large-scale instances
with standard MILP tools. In this analysis, linearity plays a crucial role. The approach
is extended to a wider class of problems in [ 42 ], but again linearity is essential.
In this work we will neither assume that the constraints _g_ are linear with respect to
_x_ , nor separable: our only assumptions are that _f_ and _g_ are convex in _x_ , _X_ is compact
and convex, andξhas finite supportΞ. Then, the probability constraint in ( 20 ) can be


Inexact stabilized Benders’ decomposition approaches... 659

modeled by a standard _disjunctive reformulation_. That is, a binary variable _zs_ ∈{ 0 , 1 }
for each _s_ ∈ _S_ is introduced which dictates whether or not the block of constraints
_g_ ( _x_ ,ξ _s_ )≤0 is going to be satisfied by the optimal solution _x_. This requires to estimate,
for each _i_ ∈ _I_ a large enough constant _Msi_ that makes the constraint redundant over
_X_ :
_Mis_ ≥max{ _gi_ ( _x_ ,ξ _s_ ): _x_ ∈ _X_ }. (21)

We remark that while ( 21 ) is an easy problem in the linear case, as it amounts to
maximizing a linear function over a well-behaved convex set, this is in principle no
longer true in the nonlinear case, as maximizing a convex function (even a quadratic
one) over a convex set (even a hypercube) is in generalNP-hard [ 14 ]. However, this
issue can be tackled in different ways, such as defining an appropriate concave upper
approximation of _gi_ (say the _concave envelope_ of _gi_ over _X_ , e.g., [ 2 ]) or approximating
_X_ as an ellipsoid which, if _gi_ is quadratic, makes the problem tractable [ 14 ]. We will
therefore assume that constants _Mis_ are available: then, ( 20 ) can be reformulated as a
MINLP using

```
P[ g ( x ,ξ)≤ 0 ]≥ p ≡
```
##### ⎧

##### ⎨

##### ⎩

```
g ∑ i ( x ,ξ s )≤ Miszs i ∈ I , s ∈ S
i ∈ S π szs ≤^1 − p
zs ∈{ 0 , 1 } s ∈ S
```
##### ⎫

##### ⎬

##### ⎭

##### ≡

##### {

```
G ( x )≤ Tz , z ∈ Z
```
##### }

##### (22)

for the obviously defined _G_ ( _x_ )=[ _g_ ( _x_ ,ξ _s_ )] _s_ ∈ _S_ , _T_ and _Z_. Therefore, ( 20 ) fits the
general scheme ( 1 ), and hence we can solve it via GBD. To the best of our knowledge,
GBD has never been used as a main tool for solving CCO programs of this form,
although some mechanics of the approach are used in [ 43 ] in a less general setting.

**5.1 Application: a hybrid robust/chance-constrained model**

We consider the minimization of some objective function _f_ :R _n_ →Rsubject to
linear constraints
_Ax_ ≤ξ, (23)

where bothξand _A_ are subject to uncertainty. It occurs in many cases that there are
different sources of uncertainty, not all equally well understood. This setting is of
interest for instance in energy management, where _x_ represents an energy production
schedule (e.g., [ 57 ] for the unit-commitment problem), and ( 23 ) means that we wish
to produce sufficient energy in all but the most extreme and implausible scenarios.
Knowledge of the distribution ofξ(energy demand) is available, since its characteri-
zation has received considerable attention, while _A_ is related to the underlying physics
of generation plants and/or to the behavior of other generation companies, and much
less information is available.
We can therefore employ a _hybrid robust/chance-constrained_ approach. Let _A_ =
[ _ai_ ] _i_ ∈ _I_ ; we will assume that the uncertainty about the coefficients matrix can be
expressed in the form _ai_ ( _u_ )= ̄ _ai_ + _Piu_ , where _a_ ̄ _i_ ∈R _n_ , _Pi_ is an _n_ × _ni_ matrix,


660 W. van Ackooij et al.

and the _uncertainty set u_ ∈U _i_ ={ _u_ ∈R _ni_ :‖ _u_ ‖≤κ _i_ }is the ball of radiusκ _i_ in the

2 norm. For the sake of notation we defineU=[U _i_ ] _i_ ∈ _I_ , and we write _A_ ( _u_ )for _u_ ∈U
to mean[ _ai_ ( _ui_ )] _i_ ∈ _I_ , where _ui_ ∈U _i_. On the other hand,ξ∈R _m_ is a random variable
with known distribution, in our setting represented by a finite setΞof realizations
(possibly obtained by appropriate sampling). We can then express our requirement
under the form of the _robust chance-constraint_

##### P

##### [

```
A ( u ) x ≤ξ ∀ u ∈U
```
##### ]

```
≥ p. (24)
```
For a fixedξ, the well-established theory of robust optimization (e.g., [ 4 ]) applies:
_ai_ ( _u_ )T _x_ ≤ξ _i_ for all _u_ ∈U _i_ if and only if max{ _ai_ ( _u_ )T _x_ : _u_ ∈U _i_ }≤ξ _i_ , which due
to our choice of _ai_ ( _u_ )reduces to

max{ _ai_ ( _u_ )T _x_ : _u_ ∈U _i_ }= ̄ _a_ T _ix_ +max{( _Pi_ T _x_ )T _u_ : _u_ ∈U _i_ }= ̄ _a_ T _ix_ +κ _i_ ‖ _Pi_ T _x_ ‖.

Consequently, ( 24 ) reduces toP

##### [

```
a ̄T ix +κ i ‖ Pi T x ‖≤ξ i i ∈ I
```
##### ]

≥ _p_ , which readily
falls in the setting of § 2 ,as _gi_ ( _x_ ,ξ)= ̄ _a_ T _ix_ +κ _i_ ‖ _Pi_ T _x_ ‖−ξ _i_ is convex in _x_ (since,
obviously,κ _i_ >0). The approach easily extends toUbeing defined by any convex

(^) _p_ norm, resulting in the conjugate norm in the constraint above, but this is well-
known and we stick to the 
2 case for simplicity; here, _gi_ is a Second-Order Cone
representable function.
It is interesting to remark that ( 24 ) implies the weaker condition

##### P

##### [

```
A ( u ) x ≤ξ
```
##### ]

```
≥ p ∀ u ∈U. (25)
```
Indeed, because in ( 24 ) the probability constraint has to hold for the _maximum over
all u_ ∈Uof _A_ ( _u_ ), a fortiori it has to hold for any specific choice. The inverse is not
true in general, as shown by the following counterexample.

_Example 1_ TakeU={ _u_ 1 , _u_ 2 }andΞ={ξ 1 ,ξ 2 }with probabilityπ 1 =π 2 = 0 .5.
Moreover, suppose that there exists _x_ ̄such that

```
A ( u 1 ) x ̄≤ξ 1 but A ( u 1 ) x ̄ξ 2 , A ( u 2 ) x ̄ξ 1 but A ( u 2 ) x ̄≤ξ 2.
```
Therefore, _x_ ̄is feasible for ( 25 ) for the choice _p_ = 0 .5:P[ _A_ ( _u_ ) _x_ ̄≤ξ]≥ 0 .5 however
chosen _u_ ∈U. However,P[ _A_ ( _u_ ) _x_ ̄≤ξ ∀ _u_ ∈U]=P[∅] =0. Numerical data may

be picked as _x_ ̄=( 1 , 1 ),ξ 1 =( 2 , 0 ),ξ 2 =( 0 , 2 ), _A_ ( _u_ 1 )=

##### (

##### 11

##### 00

##### )

```
, A ( u 2 )=
```
##### (

##### 00

##### 11

##### )

##### .

The example shows that the weaker model ( 25 ) may not behave satisfactorily: since
we do not know the underlying distribution of _u_ , when decision _x_ has been taken and
_u_ ∈Uturns up, the actual probabilityP[ _A_ ( _u_ ) _x_ ≤ξ]may turn out to be arbitrarily low.
This is in fact analogous to the difference between joint and individual probabilistic
constraints (e.g., [ 63 ]).


Inexact stabilized Benders’ decomposition approaches... 661

**5.2 Generation of problem instances**

For our experiments we focussed on problems of the form

```
min
```
##### {

```
c T x :P
```
##### [

```
A ( u ) x ≤ξ ∀ u ∈U
```
##### ]

```
≥ p , 0 ≤ x ≤ ̄ x
```
##### }

##### , (26)

where _x_ ̄∈R _n_ is a bound, the objective function is linear,ξ∈R _r_ has finite supportΞ
and _A_ ( _u_ ),Uare as described in the previous paragraph.
Instances of problem ( 26 ) were generated by using the following procedure. We
begin by setting the problem dimensions _n_ ,| _I_ |and| _S_ |.Wealsoset _pi_ = _n_ the
common dimension of the matrices _Pi_ involved in the constraints. Finally, we took
κ _i_ = 1 /2,π _s_ = 1 /| _S_ |uniformly, and _p_ = 0 .8. The next step consists of randomly
generating the matrix _A_ ̄ =[ ̄ _ai_ ] _i_ ∈ _I_ , with entries in[− 10 , 10 ]. The matrices _Pi_ and
vector _c_ were generated likewise with entries in[− 1 , 1 ]. Because coefficient matrices
are usually sparse in real-world problems, we generated both _A_ ̄and _Pi_ sparse. Finally,
we generated a random candidate solution _xc_ with entries in[ 0 , 10 ]and we computed
ξ ̄such thatξ ̄ _i_ = ̄ _ai_ T _xc_ +κ _i_ ‖ _Pi_ T _xc_ ‖for _i_ ∈ _I_. Scenarios forξwere generated as
ξ _s_ =ξ ̄+ _rs_ , where _rs_ was chosen in two different ways. For (at least) a fraction _p_ of
the scenarios, _rs_ was chosen with entries in[ 0 , 20 ], so thatξ _s_ ≥ξ ̄. For the remaining
(at most) 1− _p_ fraction of scenarios (properly rounding taking place), _rs_ is allowed to
have entries spanning[− 20 , 20 ]. Thus, it is immediate to realize that _xc_ is feasible for
problem ( 26 ) by construction. The constant _M_ , identical for all scenarios, has been
set by carefully analyzing the data of the instances with an ad-hoc approach.
The choice of the 
2 norm forUmeans that problem ( 3 ) has a Mixed-Integer
Second-Order Cone formulation, that can be directly solved by off-the-shelf tools
likeCplexprovided that it is reformulated as a Quadratically-Constrained Quadratic
Problem. By introducing auxiliary variables _yi_ for all _i_ ∈ _I_ ,( 26 ) can be rewritten by
means of the following constraints

```
a ̄T ix +κ iyi −ξ is ≤ Miszs i ∈ I , s ∈ S
x T( PiPi T) x ≤ yi^2 i ∈ I (27)
```
which are appropriately dealt with byCplexeven if the Hessian matrix of ( 27 )is
not, strictly speaking, positive semi-definite. Subproblem ( 2 ) can be written and solved
with the same tools.

**5.3 Setup and results**

We have generated several problem instances as follows. First we have chosen _n_ and
| _I_ |ranging over{ 50 , 100 }and| _S_ |∈{ 50 , 100 , 500 }, for a total of 12 combinations. We
also varied the sparsity of _A_ ̄and _Pi_ in the set{ 1 , 0. 1 , 0 .01 %}, but only considering the
combinations of adjacent sparsity levels, i.e., avoiding the combinations( 1 , 0 .01 %)
and( 0. 01 ,1%), for a total of seven cases. For each of the above we generated three
instances changing the seed of the random number generator, for a total of 12· 7 ·


662 W. van Ackooij et al.

(^0102030405060708090100)
0.2
0.4
0.6

0. 8

```
1
```
```
γ
```
```
φ(γ
```
```
)
```
```
Performance profile based on CPU(s)
```
```
BoxCP
LeMonolithicvel
```
(^0102030405060708090100)
0.2
0.4
0.6

0. 8

```
1
```
```
γ
```
```
φ(γ
```
```
)
```
```
Performance profile based on CPU(s)
```
```
BoxCP
LeMonolithicvel
```
(^0102030405060708090100)
0.2
0.4
0.6

0. 8

```
1
```
```
γ
```
```
φ(γ
```
```
)
```
```
Performance profile based on CPU(s)
```
```
BoxCP
LeMonolithicvel
```
```
(a) (b)(c)
```
**Fig. 1** Performance profiles for the different methods based on CPU time. **a** All instances. **b** Instances with
low sparsity. **c** Instances with high sparsity

3 =252 instances. We have experimented with two different values forδTol:10−^4 ,
the default optimality tolerance for MINLPs, and 10−^3 , considered “coarse” but still
acceptable in some cases. Both are to be intended as _relative_ , which, via appropriate
scalings, does not impact our analysis in § 3 and § 4 , where _absolute_ tolerances were
used for simplicity. We also set a time limit for each method of 100,000 seconds, on
a cluster with Intel Xeon X5670 CPUs, each with 8 Gb of reserved memory. Both the
“monolithic” approach and all the optimization problems in the GBD one (the MILP
master problem and the SOCP subproblems) have been solved withCplex12.4,
single-threaded. Other than that, the time limit and optimality tolerance, no other
parameters ofCplexwere tuned.
We have compared the “Monolithic” approach and three variants of GBD:
CP, the CPM of Algorithm 1 ,Box, the TRICCPM of Algorithm 3 , andLevel,the
LICCPM of Algorithm 4. In all cases a primal-dual oracle was employed, where
Cplexwas used to solve the SOCP formulation of the subproblem (cf. ( 27 )). Several
experiments were carried out with a dual oracle, but it was found not competitive in this
setting, and we do not report the corresponding results. The TRICCPM changes the
stability center whenever a better solution is found, i.e.,β=0, and moves through box
sizes{ 0. 005 , 0. 5 , 1 }| _S_ |. The LICCPM uses mechanism ( 19 ) described after Theorem
4 withα= 0 .9, updates the lower boundvlow _k_ by also solving the standard master
problem ( 6 ) at every iteration as discussed after Theorem 4 , systematically chose the
last iterate as the next stability centerˆ _zk_ , and uses a stabilization parameter close to
the value of 0.18 which is optimal in the continuous setting [ 40 ]. Some tuning was
performed, for instance about other choices for the center update rule, but the results
were quite stable.
We compare the solvers by means of performance profiles [ 24 ], which read as
follows:theparameterγrepresentsascalarvalue,andφ(γ)thepercentageofproblems
on which a given solver was no slower thanγtimes the fastest solver on that particular
problem. The valueφ( 0 )shows which solver is the fastest one and the valueφ(∞)is
the percentage of instances that the solver managed to solve. Looking atφ( 0 )in Fig.
1 a shows thatCPis faster in roughly 40% of the instances, whereas the other methods
are roughly each at 20%. The GBD approaches are noticeably more robust than the
monolithic one.
The impact of sparsity is illustrated on Fig. 1 b, c, which report performance profiles
restricted respectively to “low sparsity” instances ({ 1 , 0 .1%}) and “high sparsity”


Inexact stabilized Benders’ decomposition approaches... 663

(^0102030405060708090100)
0.2
0.4
0.6

0. 8

```
1
```
```
γ
```
```
φ(γ
```
```
)
```
```
Performance profile based on CPU(s)
```
```
Box
CPLevel
Monolithic
```
(^0102030405060708090100)
0.2
0.4
0.6

0. 8

```
1
```
```
γ
```
```
φ(γ
```
```
)
```
```
Performance profile based on CPU(s)
```
```
Box
CPLevel
Monolithic
```
```
(a) (b)
```
**Fig. 2** Performance profiles for the different methods based on CPU time withδTol= 10 −^3. **a** All instances.
**b** Instances with high sparsity

(^0102030405060708090100)
0.2
0.4
0.6

0. 8

```
1
```
```
γ
```
```
φ(γ
```
```
)
```
```
Performance profile based on CPU(s)
```
```
BoxCP
LeMonolithicvel
```
(^0102030405060708090100)
0.2
0.4
0.6

0. 8

```
1
```
```
γ
```
```
φ(γ
```
```
)
```
```
Performance profile based on CPU(s)
```
```
BoxCP
LeMonolithicvel
```
(^0102030405060708090100)
0.2
0.4
0.6

0. 8

```
1
```
```
γ
```
```
φ(γ
```
```
)
```
```
Performance profile based on CPU(s)
```
```
BoxCP
LeMonolithicvel
```
```
(a) (b) (c)
```
**Fig. 3** Performance profiles for the different methods based on CPU time on the easy instances. **a** Instances
with high sparsity. **b** Instances with low sparsity. **c** All instances

ones ({ 0. 1 , 0 .01 %}):Monolithicgets more and more outperformed by the GBD
as sparsity increases. This shows the potential of decomposition methods, since real-
world problems are often highly sparse (even below 0.001%).
The impact of the stopping tolerance can be gauged by comparing Fig. 1 a with
Fig. 2 a, where the performances are reported withδTol= 10 −^3 is required. For this
coarser tolerance, the monolithic approach outperforms the GBD ones. However, this
is mostly true for dense instances only: as shown in Fig. 2 b, for sparser instances the
GBD remain competitive even at a lower precision.
Oneissuewithperformanceprofilesisthattheydonotdiscriminateamonginstances
of widely varying “difficulty”. That is, two solvers tested on two instances such that the
first one has a running time of 2 and 1000 while and the second one has a running time
of 1 and 2000 (in whichever units) would show to have exactly the same performance
profiles, while one may be interested in knowing that the first solver is “better on harder
instances although worse on easier ones”. To investigate this issue we subdivided
our instances in three classes: easy if the fastest solver takes less then one minute,
intermediate if it takes between 1 and 10 min, and hard otherwise. Figure 3 a–c show
the performance of the solvers on the easy instances (withδTol= 10 −^4.


664 W. van Ackooij et al.

(^0102030405060708090100)
0.2
0.4
0.6

0. 8

```
1
```
```
γ
```
```
φ(
γ)
```
```
Performance profile based on CPU(s)
```
```
BoxCP
Level
Monolithic
```
(^0102030405060708090100)
0.2
0.4
0.6

0. 8

```
1
```
```
γ
```
```
φ(
γ)
```
```
Performance profile based on CPU(s)
```
```
BoxCP
Level
Monolithic
```
```
(a) (b)
```
**Fig. 4** Performance profiles for the different methods based on CPU time on the hard instances. **a** Instances
with high sparsity. **b** Instances with low sparsity

(^0102030405060708090100)
0.2
0.4
0.6

0. 8

```
1
```
```
γ
```
```
φ(
γ)
```
```
Performance profile based on CPU(s)
```
```
BoxCP
LeMonolithicvel
```
(^0102030405060708090100)
0.2
0.4
0.6

0. 8

```
1
```
```
γ
```
```
φ(
γ)
```
```
Performance profile based on CPU(s)
```
```
BoxCP
LeMonolithicvel
```
(^0102030405060708090100)
0.2
0.4
0.6

0. 8

```
1
```
```
γ
```
```
φ(
γ)
```
```
Performance profile based on CPU(s)
```
```
BoxCP
LeMonolithicvel
```
```
(a) (b) (c)
```
**Fig. 5** Performance profiles for the different methods based on CPU time, discriminated according to the
number of scenarios| _S_ |. **a** 50 Scenarios. **b** 100 Scenarios. **c** 500 Scenarios

The figures show that whileMonolithicis competitive for easy instances, this
is only so for dense problems: for sparse ones, althoughMonolithicis fastest in
around 60% of the cases, the solvers based on GBD are far more robust. The picture
is even clearer for hard instances, as shown in Fig. 4 a, b: there,Monolithicis
significantlyoutperformedalsoonlowsparsityinstances.Similarresultswereobtained
for the intermediate instances, and therefore we avoided to report the corresponding
profiles.
The impact of the number of scenarios is illustrated in Fig. 5 a–c. The results show
that GBD is quite stable as| _S_ |varies, whereasMonolithicis less and less robust
as| _S_ |increases.
Finally, we have performed some experiments related to the effect of setting a target
tar<∞. This has proved somewhat more complex than anticipated for a number of
reasons. On one hand, the dual oracle, which is the one to which the approach is most
suited, was not particularly efficient in our case. On the other hand, the _unfeasible start_
primal-dual approach implemented inCplexturned out to produce feasible solutions
(in particular, dual ones) rather late in the optimization. Hence, even intercepting them
as early as possible using the appropriate callbacks did not significantly improve the
running times. We therefore resorted to a somewhat abstract setting, by executing the
dual oracle _twice_ at each iteration: once with tar _k_ =∞, and once with a finite target


Inexact stabilized Benders’ decomposition approaches... 665

(the best current value). This has been done in the ICCPM, and only the dual solution of
the exact oracle has been used to compute the feasibility cuts. We have then compared
the oracle time when the target is specified with that when it is not. On average over
all the 252 data sets, using the target resulted in a reduction of oracle time of about
77%, with a standard deviation of around 20% and cases where the reduction was
above 99%. Only in two instances the running time increased, by a relatively minor
fraction. Although these experiments disregard the impact on the convergence speed
of the different feasibility cuts produced, they do indicate that using an inexact oracle
may be beneficial.
All in all, our results prove that, on this class of problems, and for our specific
test set, the GBD approaches are in general competitive with the monolithic one. In
particular,Monolithicis competitive (but not necessarily the best choice on sparse
instances) whenδTol= 10 −^3 , and should be preferred only for “easy” and “dense”
instances withδTol= 10 −^4. In all the other cases, the GBD approaches are significantly
better.

#### 6 Conclusions and perspectives

In this paper we have studied the combination of two approaches for improving the
performances of algorithms for the minimization of a convex functionv( _z_ ), given by
a first-order oracle, over a finite domain _Z_. One idea is to relax the conditions on
the information produced by the oracle, requiring it to only be an _informative on-
demand inexact_ one ( 10 ), so as to make each iteration cheaper. The second idea is to
employ different forms of stabilization (trust region and level) to reduce the number
of oracle calls. Employing both techniques simultaneously requires some care in the
handling of the accuracy parameters; our convergence results seem to require very
weak conditions, which basically show that the objective function may need to be
computed accurately only in a small fraction of the iterates. Our analysis should also
plainly extend to the case setting of Zaourar and Malick [ 70 ], i.e., the feasible set _Z_
is not finite but is bounded, and∂vis locally bounded. Our results would then extend
these obtained in [ 70 ], under stricter assumptions on the oracle andδTol>0.
Our analysis is primarily interesting for improving the performances of Gener-
alized Benders’ Decomposition approaches. An interesting property of oracles with
on-demand accuracy is that they are able to provide linearizations even in absence of
constraint qualification for the underlining convex problem (defining the value func-
tion). Moreover, linearizations can even be computed at points not belonging to the
domain of the value function, which may allow to implement the algorithm without
using feasibility cuts (unless the problem is infeasible). We remark that while we dis-
cuss the inexact oracle in the case where the subproblem is convex, our analysis also
applies (and it is possibly even more relevant) to the case where it is a hard prob-
lem (e.g., [ 9 , 37 , 41 , 54 ]), which makes obtaining good upper and lower estimates even
more time consuming.
In order to test the computational significance of the developed techniques, we have
applied them on a class of hybrid Robust and Chance-Constrained Optimization prob-
lems, arising when a linear program is subject to two different sources of uncertainty,


666 W. van Ackooij et al.

that need to be dealt with with different techniques (an uncertainty set plus a finite
set of scenarios). These problems allow a monolithic formulation that a commercial
solver such asCplexcan solve; however, our results show that GBD approaches are
often much more efficient, in particular for high sparsity, a large number of scenarios,
and a higher final accuracy. Also, the experiments indicate that the ideas developed in
this work are promising to improve the performances of decomposition techniques.
To conclude, this work is significant in three areas: general algorithms for the mini-
mization of oracle-provided convex functions over discrete sets, Generalized Benders’
Decomposition approaches, and Chance-Constrained Optimization. Of course, further
improvements are possible in all the three areas. However, we believe that our results
already show that the combination of different concepts such as _probability valid
inequalities_ , _strengthening formulations_ for combinatorial problems and oracles with
_on-demand accuracy_ from nonsmooth optimization will prove fruitful for solving
problems as ( 20 ).

**Acknowledgments** The authors gratefully acknowledge financial support from the Gaspard-Monge pro-
gram for Optimization and Operations Research (PGMO) project “Consistent Dual Signals and Optimal
Primal Solutions”. The first and second authors would also like to acknowledge networking support by the
COST Action TD1207.

#### References

1. Baena, D., Castro, J., Frangioni, A.: Stabilized Benders methods for large-scale combinatorial opti-
    mization: applications to data privacy (2015)
2. Bao, X., Sahinidis, N., Tawarmalani, M.: Multiterm polyhedral relaxations for nonconvex,
    quadratically-constrained quadratic programs. Optim. Methods Softw. **24** , 485–504 (2009)
3. Ben Amor, H., Desrosiers, J., Frangioni, A.: On the choice of explicit stabilizing terms in column
    generation. Discret. Appl. Math. **157** (6), 1167–1184 (2009)
4. Ben-Tal, A., Ghaoui, L.E., Nemirovski, A.: Robust Optimization. Princeton University Press, Princeton
    (2009)
5. Ben-Tal, A., Nemirovski, A.: Lectures on Modern Convex Optimization: Analysis, Algorithms, Engi-
    neering Applications. MPS-SIAM Series on Optimization. SIAM, Philadelphia (2001)
6. Benders, J.: Partitioning procedures for solving mixed-variables programming problems. Numer. Math.
    **4** (1), 238–252 (1962)
7. Boyd, S., Vandenberghe, L.: Convex optimization.http://www.stanford.edu/~boyd/cvxbook **ISBN 0**
    **521 83378 7** (2006)
8. Calafiore,G.C.,Campi,M.C.:Uncertainconvexprograms:randomizedsolutionsandconfidencelevels.
    Math.l Program. **102** (1), 25–46 (2005)
9. Caroe, C.C., Tind, J.: L-shaped decomposition of two-stage stochastic programs with integer recourse.
    Math. Program. **83** , 451–464 (1998)
10. Codato, G., Fischetti, M.: Combinatorial benders’ cuts for mixed-integer linear programming. Oper.
Res. **54** (4), 756–766 (2006)
11. Costa, A.M.: A survey on benders decomposition applied to fixed-charge network design problems.
Comput. Oper. Res. **32** (6), 1429–1450 (2005)
12. d’Ambrosio, C., Frangioni, A., Liberti, L., Lodi, A.: On interval-subgradient cuts and no-good cuts.
Oper. Res. Lett. **38** , 341–345 (2010)
13. d’Antonio, G., Frangioni, A.: Convergence analysis of deflected conditional approximate subgradient
methods. SIAM J. Optim. **20** (1), 357–386 (2009)
14. de Klerk, E.: The complexity of optimizing over a simplex, hypercube or sphere: a short survey. Central
Eur. J. Oper. Res. **16** (2), 111–125 (2008)
15. de Oliveira, W.: Regularized nonsmooth optimization methods for convex minlp problems. TOP pp.
1–28 (2016). doi:10.1007/s11750-016-0413-4


Inexact stabilized Benders’ decomposition approaches... 667

16. de Oliveira, W., Sagastizábal, C.: Level bundle methods for oracles with on demand accuracy. Optim.
    Methods Softw. **29** (6), 1180–1209 (2014)
17. de Oliveira, W., Sagastizábal, C., Lemaréchal, C.: Convex proximal bundle methods in depth: a unified
    analysis for inexact oracles. Math. Prog. Ser. B **148** , 241–277 (2014)
18. Dentcheva, D.: Optimization models with probabilistic constraints. In: Calafiore, G., Dabbene, F. (eds.)
    Probabilistic and Randomized Methods for Design Under Uncertainty, 1st edn, pp. 49–97. Springer,
    Newe York (2006)
19. Dentcheva, D.: Optimisation models with probabilistic constraints. In: Shapiro, A., Dentcheva, D.,
    Ruszczy ́nski, A. (eds.) Lectures on Stochastic Programming. Modeling and Theory, MPS-SIAM series
    on optimization, vol. 9, pp. 87–154. SIAM and MPS, Philadelphia (2009)
20. Dentcheva, D., Lai, B., Ruszczy ́nski, A.: Dual methods for probabilistic optimization problems. Math.
    Methods Oper. Res. **60** (2), 331–346 (2004)
21. Dentcheva, D., Martinez, G.: Regularization methods for optimization problems with probabilistic
    constraints. Math. Program. (Ser. A) **138** (1–2), 223–251 (2013)
22. Dentcheva, D., Prékopa, A., Ruszczy ́nski, A.: Concavity and efficient points for discrete distributions
    in stochastic programming. Math. Program. **89** , 55–77 (2000)
23. Dinter, J.V., Rebenack, S., Kallrath, J., Denholm, P., Newman, A.: The unit commitment model with
    concave emissions costs: a hybrid benders’ decomposition with nonconvex master problems. Ann.
    Oper. Res. **210** (1), 361–386 (2013)
24. Dolan, E.D., Moré, J.J.: Benchmarking optimization software with performance profiles. Math. Pro-
    gram. **91** , 201–213 (2002). doi:10.1007/s101070100263
25. Fábián, C.: Bundle-type methods for inexact data. In: Csendes, T., Rapcsk, T. (eds.), Proceedings of the
    XXIV Hungarian Operations Researc Conference (Veszprém, 1999), vol. 8 (pecial issue), pp. 35–55
    (2000)
26. Fábián, C., Wolf, C., Koberstein, A., Suhl, L.: Risk-averse optimization in two-stage stochastic models:
    computational aspects and a study. SIAM J. Optim. **25** (1), 28–52 (2015)
27. Feltenmark, S., Kiwiel, K.: Dual applications of proximal bundle methods, including lagrangian relax-
    ation of nonconvex problems. SIAM J. Optim. **10** (3), 697–721 (2000)
28. Fischetti, M., Lodi, A.: Local branching. Math. Program. **98** (1–3), 23–47 (2003)
29. Fischetti, M., Salvagnin, D., Zanette, A.: A note on the selection of Benders cuts. Math. Program.
    **124** (1), 175–182 (2010)
30. Floudas, C.A.: Generalized benders decomposition. In: Floudas, C.A., Pardalos, P.M. (eds.) Encyclo-
    pedia of Optimization, 2nd edn, pp. 1163–1174. Springer, New York (2009)
31. Frangioni, A.: Generalized Bundle methods. SIAM J. Optim. **13** (1), 117–156 (2002)
32. Frangioni, A., Gendron, B.: A stabilized structured dantzig-wolfe decomposition method. Math. Pro-
    gram. B **104** (1), 45–76 (2013)
33. Frangioni, A., Gorgone, E.: Generalized bundle methods for sum-functions with “easy” components:
    Applications to multicommodity network design. Math. Program. **145** (1), 133–161 (2014)
34. Frangioni, A., Lodi, A., Rinaldi, G.: New approaches for optimizing over the semimetric polytope.
    Math. Program. **104** (2–3), 375–388 (2005)
35. Geoffrion, A.M.: Generalized benders decomposition. J. Optim. Theory Appl. **10** (4), 237–260 (1972)
36. Hiriart-Urruty, J., Lemaréchal, C.: Convex Analysis and Minimization Algorithms II, 2nd edn. No.
    306 in Grundlehren der mathematischen Wissenschaften. Springer, Berlin (1996)
37. Hooker, J.N., Ottosson, G.: Logic-based benders decomposition. Math. Program. **96** , 33–60 (2003)
38. Kelley, J.: The cutting-plane method for solving convex programs. J. Soc. Ind. Appl. Math. **8** (4),
    703–712 (1960)
39. Kolokolov, A., Kosarev, N.: Analysis of decomposition algorithms with benders cuts for _p_ -median
    problem. J. Math. Model. Algorithms **5** (2), 189–199 (2006)
40. Lemaréchal, C., Nemirovskii, A., Nesterov, Y.: New variants of bundle methods. Math. Program. **69** (1),
    111–147 (1995)
41. Li, X., Chen, Y., Barton, P.I.: Nonconvex generalized benders decomposition with piecewise convex
    relaxations for global optimization of integrated process design and operation problems. Ind. Eng.
    Chem. Res. **51** (21), 7287–7299 (2012)
42. Liu, X., Küçükyavuz, S., Luedtke, J.: Decomposition algorithm for two-stage chance constrained
    programs. In: Mathematical Programming Series B, pp. 1–25 (2014). doi:10.1007/s10107-014-0832-7
43. Luedtke, J.: A branch-and-cut decomposition algorithm for solving chance-constrained mathematical
    programs with finite support. Math. Program. **146** (1–2), 219–244 (2014)


668 W. van Ackooij et al.

44. Luedtke, J., Ahmed, S.: A sample approximation approach for optimization with probabilistic con-
    straints. SIAM J. Optim. **19** , 674–699 (2008)
45. Marsten, R., Hogan, W., Blankenship, J.: The BOXSTEP method for large-scale optimization. Oper.
    Res. **23** (3), 389–405 (1975)
46. Oliveira, F., Grossmann, I., Hamacher, S.: Accelerating Benders stochastic decomposition for the
    optimization under uncertainty of the petroleum product supply chain. Comput. Oper. Res. **49** (1),
    47–58 (2014)
47. Prékopa, A.: Dual method for a one-stage stochastic programming problem with random rhs obeying
    a discrete probabiltiy distribution. Z. Oper. Res. **34** , 441–461 (1990)
48. Prékopa, A.: Probabilistic programming. In: Ruszczy ́nski, A., Shapiro, A. (eds.) Stochastic Program-
    ming, Handbooks in Operations Research and Management Science, vol. 10, pp. 267–351. Elsevier,
    Amsterdam (2003)
49. Prékopa, A., Vízvári, B., Badics, T.: Programming under probabilistic constraints with discrete random
    variable. In:Giannessi, F., Komlósi, S., Rapcsák, T.(eds.) New Trends in Mathematical Programming
    : Hommage to Steven Vajda, _Applied Optimization_ , vol. 13, pp. 235–255. Springer, New York (1998)
50. Ruszczy ́nski, A.: Probabilistic programming with discrete distributions and precedence constrained
    knapsack polyhedra. Math. Program. **93** , 195–215 (2002)
51. Ruszczy ́nski, A.: Decomposition methods. In: Ruszczy ́nski, A., Shapiro, A. (eds.) Stochastic Program-
    ming, Handbooks in Operations Research and Management Science, vol. 10, pp. 141–211. Elsevier,
    Amsterdam (2003)
52. Sahiridis, G.K.D., Minoux, M., Ierapetritou, M.G.: Accelerating benders method using covering cut
    bundle generation. Int. Trans. Oper. Res. **17** , 221–237 (2010)
53. Santoso, T., Ahmed, S., Goetschalcks, M., Shapiro, A.: A stochastic programming approach for supply
    chain network design under uncertainty. Eur. J. Oper. Res. **167** (1), 96–115 (2005)
54. Sen, S., Sherali, H.: Decomposition with branch-and-cut approaches for two-stage stochastic mixed-
    integer programming. Math. Program. **106** , 203–223 (2006)
55. Shapiro, A., Dentcheva, D., Ruszczy ́nski, A.: Lectures on Stochastic Programming. Modeling and
    Theory, _MPS-SIAM series on optimization_ , vol. 9. SIAM and MPS, Philadelphia (2009)
56. Sherali, H., Lunday, B.J.: On generating maximal nondominated Benders cuts. Ann. Oper. Res. **210** (1),
    57–72 (2013)
57. Tahanan, M., van Ackooij, W., Frangioni, A., Lacalandra, F.: Large-scale unit commitment under
    uncertainty: a literature survey. 4OR **13** (2), 115–171 (2015). doi:10.1007/s10288-014-0279-y
58. Tran-Dinh, Q., Necoara, I., Diehl, M.: Fast inexact decomposition algorithms for large-scale separable
    convex optimization. Optimization (to appear) 1–33 (2015). doi:10.1080/02331934.2015.1044898
59. van Ackooij, W., de Oliveira, W.: Level bundle methods for constrained convex optimization with
    various oracles. Comput. Optim. Appl. **57** (3), 555–597 (2014)
60. van Ackooij, W., Henrion, R.: Gradient formulae for nonlinear probabilistic constraints with Gaussian
    and Gaussian-like distributions. SIAM J. Optim. **24** (4), 1864–1889 (2014)
61. van Ackooij, W., Malick, J.: Decomposition algorithm for large-scale two-stage unit-commitment.
    Ann. Oper. Res. **238** (1), 587–613 (2016). doi:10.1007/s10479-015-2029-8
62. van Ackooij, W., Sagastizábal, C.: Constrained bundle methods for upper inexact oracles with appli-
    cation to joint chance constrained energy problems. SIAM J. Optim. **24** (2), 733–765 (2014)
63. van Ackooij, W., Henrion, R., Möller, A., Zorgati, R.: Joint chance constrained programming for hydro
    reservoir management. Optim. Eng. **15** , 509–531 (2014)
64. van Slyke, R., Wets, R.B.: L-shaped linear programs with applications to optimal control and stochastic
    programming. SIAM J. Appl. Math. **17** , 638–663 (1969)
65. Wentges, P.: Accelerating benders’ decomposition for the capacitated facility location problem. Math.
    Methods Oper. Res. **44** (2), 267–290 (1996)
66. Westerlund, T., Pörn, R.: Solving pseudo-convex mixed integer optimization problems by cutting plane
    techniques. Optim. Eng. **3** , 253–280 (2002)
67. Wolf, C., Fábián, C.I., Koberstein, A., Stuhl, L.: Applying oracles of on-demand accuracy in two-stage
    stochastic programming a computational study. Eur. J. Oper. Res. **239** (2), 437–448 (2014)
68. Yang, Y., Lee, J.M.: A tighter cut generation strategy for acceleration of benders decomposition.
    Comput. Chem. Eng. **44** , 84–93 (2012)
69. Zakeri, G., Philpott, A., Ryan, D.M.: Inexact cuts in benders decomposition. SIAM J. Optim. **10** (3),
    643–657 (2000)


Inexact stabilized Benders’ decomposition approaches... 669

70. Zaourar, S., Malick, J.: Quadratic stabilization of benders decomposition pp. 1–22 (2014). Draft sub-
    mitted; Privately communicated
71. Zappe, C.J., Cabot, A.V.: The application of generalized benders decomposition to certain nonconcave
    programs. Computers Math. Applic. **21** (6/7), 181–190 (1991)


