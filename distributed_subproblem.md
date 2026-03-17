## arXiv:1909.10451v3 [math.OC] 12 Jan 2021

### EFFICIENTSTOCHASTICPROGRAMMING INJULIA

##### A PREPRINT

```
Martin Biel
Division of Decision and Control Systems
School of EECS, KTH Royal Institute of Technology
SE-100 44 Stockholm, Sweden
mbiel@kth.se
```
```
Mikael Johansson
Division of Decision and Control Systems
School of EECS, KTH Royal Institute of Technology
SE-100 44 Stockholm, Sweden
mikaelj@kth.se
```
```
January 14, 2021
```
#### ABSTRACT

```
We presentStochasticPrograms.jl, a user-friendly and powerful open-source framework for
stochastic programming written in the Julia language. The framework includes both modeling tools
and structure-exploiting optimization algorithms. Stochastic programming models can be efficiently
formulated using expressive syntax and models can be instantiated, inspected, and analyzed inter-
actively. The framework scales seamlessly to distributed environments. Small instances of a model
can be run locally to ensure correctness, while larger instances are automatically distributed in a
memory-efficient way onto supercomputers or clouds and solved using parallel optimization algo-
rithms. These structure-exploiting solvers are based on variations of the classical L-shaped and
progressive-hedging algorithms. We provide a concise mathematical background for the various
tools and constructs available in the framework, along withcode listings exemplifying their usage.
Both software innovations related to the implementation ofthe framework and algorithmic innova-
tions related to the structured solvers are highlighted. Weconclude by demonstrating strong scaling
properties of the distributed algorithms on numerical benchmarks in a multi-node setup.
```
#### 1 Introduction

```
Stochastic programming is an effective mathematical framework for modeling multi-stage decision problems that
involve uncertainty [1]. It has been used to model complex real-world problems in diverse fields such as power
systems [2, 3, 4], finance [5, 6], and transportation [7, 8]. The classical setting is linear stochastic programs where an
actor takes decisions in two stages:
```
```
initial decisionx → observationω → recourse actiony(x, ξ(ω))
```
```
The actor first takes a decisionx. Then, the realization of a random eventωalters the state of the world. The actor
can observeωand take a recourse actionywith respect toxand the output of some random variableξ(ω). We
are interested in finding the optimal decisionx, accounting for the ability to make a recourse action onceωhas been
observed. The notion of an optimal decision is captured by lettingxandybe optimization variables in linear programs,
whereξ(ω)parameterizes the second-stage problem for each eventω.
```
```
In applications, a stochastic program models some real-world decision problem under a statistical model ofξ. We
can then compute approximations of optimal decision policies by solving approximated instances of the stochastic
program. In brief, this involves computing a first-stage decisionˆxthat is optimal in expectation over a set of second-
stage scenariosξ(ωi)sampled from the model ofξ. This technique is known as sampled average approximation
(SAA). In the linear setting, one can in principle formulatesampled instances on a extensive form that considers all
available scenarios at once. This mathematical program canbe solved using standard linear programming solvers,
including both open-source solvers such as GLPK [9] and commercial solvers such as Gurobi [10]. However, the
size of the extensive form grows linearly in the number of scenarios, and industry-scale applications typically involve
10,000+ scenarios. For example, the 24-hour unit commitment problem studied in [4] has 16,384 scenarios and the
resulting extensive form has 4 billion variables. Solving the extensive form in such applications becomes practically
```

infeasible. Moreover, the memory requirement for storing the stochastic program instances will eventually exceed the
capacity of a single machine. This clarifies the need for a distributed approach for modeling large-scale stochastic
programs. Structure-exploiting decomposition methods [11, 12] that operate in parallel on distributed data become
essential to solve large-scale instances.

1.1 Contribution

In this work, we present a user-friendly open-source software framework for efficiently modeling and solving stochas-
tic programs in a distributed-memory setting. The framework allows researchers to formulate complex stochastic
models and quickly typeset and test novel optimization algorithms. Stochastic programming educators will bene-
fit from the clean and expressive syntax. Also, the frameworksupports analysis tools and stochastic programming
constructs from classical theory and leading textbooks. Industrial practitioners can make use of the framework to
rapidly formulate complex models, analyze small instanceslocally, and then run large-scale instances in production
on a supercomputer or a cloud cluster. We implemented the framework in the Julia [13] programming language.
Henceforth, we refer to the framework as SPjl. The frameworkis freely available through the registered Julia package
StochasticPrograms.jl.

The design philosophy adopted during implementation of SPjl is centered around flexibility and efficiency, with the
aim to provide a feature-rich and user-friendly experience. Also, the framework should be scalable to support large-
scale problems. With this in mind, we adhered to the fundamental principle that the optimization modeling should
be separated from the data modeling. This design principle results in two key software innovations: deferred model
instantiation and data injection. Optimization models areformulated in stages using a straightforward syntax that
simultaneously specifies the data dependencies between thestages. The data structures related to future scenarios, and
their statistical properties, are defined separately. An essential consequence of this design is that we can efficiently
distribute stochastic program instances in memory, reducing interprocess communication to a minimum. Many com-
putations involving distributed stochastic programs can then natively be run in parallel. Moreover, when the sample
space is infinite, it becomes possible to adequately distinguish between the abstract representation of a stochastic pro-
gram and finite sampled instances. The design also enables swift implementation of various constructs from classical
stochastic programming theory. Another design choice is that the solver suites included in the framework are devel-
oped using policy-based techniques. We have shown in prior work how policy-based design can be used to create
customizable and efficient optimization algorithms [14]. In short, SPjl is a powerful, versatile, and extensible frame-
work for stochastic programming. It provides both an educational setting for new practitioners and a research setting
were experts can further the field of stochastic programming.

We developed SPjl in Julia, which has several distinct benefits. Through just-in-time compilation and type inference,
Julia can achieve C-like performance while being as expressive as similar dynamic languages such as Python or Mat-
lab. Using the high-level metaprogramming capabilities ofJulia, it is possible to create domain-specific tools with
expressive syntax and high performance. Another benefit is access to Julia’s large and rapidly expanding ecosystem of
libraries, many of which play a central role in SPjl. For example, the parallel capabilities of SPjl are implemented using
the standard library module for distributed computing, while optimization models are formulated using the JuMP [15]
ecosystem. JuMP is an algebraic modeling language implemented in Julia using similar metaprogramming tools. It
has been shown to achieve similar performance to AMPL [15], with syntax that is both readable and expressive. Also,
it is possible to mutate model instances at runtime, which weutilize in the structure-exploiting algorithms. Recently,
the backend of JuMP was redesigned into the new MathOptInterface [16]. The redesign introduces automatic refor-
mulation bridges, which are used frequently in the current implementation of the SPjl framework. JuMP implements
interfaces to many third-party optimization solvers, bothopen-source and commercial. These can be hooked in to
solve extensive forms of stochastic programs or subproblems that arise in decomposition methods.

1.2 Related work

We give a short survey of similar software packages and highlight distinguishing features of SPjl. The most similar
approach is the PySP framework [17], implemented in the Python language. Optimization models in PySP are created
using Pyomo [18]; an algebraic modeling language also implemented in Python. In contrast, SPjl is written in the
Julia language and formulates optimization models in JuMP,which has been shown to outperform Pyomo in various
benchmarks [15]. In PySP, stochastic programs are composedof multiple.datfiles and.pyfiles, and the models
are solved by running different solver scripts. In SPjl, allmodels are described in pure Julia and can be created,
analyzed and solved in a single interactive session. Moreover, all operations are natively distributed in memory and
run in parallel if multiple Julia processes are available. The parallel capabilities of PySP extend to running parallelized
versions of the solver scripts. The primary function of PySPis to formulate and solve stochastic programs, while SPjl
also provides a large set of stochastic programming constructs and analysis tools. The expressiveness of the modeling


syntax can be compared by observing how the well-known farmer problem [1] is modeled using PySP [17] and how
it is modeled using SPjl, as shown in Appendix 4. In particular, the PySP definition requires about 100 lines of code
spread out over four different files, while SPjl requires 30 lines of code with the added benefit of being more readable.
In addition, the resulting model can be analyzed interactively in Julia in a user-friendly way.

A more extensive list of similar software approaches is provided in [17], along with comparisons to PySP. This allows
for a transitive comparison to SPjl. Other notable examplesinclude the commercial FortSP solvers [19] coupled with
the AMPL extension SAMPL for modeling. Out of all these approaches, SPjl has the most user-friendly interface and
is also freely available.

The StructJuMP package [20] provides a simple interface to create block-structured JuMP models. The primary
reason for developing StructJuMP was to facilitate a parallel modeling interface to existing structured solvers [21, 4]
that operate in computer clusters. These parallel solvers are implemented in C++ and are parallelized using MPI.
This led to StructJuMP also making use of MPI to distribute stochastic programs in blocks. Apart from formulating
distributed stochastic programs in a cluster, StructJuMP does not offer any modeling tools nor any way to generate the
extensive form of a stochastic program. In comparison, SPjlprovides numerous analysis tools as well as a compatible
suite of structured solvers. In addition, SPjl natively distributes and solves stochastic programs using Julia, without
relying on external software such as MPI.

#### 2 Preliminaries

We give a short mathematical introduction to linear stochastic programming. The purpose is to provide background
for the code examples presented in the subsequent section and also to keep this work self-contained. A more thorough
introduction to the field is given in the textbook by [1].

2.1 Stochastic programming

A linear two-stage recourse model enables a simple but powerful framework for making decisions under uncertainty.
We formalize this procedure in the following brief review. The first-stage decision made by the actor is denoted by
x∈Rn. We associatexwith a linear cost functioncTxthat the actor pays after making the decision. Moreover,xis
constrained to the standard polyhedron in linear programming, i.e.

```
{x∈Rn|Ax=b, x≥ 0 }
```
whereA∈Rp×nandb∈Rp. The recourse actions are represented byy∈Rm. To describe the uncertainty in the
decision problem, we consider some probability space(Ω,F, π)whereΩis a sample space,Fis aσ-algebra overΩ
andπ:F →[0,1]is a probability measure. Letξ(ω) : Ω→RNbe some random variable onΩand letEξdenote
expectation with respect toξ. We can now letω∈Ωdenote a scenario observed after decidingx. The scenario affects
both cost and the constraints of the recourse action. Specifically, after realization ofω, the following second-stage
problem is formulated to determineywith respect toxandξ(ω):

```
Q(x, ξ(ω)) = min
y∈Rm
```
```
qTωy
```
```
s.t. Tωx+W y=hω
y≥ 0.
```
##### (1)

In other words, the random variable takes on the formξ(ω) = (qω Tω hω)

T
in this linear setting. Note that
qω∈Rm,Tω∈Rq×nandhω∈Rqare scenario-dependent whileW ∈Rq×mis fixed. This is a standard setting
in literature, which covers a wide range of problems [1]. It is possible to defineWas scenario-dependent in the
framework, but standard algorithms are then no longer certain to converge. Now, we formulate the two-stage recourse
problem as follows.

```
minimize
x∈Rn
```
```
cTx+Eξ[Q(x, ξ(ω))]
```
```
subject to Ax=b
x≥ 0 ,
```
##### (2)

The optimal value of (2) is referred to as thevalue of the recourse problem(VRP).

Apart from solving (2), we can compute two classical measures of stochastic performance. The first measures the
value of knowing the random outcome before making the decision. This is achieved by taking the expectation in (2)


outside the minimization, to obtain the wait-and-see problem:

```
EWS =Eξ
```
##### 

##### 

##### 

```
min
x∈Rn
```
```
cTx+Q(x, ξ(ω))
```
```
s.t. Ax=b
x≥ 0.
```
##### 

##### 

#####  (3)

Now, the first- and second-stage decisions are taken with knowledge about the uncertainty. The difference between
the expected wait-and-see value and the value of the recourse problem is known as theexpected value of perfect
information:
EVPI = EWS−VRP. (4)

The EVPI measures the expected loss of not knowing the exact outcome beforehand. It quantifies the value of having
access to an accurate forecast.

Finally, we introduce the concept of decision evaluation toquantify the performance of a candidate first-stage decision
xin the stochastic program (2). Theexpected resultofxis given by

```
V(x) =cTx+Eξ[Q(x, ξ(ω))]. (5)
```
This concept is used to compute the second classical measure. If the expectation in (2) is instead taken inside the
second-stage objective functionQ, we obtain the expected-value-problem:

```
minimize
x∈Rn
```
```
cTx+Q(x,Eξ[ξ(ω)])
```
```
subject to Ax=b
x≥ 0.
```
##### (6)

The solution to the expected-value-problemis known as theexpected value decision, and is denote byx ̄. Theexpected
resultof taking theexpected value decisionis known as theexpected result of the expected value decision:

```
EEV =cT ̄x+Eξ[Q( ̄x, ξ(ω))]. (7)
```
The difference between the value of the recourse problem andthe expected result of the expected value decision is
known as thevalue of the stochastic solution:

```
VSS = EEV−VRP. (8)
```
The VSS measures the expected loss of ignoring the uncertainty in the problem. A large VSS indicates that the second
stage is sensitive to the stochastic data.

The EVPI, VSS, and VRP are important tools when gauging the performance of a stochastic model. All of these
introduced measures are readily computed in the SPjl framework, which allows for easy analysis of user-defined
models. Next, we discuss how to calculate the VRP, EVPI, and VSS depending on the form of the sample spaceΩ.

2.2 The finite extensive form and sample average approximation

IfΩis finite, say withnscenarios of probabilityπ 1 ,... , πn, then we can represent (2) compactly as

```
minimize
x∈Rn,ys∈Rm
```
```
cTx+
```
```
∑n
```
```
s=
```
```
πsqTsys
```
```
subject to Ax=b
Tsx+W ys=hs, s= 1,... , n
x≥ 0 , ys≥ 0 , s= 1,... , n.
```
##### (9)

We refer to this problem as thefinite extensive form. It is often recognized in literature as thedeterministic equivalent
problem(DEP). Similar closed forms can be determined for the EVPI and the VSS. For smalln, it is viable to solve
this problem with standard linear programming solvers. Forlargen, decomposition approaches are required. In SPjl,
the user provides a description of the abstract stochastic model (2) and a separate description of the uncertainty model
ofξ. These are then combined internally to generate instances of the finite form (9), which are stored and solved
efficiently on a computer or a compute cluster.

IfΩis not finite, the stochastic program (2) is exactly computable only under certain assumptions [1]. However, it
is possible to formulate computationally tractable approximations of (2) using the finite form (9). The most common


approximation technique is thesample average approximation(SAA) [22]. Assume that we samplenscenarios
ωs, s= 1,... , nindependently fromΩwith equal probability. These scenarios now constitute a finite sample spaceΩ ̃
and we can use them to create a sampled model in finite extensive form (9). An optimal solution to this sampled model
approximates the optimal solution to (2) in the sense that the empirical average second-stage costVn=^1 n

```
∑n
s=1q
```
T
syˆs,
whereyˆs= arg miny∈Rm{Q(x, ξ(ωs))}, converges pointwise with probability 1 toVˆ=Eξ[Q(x, ξ(ω))]asngoes

to infinity [23]. Further, under certain assumptions it can be shown that

##### √

n(Vn−Vˆ)→N(0,Varξ[Q(ˆx, ξ)])in
distribution asngoes to infinity [24]. This result provides a basis for calculating confidence intervals around the VRP
of (2) [22, 25], as well as around the EVPI and the VSS.

2.3 Structure-exploiting solvers

Efficient methods for storing and solving finite stochastic programs on the form (9) are key for high-performance
stochastic programming computations. Therefore, this hasbeen a main focus in the development of the SPjl frame-
work. An important insight is that the finite extensive form (9) lends itself to block-decomposition approaches, which
allow the stochastic program to be efficiently distributed in memory. Moreover, structure-exploiting solvers can be
employed to solve the decomposed models efficiently. These approaches also readily extend to parallel settings where
the stochastic program is distributed over several computenodes. A key idea in the SPjl framework is to let the stor-
age of the stochastic program depend on the type of optimizerused to solve it. In this way, the memory structure is
optimized for the solver operation, and there is no redundant storage for other operations such as decision evaluation.
We say that the underlying structure of the stochastic program is induced by the solver. Henceforth, we will refer
to the treatment of (9) as one large optimization problem as thedeterministicstructure. This is the default structure
for standard third-party solvers. For block-decomposition approaches, we adopt the terminology introduced in [17]
and divide such strategies into two classes. In short, “verticalstrategies decompose a stochastic program by stages”
while “horizontalstrategies decompose a stochastic program by scenarios” [17]. In the following, we will introduce
two different solver algorithms that fall into these two categories and highlight the stochastic program structures they
induce.

2.3.1 The L-shaped algorithm

The L-shaped algorithm is an efficient cutting-plane methodfor solving the finite extensive form (9) by decomposing
into a master problem and a set of subproblems. The master problem has the form

```
minimize
x∈Rn,ys∈Rm
```
```
cTx+θ
```
```
subject to Ax=b
```
```
θ≥Q ̃(x)
x≥ 0 ,
```
whereQ ̃(x)is a lower bound on

```
Q(x) =
```
```
∑n
```
```
s=
```
```
πsQs(x).
```
Here, eachQs(x)is the optimal value to a subproblem of the form (1). The idea of the L-shaped algorithm is to
generate increasingly tight piecewise linear lower boundsonQ. We refer to the memory structure inferred by the
L-shaped algorithm henceforth as theverticalstructure.

During the L-shaped procedure, solution candidatesxkare generated by solving the master problem (11), which are
then used to parameterize subproblems of the form (1). Optimal dual variables in these subproblems are then used to
improve the bound ofQ(x)before the next solution candidatexk+1is computed. Specifically, it follows from duality
theory thatλTs(hs−Tsx), whereλsis the dual optimizer of (1), is a valid support function forQs(x), and hence,

```
∑n
```
```
s=
```
```
πsλTs(hs−Tsx)
```
is a valid support function forQ(x). In the original formulation of the L-shaped algorithm [12], the above result is
used at each iterationkto constructoptimality cutsby introducing

```
∂Qk=
```
```
∑n
```
```
s=
```
```
πsλTsTs qk=
```
```
∑n
```
```
s=
```
```
πsλTshs,
```

and add to the master problem as the constraint∂Qkx+θ≥qk. Aggregating the results from all subproblems in
this way is known as the single-cut approach. This was later extended to a multi-cut variant where separate cuts
are constructed for each subproblem [26]. If the iteratexkis not second-stage feasible, some subproblems will be
infeasible. We handle this by solving the auxilliary problem:

```
minimize
ys∈Rm
```
```
ws=eTv+s+eTv−s
```
```
subject to W ys+v+s−vs−=hs−Tsxk
ys≥ 0 , v+s ≥ 0 , vs−≥ 0.
```
##### (10)

Ifws> 0 , then subproblemsis infeasible for the current iteratexk. Further, it follows from duality theory that
σTs(hs−Tsx)≤ 0 , whereσsis the dual optimizer of (10), is necessary forxto be second-stage feasible. The above
result can be used to both check for second-stage infeasibility and constructfeasibility cutsby introducing

```
Fk=
```
##### 

##### 

##### 

```
σT 1 T 1
..
.
σfTTf
```
##### 

##### 

```
, fk=
```
##### 

##### 

##### 

```
σT 1 h 1
..
.
σfThf
```
##### 

##### 

##### 

for all infeasible subproblems 1 ,... , f. BecauseWhas a finite number of bases, finitely many feasibility cuts are
required to completely describe the set of feasible first-stage decisions [12]. The optimality cuts and the feasibility
cuts enter the master problem as follows:

```
minimize
x∈Rn
```
```
cTx+θ
```
```
subject to Ax=b
Fkx≥fk, ∀k
∂Qkx+θ≥qk, ∀k
x≥ 0.
```
##### (11)

The master problem is then re-solved to generate the next iteratexk+1, θk+1. This is repeated until the gap between the
upper boundQ(xk)and lower boundθk+1becomes small, upon which the algorithm terminates. Many variations can
be introduced to improve the performance of the L-shaped algorithm. We provide an overview of such improvements
available in SPjl in Section 4.1.

2.3.2 The progressive-hedging algorithm

The progressive-hedging algorithm was first introduced in [11]. In contrast to the L-shaped algorithm, applying
progressive-hedging to solve (9) yields a complete decomposition over thenscenarios. The method is a specialization
of the proximal-point algorithm [27], and convergence in the linear case (9) is derived in [11]. The main idea behind
this approach is to introduce individual first-stage decisionsxsto each scenario but force them to be equal. We then
relax (dualize) these consistency constraints and solve the corresponding augmented Lagrangian problem. In other
words, we consider the following problem:

```
minimize
xs∈Rn,ys∈Rm
```
```
∑n
```
```
s=
```
```
πs
```
##### (

```
cTxs+qsTys
```
##### )

```
subject to xs=ξ s= 1,... , n
Axs=b s= 1,... , n
Tsxs+W ys=hs, s= 1,... , n
xs≥ 0 , ys≥ 0 , s= 1,... , n.
```
##### (12)

The consistency constraintsxs=ξ, s= 1,... , nare callednon-anticipativebecause they make thexsindependent
of scenario and enforce the fact that the first-stage decision is known when the second-stage uncertainty is realized.
We refer to the memory structure inferred by the progressive-hedging algorithm henceforth as thehorizontalstructure.
Separability across thenscenarios is achieved by introducing the following regularized relaxation of each subproblem:

```
minimize
xs∈Rn,ys∈Rm
```
```
cTxs+qTsys+ρs(xs−ξ) +
```
```
r
2
```
```
‖xs−ξ‖^22
```
```
subject to Axs=b
Tsxs+W ys=hs
xs≥ 0 , ys≥ 0.
```

The algorithm now proceeds by iteratively alternating between generating new admissible solutionsxks, s= 1,... , n,
and an implementable solutionξk. In the two-stage setting, an admissible solution is feasible in every scenario, and an
implementable solution is consistent in the sense thatxs=ξfor alls. We obtain the implementable solution through
aggregation:

```
ξk=
```
```
∑n
```
```
s=
```
```
πsxks
```
and the Lagrange multipliers are updated scenario-wise through

```
ρks+1=ρks+r(xks−ξk).
```
Hence, the non-anticipative constraints are enforced while the dual variables converge. Progressive-hedging is a primal

dual algorithm that is run until both the primal gap‖ξk−ξk− 1 ‖^22 and the dual gap

```
∑n
s=1πs
```
##### ∥

```
∥xks−ξk
```
##### ∥

##### ∥^2

```
2 are small.
```
#### 3 StochasticPrograms.jl

In this section, we showcase the capabilities of SPjl. We first give a brief overview of the framework and introduce the
main functionality through a set of simple examples. Accompanying code excerpts are included. Next, we exemplify
the effectiveness of SPjl model creation by giving a compactdefinition of the farmer problem. Finally, we summarize
the algorithmic improvements and variations included in the framework.

SPjl extends the well-known JuMP syntax to support the definition of stages, decision variables, and uncertain pa-
rameters. Models are defined using the@stochastic_modelmacro. This creates a lightweight model object that can
be used to instantiate finite stochastic programs by supplying a description of the uncertain parameters. Specifically,
the user provides a list of discrete scenarios, or a sampler object capable of generating scenarios, to the model object.
The object then combines the model definition with the supplied uncertainty data and generates a finite stochastic
program instance. The instantiated stochastic program canthen be inspected, analyzed and solved in an interactive
Julia session. This is useful in educational settings, but also for reasoning about complex models on a small scale.
SPjl also supports reading problems specified in the SMPS format. For large-scale instances, SPjl provides scalable
block-structured instantiation and structure-exploiting solvers that can operate in parallel. In addition, operations such
as EWS calculation and decision evaluation are embarrassingly parallel over the subproblems. In other words, the
workload is readily decoupled into independent subtasks that can be executed in parallel. This is leveraged when
instantiating vertical or horizontal structures in distributed environments.

SPjl can be installed directly from the command line through Julia’s package manager
(pkg> add StochasticPrograms). Provided that a basic linear quadratic solver, such as GLPKorIpopt, is in-
stalled, all code examples in this paper can be repeated by copying the lines verbatim. A more extensive introduction
to the framework is given by the “Quick start” section of the online documentation^1.

3.1 A simple textbook example

Consider the following simple instance of (2)

```
minimize
x 1 ,x 2 ∈R
```
```
100 x 1 + 150x 2 +Eω[Q(x 1 , x 2 , ξ(ω))]
```
```
subject to x 1 +x 2 ≤ 120
x 1 ≥ 40
x 2 ≥ 20
```
##### (13)

where
Q(x 1 , x 2 , ξ(ω)) = max
y 1 ,y 2 ∈R

```
q 1 (ω)y 1 +q 2 (ω)y 2
```
```
s.t. 6 y 1 + 10y 2 ≤ 60 x 1
8 y 1 + 5y 2 ≤ 80 x 2
0 ≤y 1 ≤d 1 (ω)
0 ≤y 2 ≤d 2 (ω)
```
##### (14)

and the stochastic variable
ξ(ω) = (q 1 (ω) q 2 (ω) d 1 (ω) d 2 (ω))T

(^1) https://martinbiel.github.io/StochasticPrograms.jl/dev/


```
✞ Listing 1: Definition of (13) in SPjl. ☎
# Load SPjl framework
julia> using StochasticPrograms
# Create simple stochastic model
julia> simple_model =@stochastic_modelbegin
@stage 1 begin
@decision(model, x 1 >= 40 )
@decision(model, x 2 >= 20 )
@objective(model, Min, 100 *x 1 + 150 *x 2 )
@constraint(model, x 1 +x 2 <= 120 )
end
@stage 2 begin
@uncertainq 1 q 2 d 1 d 2
@recourse(model, 0 <= y 1 <= d 1 )
@recourse(model, 0 <= y 2 <= d 2 )
@objective(model, Max, q 1 *y 1 + q 2 *y 2 )
@constraint(model, 6 *y 1 + 10 *y 2 <= 60 *x 1 )
@constraint(model, 8 *y 1 + 5 *y 2 <= 80 *x 2 )
end
end;
✝ ✆
```
parameterizes the second-stage model. This is a recurring textbook example and correctness of our numerical results
can be verified by comparing with [1].

In SPjl, we create the model (13) in two steps. First, we formulate the optimization models as shown in Listing 1.
This creates a stochastic model where the two stages are given by the mathematical programs (13) and (14), expressed
using an enhanced JuMP syntax. The@decisionand@recourselines work as standard@variabledefinitions in
JuMP, but behind the scenes they also specify internal data dependencies between the first and second stage; and the
@uncertainline annotates the random parameters and defines a point of data injection. The code specifies how the
optimization models should be defined, but the actual model instantiation is deferred until we add a stochastic model
of the uncertainties. We will consider two different distributions ofξand use the same model objectsimple_model
from Listing 1 to instantiate stochastic programs. This is akey feature in SPjl. The underlying stochastic model (2)
object can be re-used to generate different finite stochastic program instances. Regardless of the distribution ofξ, a
stochastic program instance is always a finite program of theform (9). This allows us to evaluate the same problem
under different uncertainty models and to automatically adapt the underlying memory structure to optimize solver
performance.

3.2 Finite sample space

First, letξbe a discrete distribution, taking on the values

```
ξ 1 = (500 100 24 28)
```
```
T
, ξ 2 = (300 300 28 32)
```
```
T
```
with probability 0. 4 and 0. 6 respectively. In Listing 2, an instance of the stochastic program (13) is created for this
distribution. This code uses the model recipe created in Listing 1 to create second-stage models for each of the supplied
scenarios. Here, we have used the default scenario constructor@scenario, where data values are named in accordance
with the@uncertainannotation. The deterministic structure (extensive form)is used by default. Because this is a
small example, correctness of the generated problem is easily verified. We can now set an optimizer and solve the
model, as shown in Listing 3. The underlying memory structure can be set explicitly by setting theinstantiation
keyword to any of the supported structures during model instantiation. Alternatively, if an optimizer is chosen during
instantiation, an appropriate structure is chosen automatically. For example, if we instantiate the same problem with
an L-shaped optimizer the vertical structure is used instead, as can be seen in Listing 4. The same stochastic program
has now been decomposed into a first-stage master problem andtwo second-stage subproblems. For completeness we
also exemplify how the same problem is instantiated and solved using the progressive-hedging algorithm in Listing 5.

3.3 Infinite sample space

To demonstrate how SPjl handles continuous distributions for uncertain parameters, we assume that the uncertainties
in our simple example follow a multivariate normal distribution,ξ∼ N(μ,Σ). In general, there is no closed form solu-
tion of (2) whenξhas a continuous distribution. However, by the law of large numbers, a viable discrete approximation


✞ Listing 2: Instantiation of (13). ☎
# Create two scenarios
julia>ξ 1 =@scenarioq 1 =24.0q 2 =28.0d 1 =500.0d 2 =100.0probability =0.4;
ξ 2 =@scenarioq 1 =28.0q 2 =32.0d 1 =300.0d 2 =300.0probability =0.6;
# Instantiate without optimizer
julia> sp = instantiate(simple_model, [ξ 1 ,ξ 2 ])
Stochastic program with:
* 2 decision variables
* 2 scenarios of type Scenario
Structure: Deterministic equivalent
Solver name: No optimizer attached.
# Print to show structure of generated problem
julia> print(sp)
Deterministic equivalent problem
Min 100 x 1 + 150 x 2 - 9.6y 11 - 11.2y 21 - 16.8y 12 - 19.2y 22
Subject to
y 11 ≥0.
y 21 ≥0.
y 12 ≥0.
y 22 ≥0.
y 11 ≤500.
y 21 ≤100.
y 12 ≤300.
y 22 ≤300.
x 1 ∈Decisions
x 2 ∈Decisions
x 1 ≥40.
x 2 ≥20.
x 1 + x 2 ≤120.

- 60 x 1 + 6 y 11 + 10 y 21 ≤0.
- 80 x 2 + 8 y 11 + 5 y 21 ≤0.
- 60 x 1 + 6 y 12 + 10 y 22 ≤0.
- 80 x 2 + 8 y 12 + 5 y 22 ≤0.
Solver name: No optimizer attached.
✝ ✆

✞ Listing 3: Solving the finite extensive form of (13). ☎
julia> using GLPK
# Set the optimizer to GLPK
julia> set_optimizer(sp, GLPK.Optimizer)
# Optimize (deterministic structure)
julia> optimize!(sp)
# Check termination status
julia>@showtermination_status(sp);
termination_status(sp) = MathOptInterface.OPTIMAL
# Query optimal value
julia>@showobjective_value(sp);
objective_value(sp) = -855.
# Calculate EVPI
julia> EVPI(sp)
662.
# Calculate VSS
julia> VSS(simple_model, SimpleSampler(μ,Σ))
286.
✝ ✆


✞ Listing 4: Re-instantiation and optimization of (13) with an L-shaped optimizer ☎
# Instantiate with L-shaped optimizer
julia> sp = instantiate(simple_model, [ξ 1 ,ξ 2 ], optimizer = LShaped.Optimizer)
Stochastic program with:
* 2 decision variables
* 2 scenarios of type Scenario
Structure: Vertical
Solver name: L-shaped with disaggregate cuts
# Print to compare structure of generated problem
julia> print(sp)
First-stage
==============
Min 100 x 1 + 150 x 2
Subject to
x 1 ∈Decisions
x 2 ∈Decisions
x 1 ≥40.
x 2 ≥20.
x 1 + x 2 ≤120.
Second-stage
==============
Subproblem 1 (p =0.40):
Max 24 y 1 + 28 y 2
Subject to
y 1 ≥0.
y 2 ≥0.
y 1 ≤500.
y 2 ≤100.
x 1 ∈Known
x 2 ∈Known
6 y 1 + 10 y 2 - 60 x 1 ≤0.
8 y 1 + 5 y 2 - 80 x 2 ≤0.
Subproblem 2 (p =0.60):
Max 28 y 1 + 32 y 2
Subject to
y 1 ≥0.
y 2 ≥0.
y 1 ≤300.
y 2 ≤300.
x 1 ∈Known
x 2 ∈Known
6 y 1 + 10 y 2 - 60 x 1 ≤0.
8 y 1 + 5 y 2 - 80 x 2 ≤0.
Solver name: L-shaped with disaggregate cuts
# Set GLPK optimizer for the solving master problem and subproblems
julia> set_optimizer_attribute(sp, MasterOptimizer(),GLPK.Optimizer)
julia> set_optimizer_attribute(sp, SubproblemOptimizer(), GLPK.Optimizer)
# Optimize (vertical structure)
julia> optimize!(sp)
L-Shaped Gap Time: 0 : 00 : 02 ( 6 iterations)
Objective: -855.
Gap: 0.
Numberof cuts: 8
Iterations: 6
# Check termination status and query optimal value
julia>@showtermination_status(sp);
termination_status(sp) = MathOptInterface.OPTIMAL
julia>@showobjective_value(sp);
objective_value(sp) = -855.
✝ ✆


✞ Listing 5: Re-instantiation and optimization of (13) with aprogressive-hedging optimizer ☎
# Instantiate with progressive-hedging optimizer
julia> sp = instantiate(simple_model, [ξ 1 ,ξ 2 ],
optimizer = ProgressiveHedging.Optimizer)
Stochastic program with:
* 2 decision variables
* 2 scenarios of type Scenario
Structure: Horizontal
Solver name: Progressive-hedging with fixed penalty
# Print to compare structure of generated problem
julia> print(sp)
Horizontal scenario problems
==============
Subproblem 1 (p =0.40):
Min 100 x 1 + 150 x 2 - 24 y 1 - 28 y 2
Subject to
y 1 ≥0.
y 2 ≥0.
y 1 ≤500.
y 2 ≤100.
x 1 ∈Decisions
x 2 ∈Decisions
x 1 ≥40.
x 2 ≥20.
x 1 + x 2 ≤120.

- 60 x 1 + 6 y 1 + 10 y 2 ≤0.
- 80 x 2 + 8 y 1 + 5 y 2 ≤0.
Subproblem 2 (p =0.60):
Min 100 x 1 + 150 x 2 - 28 y 1 - 32 y 2
Subject to
y 1 ≥0.
y 2 ≥0.
y 1 ≤300.
y 2 ≤300.
x 1 ∈Decisions
x 2 ∈Decisions
x 1 ≥40.
x 2 ≥20.
x 1 + x 2 ≤120.
- 60 x 1 + 6 y 1 + 10 y 2 ≤0.
- 80 x 2 + 8 y 1 + 5 y 2 ≤0.
Solver name: Progressive-hedging with fixed penalty
julia> using Ipopt
# Set Ipopt optimizer for soving emerging subproblems
julia> set_optimizer_attribute(sp, SubproblemOptimizer(), Ipopt.Optimizer)
# Silence Ipopt
julia> set_optimizer_attribute(sp, RawSubproblemOptimizerParameter("print_level"), 0 )
# Optimize (horizontal structure)
julia> optimize!(sp)
Progressive HedgingTime: 0 : 00 : 05 ( 303 iterations)
Objective: -855.
Primal gap: 7.2622997706326046e-
Dual gap: 8.749063651111478e-
Iterations: 302
# Check termination status and query optimal value
julia>@showtermination_status(sp);
termination_status(sp) = MathOptInterface.OPTIMAL
julia>@showobjective_value(sp);
objective_value(sp) = -855.
✝ ✆


```
✞ Listing 6: Creating a sampled instance of (13) in SPjl. ☎
julia> using Distributions
# Define sampler object
julia>@samplerSimpleSampler = begin
N::MvNormal# Normal distribution
SimpleSampler(μ,Σ) = new(MvNormal(μ,Σ))
@sample Scenario begin
# Sample from normal distribution
x = rand(sampler.N)
# Create scenario matching @uncertain annotation
return@scenarioq 1 = x[ 1 ] q 2 = x[ 2 ] d 1 = x[ 3 ] d 2 = x[ 4 ]
end
end
# Create mean
julia>μ= [ 24 , 32 , 400 , 200 ];
# Create variance
julia>Σ= [2 0.5 0 0
0.5 1 0 0
0 0 50 20
0 0 20 30];
# Instantiate sampled stochastic program with 100 scenarios
julia> sp = instantiate(simple_model, SimpleSampler(μ,Σ), 100 )
Stochastic program with:
* 2 decision variables
* 100 scenarios of type Scenario
Structure: Deterministic equivalent
Solver name: No optimizer attached.
✝ ✆
```
can be obtained by sampling scenarios from the continuous distribution. In SPjl, we achieve this by creating a sampler
object associated with the defined scenario structure. In Listing 6, a sampler object for a multivariate distribution with

```
μ=
```
##### 

##### 

##### 

##### 24

##### 32

##### 400

##### 200

##### 

##### 

##### , Σ =

##### 

##### 

##### 

##### 2 0.5 0 0

##### 0 .5 1 0 0

##### 0 0 50 20

##### 0 0 20 30

##### 

##### 

##### 

is created and used to generate an instance of (13) with 100 sampled scenarios. Note that the same stochastic model
object defined in Listing 1 is used in Listing 6 to generate thesampled instance.

With the ability to instantiate sampled models with an arbitrary number of scenarios, we can adopt the SAA method-
ologies developed in [22] to calculate confidence intervalsaround the optimal value of (13) as well as around the
EVPI and the VSS. This is exemplified in Listing 7. These methods require re-solving sampled stochastic programs
multiple times and the accuracy of the solution is increasedby increasing the number of scenarios in the sampled
models. Consequently, the parallel capabilities of SPjl become significant as these subproblems can become too large
for single-core approaches. If multiple Julia processes are available, either locally or remotely, then the code in List-
ing 6 would automatically distribute the stochastic program on the available nodes in either a vertical or a horizontal
structure. Although not practically required for this small example, this leads to significant performance gains for
large-scale industrial models. See for example the scalingresults presented in Section 6.

#### 4 The farmer problem

To exemplify functional correctness, and allow for comparisons with similar tools, we consider the instructive farmer
problem by [1]. Listing 8 shows a suggested code excerpt for how the farmer problem can be defined in SPjl and
Listing 9 shows how the problem can be instantiated, solved,and analyzed using various solvers. The correctness of
the numerical values can be verified in [1]. For comparison, the same problem is defined in PySP as outlined in [17]
in about 100 lines spread out in separate files. Again, we stress that only 30 lines of Julia code are required to define
the farmer problem in SPjl. Moreover, the optimal value, as well as the EVPI and VSS, can be calculated interactively
in the same Julia session. This feature distinguished SPjl from other similar tools such as PySP. The time required to
solve the farmer problem using the L-shaped algorithm was 0. 57 seconds for both SPjl and PySP, measured on the
same master node as the numerical benchmarks presented in the paper. Hence, there is no performance decrease from
using SPjl instead of PySP for this small problem, with the added benefit of SPjl being more user-friendly.


✞ Listing 7: Approximately solving (13) whenξfollows a normal distribution. ☎
# Set optimizer to SAA
julia> set_optimizer(simple_model, SAA.Optimizer)
# Emerging stochastic programming instances solved by GLPK
julia> set_optimizer_attribute(simple_model, InstanceOptimizer(), GLPK.Optimizer)
# Set attributes that value solution speed over accuracy
julia> set_optimizer_attribute(simple_model, NumEvalSamples(), 300 )
# Set target relative tolerance of the resulting confidenceinterval
julia> set_optimizer_attribute(simple_model, RelativeTolerance(),5e-2)
# Approximate optimization using sample average approximation
julia> optimize!(simple_model, SimpleSampler(μ,Σ))
SAA gapTime: 0 : 00 : 03 ( 4 iterations)
Confidence interval: Confidence interval (p = 95 %): [-1095.65— -1072.36]
Relative error: 0.
Sample size: 64
# Check termination status
julia>@showtermination_status(simple_model);
termination_status(sp) = MathOptInterface.OPTIMAL
# Query optimal value
julia>@showobjective_value(simple_model);
objective_value(simple_model) = Confidence interval (p = 95 %): [-1095.65— -1072.36]
# Disable logging
julia> set_optimizer_attribute(simple_model, MOI.Silent(),true)
# Calculate approximate EVPI
julia> EVPI(simple_model, SimpleSampler(μ,Σ))
Confidence interval (p = 99 %): [32.96—144.51]
# Calculate approximate VSS
julia> VSS(simple_model, SimpleSampler(μ,Σ))
Warning: VSS is not statistically significant to the chosenconfidence level and tolerance
Confidence interval (p = 95 %): [-0.05—0.05]
✝ ✆

✞ Listing 8: Definition of the farmer problem in SPjl ☎
farmer =@stochastic_modelbegin
@stage 1 begin
@parametersbegin
Crops = [:wheat, :corn, :beets]
Cost =Dict(:wheat=> 150 , :corn=> 230 , :beets=> 260 )
Budget = 500
end
@decision(model, x[c in Crops] >= 0 )
@objective(model, Min, sum(Cost[c]*x[c] for c in Crops))
@constraint(model, sum(x[c] for c in Crops) <= Budget)
end
@stage 2 begin
@parametersbegin
Crops = [:wheat, :corn, :beets]
Required =Dict(:wheat=> 200 , :corn=> 240 , :beets=> 0 )
PurchasePrice =Dict(:wheat=> 238 , :corn=> 210 )
SellPrice =Dict(:wheat=> 170 , :corn=> 150 , :beets=> 36 , :extra_beets=> 10 )
end
@uncertainξ[c in Crops]
@recourse(model, y[p in setdiff(Crops, [:beets])] >= 0 )
@recourse(model, w[s in Crops∪[:extra_beets]] >= 0 )
@objective(model, Min, sum(PurchasePrice[p] * y[p] for p in setdiff(Crops, [:beets]))

- sum(SellPrice[s] * w[s] for s in Crops∪[:extra_beets]))
@constraint(model, minimum_requirement[p in setdiff(Crops, [:beets])],
ξ[p] * x[p] + y[p] - w[p] >= Required[p])
@constraint(model, minimum_requirement_beets,
ξ[:beets] * x[:beets] - w[:beets] - w[:extra_beets] >= Required[:beets])
@constraint(model, beets_quota, w[:beets] <= 6000 )
end
end
✝ ✆


✞ Listing 9: Instantiation, optimization, and analysis of the farmer problem in SPjl ☎
# Define the three yield scenarios
julia> Crops = [:wheat, :corn, :beets];
ξ 1 =@scenarioξ[c in Crops] = [3.0,3.6,24.0] probability = 1 / 3 ;
ξ 2 =@scenarioξ[c in Crops] = [2.5,3.0,20.0] probability = 1 / 3 ;
ξ 3 =@scenarioξ[c in Crops] = [2.0,2.4,16.0] probability = 1 / 3 ;
# Instantiate with GLPK optimizer
julia> farmer_problem = instantiate(farmer_model, [ξ 1 ,ξ 2 ,ξ 3 ], optimizer = GLPK.Optimizer)
# Optimize stochastic program (through extensive form)
julia> optimize!(farmer_problem)
# Inspect optimal decision
julia> ˆx = optimal_decision(farmer_problem)
3 -elementArray{Float64, 1 }:
170.
80.
250.
# Inspect optimal recourse decision in scenario 1
julia> optimal_recourse_decision(farmer_problem, 1 )
6 -elementArray{Float64, 1 }:
0.
0.
310.
48.
6000.
0.
# Inspect optimal value
julia> objective_value(farmer_problem)

- 108390.
# Calculate expected value of perfect information
julia> EVPI(farmer_problem)
7015.
# Calculate value of the stochastic solution
julia> VSS(farmer_problem)
1150.
# Initialize with vertical structure
julia> farmer_ls = instantiate(farmer_model, [ξ 1 ,ξ 2 ,ξ 3 ], optimizer = LShaped.Optimizer);
# Set GLPK optimizer for the solving master problem
julia> set_optimizer_attribute(farmer_ls, MasterOptimizer(), GLPK.Optimizer);
# Set GLPK optimizer for the solving subproblems
julia> set_optimizer_attribute(farmer_ls, SubproblemOptimizer(), GLPK.Optimizer);
# Solve using L-shaped
julia> optimize!(farmer_ls)
L-Shaped Gap Time: 0 : 00 : 00 ( 6 iterations)
    Objective: -108390.
    Gap: 0.
    Numberof cuts: 14
    Iterations: 6
# Initialize with horizontal structure
julia> farmer_ph = instantiate(farmer_model, [ξ 1 ,ξ 2 ,ξ 3 ],
    optimizer = ProgressiveHedging.Optimizer);
# Set Ipopt optimizer for soving emerging subproblems
julia> set_optimizer_attribute(farmer_ph, SubproblemOptimizer(), Ipopt.Optimizer)
# Silence Ipopt
julia> set_optimizer_attribute(farmer_ph, RawSubproblemOptimizerParameter("print_level"), 0
# Solve using progressive-hedging
julia> optimize!(farmer_ph)
Progressive HedgingTime: 0 : 00 : 05 ( 86 iterations)
    Objective: -108390.
    Primal gap: 3.984637579811031e-
    Dual gap: 5.634811373041405e-
    Iterations: 85
✝ ✆


4.1 Advanced solver configurations in the SPjl framework

The SPjl framework includes a variety of customizable improvements to the L-shaped and progressive-hedging al-
gorithms. The possible variations of the classical algorithms included in the framework range from efficient imple-
mentations of influential research papers [28, 29, 30] to novel variants developed by the framework authors [31] or
others [32, 33, 34]. We provide a summary of the improvementsavailable for both L-shaped and progressive-hedging.
In brief, each algorithm has a set of options that can be varied through a simple interface. In all examples, it is as-
sumed that a given stochastic program instancesphas been instantiated with an appropriate optimizer. We canthen
useset_optimizer_attribute(sp, option, value)to customize the optimizer algorithm used bysp.

4.1.1 L-shaped

The L-shaped solver suite of SPjl includes a large set of customizable options. These are summarized below.

Regularization: A Regularization procedure limits the candidate search toa neighborhood of the current best iterate
in the master problem. It tends to result in more effective cutting planes and improved performance of the L-shaped
algorithm. Moreover, regularization enables warm-starting the L-shaped procedure with initial decisions. We have
previously covered the regularization procedures in SPjl more in depth in [35].

The SPjl framework includes the following regularizations: Trust-region regularization [29], Regularized decomposi-
tion [28], Level set regularization [30]. Since the two latter techniques involve solving problems with quadratic penalty
terms, the SPjl framework also provide an option for replacing quadratic penalties with various linear approximations,
if only a linear solver is available.

Aggregation: Cut aggregation can reduce communication overhead and load imbalance and yield major performance
improvements in distributed settings. In the classical L-shaped algorithm [12], all cuts are aggregated every iteration.
The authors of [26] suggested a multi-cut variant where cutsare added separately in a disaggregate form, which on
average yields faster convergence. We recently explored a novel set of aggregation approaches [31], which are all
included in SPjl.

Consolidation: Cut consolidation, as proposed by [33], is also implemented in SPjl to reduce load imbalance by
removing stale cuts from the master.

Execution: In a distributed environment with multiple Julia processes, the execution policy of the L-shaped algorithm
can be executed in a serial, synchronous or asynchronous mode. The synchronous variant runs the L-shaped algorithm
in parallel using a map-reduce pattern each iteration. The asynchronous scheme is appropriate in a heterogeneous
environment where some workers may finish slower than others. We show how these algorithm policies can be applied
to increase performance on large-scale problems in Section6.

4.1.2 Progressive-hedging

The progressive-hedging solver suite shares a few options with the L-shaped suite. First, as each subproblem in the
progressive-hedging procedure includes a quadratic penalty term, the same linear approximations as for L-shaped
regularizations can be applied. Second, just like the L-shaped solvers, the progressive-hedging algorithms can be run
serially, synchronously or asynchronously.

Penalization: The convergence rate of the progressive-hedging algorithm is sensitive to the choice of the penalty
parameterr. The SPjl framework supports both a fixed penalty parameter and the adaptive strategy introduced in [32].

#### 5 Implementation details

In this section, we provide a summary of the main software innovations in SPjl. We also discuss the implementation
of the framework’s distributed capabilities. The inner workings of SPjl are primarily based on two ideas: deferred
model instantiation and data injection. In brief, a model definition in SPjl is a recipe for how to use data structures
when building optimization models, while the actual model creation is deferred until data is provided. When a specific
model is instantiated, the provided data is injected where required to construct the model. The main effect of this
approach is that the stochastic model formulation is separated from the design of stochastic data parameters, which
makes the SPjl framework versatile and flexible to use. For instance, it is possible to test small instances of a model
locally to ensure that it is properly defined, and then run thesame model in a distributed environment with a large set
of scenarios. Deferred model instances and data injection also play a large role when distributing stochastic program
instances in memory.


5.1 Deferred model instantiation

The advantages of deferred model instantiation is a smallermemory footprint and the ability to create various structures
that use the first- and second-stage recipes as building blocks in a clever way. Examples include the deterministic,
vertical, and horizontal structures, as well as wait-and-see problems and expected-value problems. The technique is
also a premise for implementing data injection. In contrastto standard JuMP models, SPjl models defined through the
@stagemacros are not necessarily instantiated immediately. Instead, the user-defined Julia code that constructs the
optimization problems is stored in lambda functions as model recipes. In other words, instead of creating and storing
a JuMP object, the lines of code required to create the JuMP object is stored. This is achievable since Julia code is
itself a data structure defined within the Julia language.

Deferred model instantiation is made possible through metaprogamming and the automatic reformulation bridges
introduced inMathOptInterface[16]. These techniques allow us to add linking constraints between the stages that
adhere to the data dependencies defined by the user. During model creation, any@decisionline in a@stagedefinition
creates special JuMP variables whose behaviour depends on the context of the instantiation. Any variable defined in
this way can be included in@constraintdefinitions in subsequent stages. See for example Listing 1,where the last
two constraint definitions in the second stage include references tox 1 andx 2 which were defined with@decisionin the
first stage. Next, we will discuss in more detail how instantiation is implemented for the main underlying structures:
deterministic, vertical, and horizontal. In addition, we explain how decision evaluation is implemented in the different
structures.

5.1.1 Deterministic structure

We construct the extensive form of a finite model (9) in steps using the stored model recipes. First, we generate
the first-stage model in full using the corresponding recipe. Next, we process all available scenarios iteratively. For
each scenario, we apply the second-stage recipe and append the resulting subproblem to the extensive model. In
this context, any variables defined with@decisionin the first stage are treated as regular JuMP variables. Before
generating the subsequent scenario problem, we internallyannotate the variables and constraints to associate them
with the scenario they originated from. This labeling is visible in the printout shown in Listing 2. During decision
evaluation, all variables defined with@decisionare fixed to their corresponding values. The deterministic equivalent
problem is then solved as usual, giving exactly (5).

5.1.2 Vertical structure

The vertical structure, introduced in Section 2.3.1, is also instantiated in steps. First, the first-stage master problem (11)
is created using the corresponding recipe. Here, the@decisionvariables are again treated as regular JuMP variables.
Next, subproblem instances of the form (1) are created for each possible scenario using the second-stage recipe. During
second-stage generation, first-stage variables annotatedwith@decisionenter the model as so calledknown decisions.
These are not optimization variables, but rather parameters with given values. This design reflects the fact that the first-
stage decisions have already been taken when the second stage is reached. The values of the first-stage decisions can
be entered into the second-stage constraints in which they appear through automatic reformulation bridges. Internally,
all decisions defined in the first-stage are made known to the second stage by the@stochastic_modelmacro. It is
also possible to explicitly add@knownannotations to the second-stage definition to mark variables that originate from
previous stages.

The subproblems are either stored in vector format on the master node or distributed on remote nodes as described in
Section 5.3. We distribute new scenarios and generated subproblems as evenly as possible on remote nodes to achieve
load balance. During decision evaluation, all variables defined with@decisionare fixed to their corresponding values
in the first stage. Further, these values are communicated toall subproblems, that can then update their respective
second-stage constraints. The first stage and second stage problems are then solved separately, in parallel if possible,
and the results are map-reduced to form (5).

5.1.3 Horizontal structure

Instantiation of the horizontal structure introduced in Section 2.3.2 is similar to instantiation of the vertical structure.
They differ in that there is no master problem and in that the subproblems have the structure given in (12) instead
of (1). The process for generating subproblems of this wait-and-see form is equivalent to one iteration of the finite
extensive form generation. In short, the first-stage recipeis applied, followed by applying the second-stage recipe
on the scenario data corresponding to the subproblem. Now, the variables defined with@decisionare again treated
as standard JuMP variables. Note that generation of the expected-value-problem (6) is equivalent to generating a
wait-and-see model on the expected scenario of all available scenarios. The implementable solutionξthat enter the


```
✞ Listing 10: Simple showcase of data injection in SPjl ☎
@stochastic_modelbegin
@stage 1 begin
@decision(model, x)
end
@stage 2 begin
@parametersd
@knownx
@uncertainξ
@recourse(model, y <= d)
@constraint(model, x + y <=ξ)
end
end
✝ ✆
```
horizontal form (12) through the non-anticipative constraints is added as a known decision to the subproblems. During
the progressive-hedging procedure, the value ofξcan then be updated efficiently through bridges. This designis
also used to implement proximal terms in the regularized variants of the L-shaped algorithm. Decision evaluation is
performed similar to the other structures. In each subproblem, the first-stage decisions are fixed to their corresponding
values and the subproblem is solved as usual. The results arethen map-reduced to form (5). Again, the decision
evaluation process is embarrassingly parallel in a distributed environment.

5.2 Data injection

Data injection is the second software pattern used to separate model and data design in SPjl. The aim is to make
an object independent of how its dependencies are created. In SPjl, the dependencies consist of the data required to
construct the optimization problems as described by the model recipes. The data includes uncertain parameters, as well
as first-stage decisions and deterministic parameters. By adopting this approach, users of SPjl can focus on the design
of the optimization model and the uncertainty model separately, while the framework is responsible for combining
these designs into actual stochastic program instances. Inthe following, we describe the data injection functionality
in more detail.

When an SPjl model is formulated using@stochastic_model, special annotations are used inside the@stageblocks
to specify points of data injection. These annotations inform the framework which parameters are necessary to con-
struct the model according to the@stochastic_modeldefinition. The@stagemacro transforms the stage blocks into
anonymous lambda functions that map supplied data into optimization problems. Internally, when the user wants to
instantiate the defined SPjl model, the required data is passed to the stored lambda functions according to one of the
instantiation procedures outlined in the previous section.

We give a short review of the different types of data dependencies that can be specified in an SPjl model. Consider the
simple second-stage formulation in Listing 10, which includes several data injection annotations.

Deterministic data: The@parametersannotation specifies scenario-independent data, i.e., deterministic parameters
that are the same across all scenarios. Default parameter values can be specified inside the@parametersblock. Other-
wise, the values must be supplied during instantiation.

Uncertain data: The@uncertainannotation specifies the scenario-specific data. The scenario-dependent values are
either created and supplied directly by the user or by some user-defined sampler object that models the uncertainty.

Decisions: The@knownannotation makes the first-stage decisionxavailable in the second-stage. Note again that
@knownannotations are implicitly added by@stochastic_modelbecause of the@decisionxin the first stage. When
the second-stage generator is run, the framework will have already created a decision variablexusing the first-stage
generator, either as a standard JuMP variable or as a fixed known decision. All such first-stage variables are injected
into the second-stage generator. These can then be used as ifthey were ordinary JuMP variables. See for example the
last@constraintdefinition in the second stage of Listing 10.

Models: Themodelkeyword is a placeholder for a JuMP object that stores the actual optimization problem. In a
deterministic structure, the model object is the same in every generator call. In the block-decomposition structures,
the generators are instead applied to multiple JuMP models that form subproblems.

The use of data injection adds versatility to the framework.The user is only restricted to use themodelkeyword in
the JuMP macro calls. Otherwise, all JuMP features are supported in the stage blocks. Also, there is no restriction on


the scenario data types. Hence, instead of a simple structure with fields, it is possible to define a more complex data
type that for example performs calculations at runtime to determine optimization parameters. In addition, any Julia
methods defined on the scenario type become available in the stage blocks. This allows the user to design complex
models of the uncertainty orthogonally to the definition of the stochastic program.

Because the model definition is decoupled from the data, it ispossible to send the model recipe to a remote process
where the scenario data resides and create the model from there. This is the foundation of the distributed implementa-
tion described next.

5.3 Distributed computations

SPjl has distributed capabilities for both modeling, analysis and optimization. All implementations rely on the
Distributedmodule in Julia. This allows us to develop SPjl using high-level abstractions that utilize the efficient
low-level communication protocols in Julia. In this way, the same codebase can be used to distribute computations
locally, using shared-memory, and remotely, in a cluster orin the cloud.

Distributed computing in Julia is centered around the concepts of remote references and remote calls. Remote refer-
ences are used to administer which node particular data resides on and to provide the remaining processes access to
the remote data. Remote calls are used to schedule tasks on the nodes. Any process canwaiton a remote reference,
which blocks until data can be fetched, and thenfetchthe result when it is ready. TheRemoteChannelobjects are
special remote references where processes can alsoput!data. Besides, specialized channel objects can be designed
for specific data types. This feature is used frequently in the implementation of the distributed structured solvers.

5.3.1 Distributed stochastic programs

The distributed capabilities of SPjl were designed with theaim to minimize the amount of data passing. This is mainly
achieved through the deferred instantiation and data injection techniques outlined above. In principle, a stochastic
program can be instantiated in a distributed environment bypassing all necessary data to each worker node. However,
the data injection technique is independent of the way data is created. Therefore, a far more efficient approach is to let
the workers generate the necessary scenario data and the optimization models themselves, with minimal data passing.
This is possible since SPjl has support for passing lightweight sampler objects capable of randomly generating scenario
data, such as the one in defined in Listing 6, along with passing the lightweight model recipes created in the@stage
blocks. Scenario data and subproblems can then be generatedin parallel on the worker nodes. The master keeps track
of the scenario distribution and ensures that new scenariosand subproblems are generated on available workers in a
way that promotes load-balance.

If multiple Julia processes are available, then any instantiated stochastic program in SPjl is automatically distributed
in memory according to either a vertical structure or a horizontal structure. In a vertical structure, the master node
administers the first-stage problem and schedules tasks anddata transfers. In a horizontal structure, the master node is
only responsible for task scheduling and data transfers. Aside from distributing the models in memory, SPjl parallelizes
as many computations as possible. In many cases, speedups stem from subtasks being embarrassingly parallel over the
independent subproblems. For example, this occurs during decision evaluation and calculation of EVPI and VSS. In
these instances, the master schedules the same computationtasks on all workers using remote calls and then initiates
any necessary reductions after the workers have finished using a standard map-reduce pattern. The more involved
parallelization strategies in SPjl relate mostly to the structure-exploiting distributed solvers described in more detail in
Appendix 5.3.2.

5.3.2 Distributed structured optimization algorithms

The implementations of the distributed structured solversare also centered around remote calls and channels. Here,
remote calls are used to initiate running tasks on every worker node, and the algorithm logic is driven by having the
master and worker tasks wait on and write/fetch to/from specialized queue channels.

In the case of the L-shaped method, whenever the master node re-solves the master problem (11), it writes the new
decision vector to a specializedDecisionchannel. It then sends a corresponding index to aWorkchannel on every
remote node. Every worker continuously fetches tasks from itsWorkchannel and uses the acquired index to fetch the
latest decision vector from the master. Every new decision candidate infers a batch of subproblems to solve for each
worker. After a worker has solved a subproblem (1), it sends the computed cutting planes to aCutQueuechannel on
the master. The master continuously fetches cuts from theCutQueueand appends them to the master problem. In the
synchronous variant, the master only updates after all workers have finished their work for the current iteration. In
other words, the synchronous algorithm is driven by the master node initiating and waiting for worker tasks through


remote calls. In the asynchronous version, the master updates after it has receivedκncuts, wherenis the total
number of subproblems. Timestamps are communicated throughout to keep track of the algorithm history and allow
synchronized convergence checks. All subproblems are solved to completion each iteration regardless of the value of
κ, to be able to check convergenceproperly. When the master has received all cuts corresponding to a specific iteration,
it performs a convergence check and terminates if appropriate. For clarity, the procedure is illustrated in Fig. 1. A
similar design is used to implement synchronous and asynchronous variants of the progressive-hedging algorithm.

#### 6 Numerical benchmarks

We now evaluate the distributed performance of SPjl by benchmarking the structure-exploiting solvers on a large-scale
planning problem. The numerical experiments are performedin a multi-node setup where a laptop computer acts as
the master node and a desktop compute server of up to 32 cores provides worker nodes.

6.1 The SSN problem

We evaluate the solvers on the telecommunications problem SSN, first introduced in [36]. This problem is often
included in similar benchmarks [29, 34]. The SSN problem is formulated to plan bandwidth capacity expansion in a
network before customer demands are known. The problem is freely available in the SMPS format^2. The problem has
89 decision variables in the first stage, and 706 variables and 175 constraints in the second stage. We first run an SAA
procedure to gauge the number of scenarios required to obtain a stable solution. The results are shown in Figure 2.
There is no visible improvement after 6000 scenarios. Moreover, the confidence interval around the optimal value
is considered relatively tight at this point and is consistent with similar experiments [25]. With 6000 scenarios, the
extensive form of the SAA model has 4.2 million variables and5.3 million constraints, and about 20 minutes is required
to build and solve the extensive form using Gurobi [10]. Fromthis baseline, we run the distributed benchmarks.

6.1.1 Benchmarks

We evaluate the structured solvers by solving distributed SSN instances of 6000 scenarios. Benchmarks are performed
using the Julia package BenchmarkTools.jl, which schedules multiple solve procedures and reports median compu-
tation times. Every solver runs until convergence criteriaare reached with a relative tolerance of 10 −^2. The master
node is a laptop computer with a 2.6 GHz Intel Core i7 processor and 16 GB of RAM. We spawn workers on a remote
multi-core machine with two 3.1 GHz Intel Xeon processors (total 32 cores) and 128 GB of RAM. The two machines
were 30 kilometers apart at the time of the experiments. The time required to pass a single decision or optimality
cut at this distance is about 0. 01 seconds. Hence, the communication latency is small, but notnegligible as will be
apparent in the results. For single-core experiments we only run the procedures once because the time to convergence
is long and the measurement variance becomes relatively small. Throughout, the Gurobi optimizer [10] is used to
solve emerging subproblems.

We first benchmark a set of L-shaped solvers. The nominal method is the multi-cut L-shaped algorithm without any
advanced configuration. On average, this algorithm requires 19 iterations and92 000optimality cuts to solve an SSN
instance of 6000 scenarios. This takes just over 30 minutes on the master node under serial execution. We run a
strong scaling test where the number of worker cores on the remote machine is doubled in size up to 32 cores. Apart
from multi-cut L-shaped, we also evaluate two variants withadvanced algorithm policies. Specifically, one solver is
configured to use trust-region regularization and partial cut aggregation with 32 cuts in each bundle. This aggregation
scheme is static; the cuts are partitioned into groups of 32 in the same order each iteration. The second solver is
configured to use level-set regularization and K-medoids cluster aggregation. This is a dynamic aggregation scheme
where the cuts are clustered using the K-medoids algorithm based on a generalized cut distance matrix each iteration.
We fix the partitioning scheme of the dynamic method after thefirst five iterations, as outline in [31]. All solvers are
configured to use synchronous execution. The results from these experiments are shown in Figure 3.

First, we just consider the multi-cut method. The initial scaling is very poor with almost no speedup. We then observe
speedups up to eight cores upon which the scaling curve flattens. The primary sources of inefficiency in distributed
L-shaped algorithms are communication latency and load imbalance. This is especially true for multi-cut L-shaped
because all cuts are passed separately and the master increases in size by the maximum number of constraints possible
each iteration. We re-ran the two-core experiment on the master node with local threads as workers. In other words,
without communication overhead. The time to convergence was then about 18 minutes. With 0. 01 seconds required
to pass a single cut and92 000cuts passed in total, this accounts for the extra 15 minutes required to converge in the
multi-node setup. Therefore, we can conclude that much of the inefficiency stems from communication latency. The

(^2) https://core.isrd.isi.edu/


```
D: x^0 · · ·
C: · · ·
```
```
minimizex∈Rn cTx+
s.t.Ax=b
x≥ 0
```
```
Master
```
```
W 1 : 1 · · ·
S 1 : minimize
yi∈Rm q
```
```
Tsys
s.t.W ys=hs−Tsx 0
ys≥ 0
```
```
Worker 1
```
```
Wr: 1 · · ·
Sr: minimize
yi∈Rm q
sTys
s.t.W ys=hs−Tsx 0
ys≥ 0
```
```
Workerr
```
# · · ·

```
pass pass
fetch fetch
```
```
(a) Master sends task to workers. Workers fetch latest decision vector.
```
```
D: x 0 x 1 · · ·
C: · · ·
```
```
minimizex∈Rn cTx+
∑n
s=
```
```
θs
s.t.Ax=b
∂Qx+θs≥q, i= 1,... , n
x≥ 0
```
```
Master
```
```
W 1 : 1 2 · · ·
S 1 : minimizey
1 ∈Rm q
1 Ty 1
s.t.W y 1 =h 1 −T 1 x 0
y 1 ≥ 0
```
```
Worker 1
```
```
Wr: 1 2 · · ·
Sr: minimizey
r∈Rm q
```
```
Tryr
s.t.W yr=hr−Trx 0
yr≥ 0
```
```
Workerr
```
# · · ·

```
pass
```
```
pass pass pass
```
(b) Workers solve subproblems and send cuts to master. Master problem re-solved afterκncuts have been received.
Master sends new task to workers when a new decision vector isready.

```
D: x 0 x 1 · · ·
C: · · ·
```
```
minimizex∈Rn cTx+
∑n
s=
```
```
θs
s.t.Ax=b
∂Qx+θs≥q, i= 1,... , n
x≥ 0
```
```
Master
```
```
W 1 : 1 2 · · ·
S 1 : minimize
y 1 ∈Rm q
1 Ty^1
s.t.W y 1 =h 1 −T 1 x 0
y 1 ≥ 0
```
```
Worker 1
```
```
Wr: 1 2 · · ·
Sr: minimize
yr∈Rm q
```
```
Tryr
s.t.W yr=hr−Trx 1
yr≥ 0
```
```
Workerr
```
# · · ·

```
pass fetch
```
```
|Q−Θ| ≤τ(ǫ+|Q|)?
```
```
(c) Convergence check when all cuts have been received. Ready workers fetch latest decision. Procedure continues.
```
```
Figure 1: Asynchronous L-shaped procedure
```

```
0 2000 4000 6000 8000
```
```
8
```
```
9
```
```
10
```
```
11
```
```
12
```
```
Number of samples N
```
```
Confidence interval CI
```
```
SSN planning problem
```
```
Figure 2:90%Confidence intervals around the optimal value of the SSN problem as a function of sample size.
```
```
1 2 4 8 16 32
```
```
5
```
```
10
```
```
15
```
```
20
```
```
25
```
```
30
```
```
35
```
```
40
```
```
45
```
```
Number of cores
```
```
Computation Time T [min]
```
#### Strong scaling

```
Multi-cut L-shaped
L-shaped with trust-region regularization and partial cutaggregation
L-shaped with level-set regularization and K-medoids cluster aggregation
```
Figure 3: Median computation time required for different L-shaped algorithms to solve SSN instances of 6000 scenar-
ios, as a function of number of worker cores. All experimentswere run under synchronous execution.


```
1 2 4 8 16 32
```
```
10
```
```
20
```
```
30
```
```
40
```
```
50
```
```
60
```
```
70
```
```
80
```
```
90
```
```
Number of cores
```
```
Computation Time T [min]
```
#### Strong scaling

```
Progressive-hedging with adaptive penalty parameter
```
Figure 4: Median computation time required for the progressive-hedging algorithm to solve SSN instances of 6000
scenarios, as a function of number of worker cores. All experiments were run under synchronous execution.

fact that the scaling curve flattens stems mostly from load imbalance. In the final iterations, most of the time is spent
solving the now large master problem or passing cuts, so the worker nodes are not utilized optimally.

Next, we consider the advanced methods. The distributed performance is significantly improved compared to the
multi-cut method. The main reason for this is that cut aggregation reduces both communication latency and load
imbalance. Because cuts are aggregated, less data is passedeach iteration. Further, the master problem does not
grow as fast. Hence, the workload is more evenly spread out between master and workers, which improves parallel
performance. In this particular case, the more advanced aggregation scheme yields slightly better performance, but it
could also hold that level-set regularization is more performant than trust-region regularization on the SSN problem.
Even with cut aggregation, the size of the master eventuallyexceeds the size of the subproblems and data passing still
becomes a bottle-neck as the number of cores increase. Therefore, the scaling curves still flatten for larger numbers of
cores. We do not claim that these configurations are the best possible. We can for example note that they are not optimal
for single-core execution where both variants are outperformed by the multi-cut method. Also, the parallel efficiency
increase as workers are added is not uniform. This is becausethe aggregation schemes are more optimal for some
work granularities. We could possibly improve the convergence times further by parameter tuning. For this particular
configurations, we could also let a processor on the remote machine act as the master node and remove communication
latency all together. However, we believe that our results are a strong encouragement for the distributed capability of
the SPjl framework. With non-negligible communication latency we are able to solve a large-scale planning problem
in just over a minute by employing some of the readily available algorithm policies in the framework. This can be seen
as a proof of concept for running industrial planning problems in a modern cloud architecture.

We tested the algorithms with asynchronous execution as well, but saw no performance improvements. Even though
there is communication latency between the master node and the remote node, worker performance is even. Moreover,
the subproblems are equally difficult to solve. There is therefore no immediate gain from introducing asynchrony and
the overhead from doing so decreases performance. The asynchronous variants are expected to yield better perfor-
mance in a more heterogeneous environment with stalling workers.

Next, we evaluate the performance of the progressive-hedgingmethods. Using the nominal method, we did not observe
convergence even after long waiting times. Using the adaptive penalty policy eventually yields convergence. We
configure the solvers to use adaptive penalty and synchronous execution and run the same strong scaling experiment
as for the L-shaped methods. The results are shown in Figure 4.

Although at much worse time-to-solution than the L-shaped methods initially, the distributed progressive-hedging al-
gorithm displays great scaling and outperforms the multi-cut L-shaped method after 16 cores. The efficiency probably
stems from the problem being load-balanced across the workers. Communication latency again becomes a bottle-neck
at 32 cores from which we attribute the worsened scaling. Again, the subproblems appear equally difficult as there
were no stalling workers. Consequently, we did not observe any speedups from running the asynchronous variant.


The time to convergence is notably large and the progressive-hedging method is consistently outperformed by the
advanced L-shaped methods. This is not surprising as we havespent more time on L-shaped improvements. Future
work includes further algorithmic improvements to the progressive-hedging algorithms.

#### 7 Conclusion

In this work, we have presented an open-source framework,StochasticPrograms.jl, for large-scale stochastic pro-
gramming. It is written entirely in Julia and includes both modeling tools and solver algorithms. The framework is
designed for distributed computations and naturally scales to high-performance clusters or the cloud. By using the
extensive form, which is efficiently generated using metaprogramming techniques, stochastic program instances can
be solved using open-source or commercial solvers. Throughdeferred model instantiation, data injection, and clever
algorithm policies, the framework can operate in distributed architectures with minimal data passing. In addition, sev-
eral analysis tools and stochastic programming constructsare included with efficient implementations, many of which
can run in parallel.

The framework also includes a solver suite of scalable algorithms that exploit the structure of the stochastic pro-
grams. The structured solvers are shown to perform well on large-scale planning problems. High parallel efficiency
is achieved for distributed L-shaped methods using cut aggregation techniques and regularizations. Moreover, dis-
tributed progressive-hedging algorithms are acceleratedusing an adaptive penalty procedure. The solver suites are
made modular through a policy-based design, so that future improvements can readily be added.

There are several directions for future additions to the framework. First, SPjl does not yet fully support multi-stage
problems. We have finished an infrastructure for representing multi-stage problems in a way that leverages the two-
stage design. Ongoing work involves designing a suitable Julian syntax for encoding transitive probabilities in a
multi-stage scenario tree. Second, we will consider further algorithmic improvements to the existing L-shaped and
progressive-hedging solvers. We also want to explore alternative sample-based approaches to the SAA method where
the sampling is instead performed inside the structure-exploiting algorithm procedure. Examples of such approaches
include L-shaped with importance sampling [37] or stochastic decomposition [38].

The framework is well-tested through continuous integration and is freely available on Github^3. A comprehensive
documentation is included^4. The modeling framework,StochasticPrograms.jl, exists as a registered Julia package,
which can be installed and run in any interactive Julia session.

#### References

```
[1] John R. Birge and François Louveaux.Introduction to Stochastic Programming. Springer New York, 2011.
[2] Stein-Erik Fleten and Trine Krogh Kristoffersen. Stochastic programming for optimizing bidding strategies of a
nordic hydropower producer.European Journal of Operational Research, 181(2):916–928, 2007.
[3] Nicole Gröwe-Kuska and Werner Römisch. Stochastic unitcommitment in hydrothermal power production
planning. InApplications of Stochastic Programming, pages 633–653. Society for Industrial and Applied Math-
ematics, 2005.
[4] C. G. Petra, O. Schenk, and M. Anitescu. Real-Time Stochastic Optimization of Complex Energy Systems on
High-Performance Computers.Computing in Science Engineering, 16(5):32–42, 2014.
[5] P. Krokhmal, S. Uryasev, and G. Zrazhevsky. Numerical comparison of conditional value-at-risk and conditional
drawdown-at-risk approaches: Application to hedge funds.InApplications of Stochastic Programming, pages
609–631. Society for Industrial and Applied Mathematics, 2005.
[6] Stavros A. Zenios. Optimization models for structuringindex funds. InApplications of Stochastic Programming,
pages 471–501. Society for Industrial and Applied Mathematics, 2005.
[7] Warren B. Powell. An operational planning model for the dynamic vehicle allocation problem with uncertain
demands.Transportation Research Part B: Methodological, 21(3):217–232, 1987.
[8] Warren B. Powell and Huseyin Topaloglu. Fleet management. InApplications of Stochastic Programming, pages
185–215. Society for Industrial and Applied Mathematics, 2005.
[9] Andrew Makhorin. Gnu linear programming kit, 2020.https://www.gnu.org/software/glpk/.
```
(^3) https://github.com/martinbiel/StochasticPrograms.jl
(^4) https://martinbiel.github.io/StochasticPrograms.jl/dev/


[10] Gurobi Optimization. Gurobi optimizer reference manual, 2020.http://www.gurobi.com.

[11] R. T. Rockafellar and Roger J.-B. Wets. Scenarios and policy aggregation in optimization under uncertainty.
Mathematics of Operations Research, 16(1):119–147, 1991.

[12] R. Van Slyke and Roger J.-B Wets. L-Shaped Linear Programs with Applications to Optimal Control and Stochas-
tic Programming.SIAM Journal on Applied Mathematics, 17(4):638–663, 1969.

[13] Jeff Bezanson, Alan Edelman, Stefan Karpinski, and Viral B. Shah. Julia: A fresh approach to numerical com-
puting.SIAM Review, 59(1):65–98, 2017.

[14] Martin Biel, Arda Aytekin, and Mikael Johansson. POLO.jl: Policy-based optimization algorithms in Julia.
Advances in Engineering Software, 136:102695, 2019.

[15] Iain Dunning, Joey Huchette, and Miles Lubin. JuMP: A modeling language for mathematical optimization.
SIAM Review, 59(2):295–320, 2017.

[16] Benoit Legat, Oscar Dowson, Joaquim Dias Garcia, and Miles Lubin. Mathoptinterface: a data structure for
mathematical optimization problems.arXiv preprint arXiv:2002.03447, 2020.

[17] Jean-Paul Watson, David L. Woodruff, and William E. Hart. PySP: modeling and solving stochastic programs in
python.Mathematical Programming Computation, 4(2):109–149, 2012. Cited on p. 129.

[18] William E. Hart, Carl D. Laird, Jean-Paul Watson, DavidL. Woodruff, Gabriel A. Hackebeil, Bethany L. Nichol-
son, and John D. Siirola.Pyomo — Optimization Modeling in Python. Springer International Publishing, 2017.

[19] Francis Ellison, Gautam Mitra, Chandra Poojari, and Victor Zverovich. Fortsp: A stochastic programming solver.
[http://www.optirisk-systems.com/manuals/FortspManual.pdf,](http://www.optirisk-systems.com/manuals/FortspManual.pdf,) 2009.

[20] Joey Huchette, Miles Lubin, and Cosmin Petra. Parallelalgebraic modeling for stochastic optimization. In 2014
First Workshop for High Performance Technical Computing inDynamic Languages. IEEE, 2014.

[21] Miles Lubin, J. A. Julian Hall, Cosmin G. Petra, and Mihai Anitescu. Parallel distributed-memory simplex for
large-scale stochastic LP problems.Computational Optimization and Applications, 55(3):571–596, 2013.

[22] Wai-Kei Mak, David P. Morton, and R.Kevin Wood. Monte carlo bounding techniques for determining solution
quality in stochastic programs.Operations Research Letters, 24(1):47 – 56, 1999.

[23] Alan J. King and Roger J.-B. Wets. Epi-consistency of convex stochastic programs.Stochastics and Stochastics
Reports, 34(1-2):83–92, 1991.

[24] Alexander Shapiro. Asymptotic analysis of stochasticprograms.Annals of Operations Research, 30(1):169–186,
1991.

[25] Jeff Linderoth, Alexander Shapiro, and Stephen Wright. The empirical behavior of sampling methods for stochas-
tic programming.Annals of Operations Research, 142(1):215–241, 2006.

[26] John R. Birge and François V. Louveaux. A multicut algorithm for two-stage stochastic linear programs.Euro-
pean Journal of Operational Research, 34(3):384–392, 1988.

[27] R. T. Rockafellar. Monotone operators and the proximalpoint algorithm.SIAM Journal on Control and Opti-
mization, 14(5):877–898, 1976.

[28] Andrzej Ruszczynski. A regularized decomposition method for minimizing a s ́ um of polyhedral functions.
Mathematical Programming, 35(3):309–333, 1986.

[29] Jeff Linderoth and Stephen Wright. Decomposition Algorithms for Stochastic Programming on a Computational
Grid.Computational Optimization and Applications, 24(2-3):207–250, 2003.

[30] Csaba I. Fábián and Zoltán Szoke. Solving two-stage stochastic programming problems wi ̋ th level decomposi-
tion.Computational Management Science, 4(4):313–353, 2006.

[31] Martin Biel and Mikael Johansson. Dynamic cut aggregation in L-shaped algorithms. arXiv preprint
arXiv:1910.13752, 2019. Submitted for consideration to the European Journalof Operational Research. Un-
der review.

[32] Shohre Zehtabian and Fabian Bastin. Penalty parameter update strategies in progressive hedging algorithm.
CIRRELT, 2016.

[33] Christian Wolf and Achim Koberstein. Dynamic sequencing and cut consolidation for the parallel hybrid-cut
nested l-shaped method.European Journal of Operational Research, 230(1):143–156, 2013.

[34] Svyatoslav Trukhanov, Lewis Ntaimo, and Andrew Schaefer. Adaptive multicut aggregation for two-stage
stochastic linear programs with recourse.European Journal of Operational Research, 206(2):395–406, 2010.


[35] Martin Biel and Mikael Johansson. Distributed L-shaped algorithms in Julia. In2018 IEEE/ACM Parallel
Applications Workshop, Alternatives To MPI (PAW-ATM). IEEE, 2018.

[36] Suvrajeet Sen, Robert D. Doverspike, and Steve Cosares. Network planning with random demand.Telecommu-
nication Systems, 3(1):11–30, 1994.

[37] Gerd Infanger. Monte carlo (importance) sampling within a benders decomposition algorithm for stochastic
linear programs.Annals of Operations Research, 39(1):69–95, 1992.

[38] Julia L. Higle and Suvrajeet Sen. Stochastic decomposition: An algorithm for two-stage linear programs with
recourse.Mathematics of Operations Research, 16(3):650–669, 1991.


