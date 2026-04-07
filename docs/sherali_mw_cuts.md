DOI 10.1007/s10479-011-0883-

# On generating maximal nondominated Benders cuts

**Hanif D. Sherali** · **Brian J. Lunday**

Published online: 7 April 2011
© Springer Science+Business Media, LLC 2011

**Abstract** In this paper, we explore certain algorithmic strategies for accelerating the con-
vergence of Benders decomposition method via the generation of maximal nondominated
cuts. Based on interpreting the seminal work of Magnanti and Wong (Operations Research,
29(3), 464–484, 1981 ) for generating nondominated cuts within a multiobjective frame-
work, we propose an algorithmic strategy that utilizes a preemptively small perturbation of
the right-hand-side of the Benders subproblem to generate maximal nondominated Benders
cuts, as well as a complimentary strategy that generates an additional cut in each iteration
via an alternative emphasis on decision variable weights. We also examine the computa-
tional effectiveness of solving a secondary subproblem using an objective cut as proposed
by Magnanti and Wong versus identifying the Pareto-optimality region for cut generation by
utilizing complementary slackness conditions. In addition, we exhibit how a standard fea-
sibility cut can be extracted from the solution of subproblems that generate only optimality
cuts through the use of artificial variables. With Magnanti and Wong’s baseline procedure
approximated during implementation via the use of a core point estimation technique (Pa-
padakos in Computers and Operations Research, 36(1), 176–195, 2009 ), these algorithmic
strategies are tested on instances from the literature concerning the fixed charge network
flow program.

**Keywords** Benders decomposition·Maximal cuts·Nondominated cuts·Pareto-optimal
cuts

H.D. Sherali
Grado Department of Industrial and Systems Engineering, Virginia Tech., Blacksburg, VA, 24061, USA
e-mail:hanifs@vt.edu

## B.J. Lunday ()

Department of Mathematical Sciences, United States Military Academy, West Point, NY, 10996, USA
e-mail:brian.lunday@usma.edu

B.J. Lunday
e-mail:brian.lunday@us.army.mil


**1 Introduction**

Since the introduction of Benders decomposition method in 1962 (Benders 1962 ), several
research investigations have sought either to broaden its applicability or to improve its rate
of convergence in attaining optimal solutions. Within the originally envisioned context for
applying Benders decomposition, most research has sought to improve the convergence be-
havior of this approach either by utilizing hybrid strategies to reduce the computational
effort required to solve certain integer programs or by determining more effective cuts to
eliminate regions that contain suboptimal or nonimproving solutions.
Among the notable efforts to reduce the computational effort required to solve the inte-
ger programs, Zakeri et al. ( 1998 ) proposed the generation of inexact cuts for multi-stage
stochastic linear programs based on deriving suboptimal (-optimal) solutions to Benders
subproblems, while maintaining the convergence property of the algorithm. The authors
demonstrated the efficiency of this procedure for solving a hydroelectric scheduling prob-
lem in which the generation of exact cuts turns out to be computationally expensive. Van
Roy ( 1983 ) proposed a cross decomposition approach that utilizes components of both
primal (Benders) and dual (Lagrangian relaxation) decomposition methods, and Wentges
( 1996 ) applied this cross decomposition method to instances of the capacitated facility loca-
tion problem, combined with solving the initial master problem only heuristically and with
generating specialized nondominated cuts, thereby reducing the required computational ef-
fort.
Within the context of determining more effective cuts, most research has sought to gener-
ate an initial set of cuts complemented with additional multiple strong cuts at each iteration,
or by modifying the form of the standard Benders feasibility or optimality cuts. McDaniel
and Devine ( 1977 ) showed that by generating an initial set of cuts based on solving the
linear program (LP) relaxation of the underlying MIP before switching over to solving dis-
crete master programs, one can significantly reduce the computational effort, particularly for
problems having tight LP relaxations. Geoffrion and Graves ( 1974 ) proposed a framework
for solving the grand master program directly using a branch-and-bound approach by gen-
erating Benders cuts in a row-generation strategy whenever a feasible solution is detected
to the relaxed master program that improves upon the incumbent solution to the master
program, in lieu of solving each such relaxed program to optimality. Côté and Laughton
( 1984 ) designed heuristics for the relaxed master program to generate multiple cuts within
the context of applying the procedure of Geoffrion and Graves ( 1974 ), prior to ultimately
switching over to the exact implementation of the latter approach to guarantee convergence
to an optimum. Codato and Fischetti ( 2006 ) examined mixed-integer programming reformu-
lations of statistical analysis and map labeling problems, wherein conditional constraints are
typically formulated using the Big-M technique that yields weak relaxations, and proposed
an improvement of the standard Benders methodology by generating instead combinatorial
Benders cuts that are devoid of Big-M effects and that rely solely on minimal infeasible sub-
systems. Magnanti and Wong ( 1981 , 1990 ) set forth a seminal procedure for generating non-
dominated or Pareto optimal cuts to strengthen the standard Benders optimality cuts, albeit
with the oft-challenging requirement of identifying (and sometimes updating) a core point,
which is required to lie in the relative interior of the convex hull of the problem subregion
that is defined only in terms of the complicating variables. Fischetti et al. ( 2010 ) proposed an
alternative unifying scheme for generating Benders cuts in situations where both optimality
and feasibility cuts exist. Stemming from a characterization of a system that identifies the
existence of violated cuts, they formulated a subproblem using an auxiliary variable that
scales the subproblem as well as the resulting cut, where the generated cut mimics optimal-
ity and feasibility cuts according to whether the variable takes on positive or zero values.


Because of this scaling phenomenon, an appropriate normalization constraint involving the
dual variables plus the auxiliary variable was accommodated within the subproblem. As
represented by their results, a proper tuning of the normalization constraint was found to be
essential to attain computational effectiveness (which could therefore be problem-specific).
Examining network design problems, Rei et al. ( 2009 ) utilized local branching cuts of Fis-
chetti and Lodi ( 2003 ) to actively search for improving solutions to each master program for
generating suitable multiple cuts (including combinatorial cuts that replace the Benders fea-
sibility cuts), and for tightening both upper and lower bounds at each iteration. Saharidis et
al. ( 2010 ) examined two applications of a scheduling problem, for which they demonstrated
the effectiveness of generating covering cut bundles to augment Benders cuts. Whereas a
typical Benders cut is usually of low density and thereby affects only a few complicating
decision variables, a covering cut bundle generates multiple complementary low-density
cuts that jointly involve as many complicating variables as possible, thereby approximating
the improved effect of a high-density cut. The authors showed that this insightful technique
significantly reduces the number of iterations required as well as the computational effort.
Saharidis and Ierapetritou ( 2010 ) proposed the generation of an additional valid Benders
cut based on a maximum feasible subsystem whenever a Benders feasibility cut is gener-
ated. Such cuts were shown to significantly accelerate the convergence process for problems
where more feasibility than optimality cuts are normally generated, and they illustrated this
feature through the solution of certain scheduling production problems involving multipur-
pose and multiproduct batch plants.
In this paper, we propose the generation of single or multiple Benders cuts at each iter-
ation where the cuts utilize a computationally simple perturbation technique and are non-
dominated in a _maximal_ sense as typically used in cutting plane theory for integer pro-
grams (see Nemhauser and Wolsey 1999 ). More specifically, this work makes the following
contributions. First, we interpret the seminal approach set forth by Magnanti and Wong
( 1981 ) within a multiobjective strategy, based on which we propose the generation of spe-
cial types of maximal nondominated cuts by suitably perturbing the right-hand-side of the
Benders (primal) subproblem. (During computational testing, the former procedure is ap-
proximated by estimating a point in the relative interior of the convex hull of the feasible
region.) Second, we explore the computational effectiveness of a variant of Magnanti and
Wong’s algorithm that applies the complementary slackness conditions to simplify the cut
generation process, in contrast with utilizing an objective function-based cut within a sec-
ondary problem as advocated by Magnanti and Wong. Third, we exhibit how a standard
feasibility cut can be readily extracted from the solution of a subproblem that incorporates
a penalized artificial variable in order to more conveniently generate only optimality cuts,
whenever this artificial variable is positive at optimality. In such a case, the resulting feasi-
bility cut that is devoid of the penalty coefficient can be used to replace the usual optimality
cut that would otherwise have been generated. Finally, we provide extensive computational
results using a standard class of problems from the literature related to fixed charge net-
work flow problems in order to test the relative efficacy of the proposed algorithmic strate-
gies.
The remainder of this paper is organized as follows. In Sect. 2 , we recall the basic Ben-
ders decomposition approach, summarize the procedure prescribed by Magnanti and Wong
( 1981 ) for generating nondominated cuts, and propose both a less computationally burden-
some perturbation technique for generating a special class of maximal nondominated cuts,
as well as investigate a variant of the secondary subproblem solved by Magnanti and Wong
( 1981 ). We also derive a technique for extracting feasibility cuts from optimality cuts that


are generated in the presence of an artificial variable used within the subproblem, when-
ever such a variable turns out to be positive at optimality. In Sect. 3 , we present computa-
tional results using a class of problems from the literature to test the relative performance
of these algorithmic strategies and to provide insights into computational expediency, and
we conclude in Sect. 4 with a discussion of results and recommendations for future re-
search.

**2 Alternative schemes for generating nondominated Benders cuts**

Consider the following mixed-integer programming problem:

```
MIP: MaximizecTx+dTy−Mσ
subject to:Ax+Dy−emσ≤b,
x∈X, (y, σ )≥ 0 ,
```
##### (1)

whereX⊆Rnis defined by linear constraints and includes integrality restrictions on the
x-variables,y∈Rp,Aism×n,Dism×p,b∈Rm,c∈Rn,d∈Rp,σ∈Ris an artificial
variable that is suitably penalized in the objective function with a sufficiently large penalty
coefficientM>0, and whereem∈Rmis a vector ofmones. The purpose of incorporating
the artificial variableσin MIP is one of convenience in discussion so as to ensure feasibility
in ( 1 )foranyfixedx∈X(i.e., (relatively) complete recourse), and thereby to preclude the
generation of feasibility cuts in the context of Benders decomposition (Benders 1962 )as
discussed next. In the sequel, we also discuss and test the direct extraction of feasibility cuts
from this construct, which are devoid of the parameterM.
Using Benders decomposition to project MIP onto the space of thex-variables, and let-
tingηdenote the value function in ( 1 ) for the resulting linear program in the variables(y, σ )
whenx∈Xis fixed, we obtain the _master program_ ( **MP** )andthe _subproblem_ ( **SP(** x ̄ **)** )fora
fixedx ̄∈Xas follows:

```
MP: MaximizecTx+η (2a)
```
```
subject to:η≤πTb−πTAx, ∀π∈vert(), (2b)
```
```
x∈X,
```
```
SP( x ̄ ): MaximizedTy−Mσ (3a)
```
```
subject to:Dy−emσ≤b−Ax, ̄ (3b)
```
```
(y, σ )≥ 0 , (3c)
```
whereis the dual feasible region to SP(x ̄) and is given by

```
≡
```
##### {

```
π∈Rm:DTπ≥d, eTmπ≤M, π≥ 0
```
##### }

##### , (4)

and where vert() denotes the set of vertices or extreme points of. We assume that
= ∅, and so MP (or MIP) has an optimal solution. In a standard implementation of Ben-
ders algorithm, we have at any stage a _relaxed master program_ ( **RMP(** K **)** )givenby

```
RMP( K ): MaximizecTx+η (5a)
```

```
subject to:η≤πkTb−πkTAx, ∀k= 1 ,...,K, (5b)
```
```
x∈X, (5c)
```
whereπk∈vert(), fork= 1 ,...,K,definethesetofK currently available Benders
cuts (5b). Let(η, ̄x) ̄ solve RMP(K).ThencTx ̄+ ̄ηyields an upper bound on MP (or MIP).
We next solve SP(x ̄), which has an optimum by construction/assumption, to obtain a pair of
primal-dual optimal solutions(y, ̄σ) ̄ andπ ̄, respectively, of objective valueν ̄=ν[SP(x ̄)],
where we letν[P]denote the optimal objective value for any optimization problem P. Hence,
cTx ̄+ ̄νprovides a lower bound on MP (or MIP). Ifη ̄≤ ̄ν+for some optimality toler-
ance>0, we terminate the process with(x, ̄y, ̄σ) ̄ being an-optimal solution to MIP.
Otherwise, we generate the Benders cut

```
η≤ ̄πTb− ̄πTAx (6)
```
to exclude the current solution (η, ̄x ̄), and we append this cut to RMP(K)toderive
RMP(K+1) withπk+ 1 ≡ ̄π, incrementKby 1, and reiterate.
In this iterative process, Magnanti and Wong ( 1981 ) recommended the generation of
nondominated Benders cuts to accelerate the convergence behavior of the algorithm. For
this purpose, they suggested selecting somexˆ∈relint[conv(X)] (the relative interior of the
convex hull ofX) as a so-called _core point_ , and having solved any subproblem SP(x ̄), they
prescribed the generation of a Benders cut by subsequently solving the following secondary
problem:

```
minimize
```
##### {

```
πT(b−Ax)ˆ :π∈opt
```
##### }

```
, (7a)
```
whereoptrepresents the set of alternative optimal dual solutions to SP(x ̄)asgivenby

```
opt≡
```
##### {

```
π∈:πT(b−Ax) ̄ =ν[SP(x ̄)]
```
##### }

. (7b)

Lettingπ∗solve (7a), Magnanti and Wong proved that the resulting Benders cut

```
η≤π∗Tb−π∗TAx (8)
```
is a nondominated Benders cut that could be used in lieu of ( 6 ) to augment the set of cuts for
defining RMP(K+1), and the iterative process could then be continued as before. Here, as
defined by Magnanti and Wong ( 1981 ), we say that ( 8 )is _nondominated_ (or _Pareto optimal_ )
if there does not exist aπ∈optsuch thatπTb−πTAx≤π∗Tb−π∗TAx,∀x∈X, with at
least onex∈Xfor which a strict inequality holds.
However, there exist certain implementation issues of concern related to Magnanti and
Wong’s ( 1981 ) modified generation of cuts, which include the potential difficulty of finding
a core point (Sandhu and Klabjan 2004 ; Mercier et al. 2005 ; Santoso et al. 2005 ; Papadakos
2008 , 2009 ), and that this strategy does not always yield a net computational advantage in
spite of a reduction in the number of cuts due to a two-fold increase in the number of linear
programs solved to generate each cut (Mercier and Soumis 2007 ). (We note, however, that
Saharidis et al. ( 2010 ) recommend generating a pair of Benders cuts at each iteration, one
from the solution to each of the problems (3a)–(3c)and(7a), in order to improve the com-
putational performance.) To address these challenges, researchers often approximate core
points (Santoso et al. 2005 ; Papadakos 2009 ), use alternative points for a given problem
structure (Papadakos 2008 ), or arbitrarily fix components of the core point vector (Mercier
et al. 2005 ), acknowledging that such a fixing of components does not assure the genera-
tion of nondominated cuts, although convergence to an optimal solution is still guaranteed.


For a particular class of airline scheduling problems, Papadakos ( 2008 ) sought to reduce
the computational effort required by Magnanti and Wong’s approach by generating optimal-
ity cuts based on specially defined core point approximations, combined with algorithmic
modifications to circumvent degeneracy effects, and also generating standard feasibility cuts
whenever an infeasible subproblem is obtained for fixed values of the integer decision vari-
ables.
In this paper, we propose a simpler, more direct and computationally effective procedure
for generating a special class of maximal nondominated Benders cuts as defined below, and
we provide computational results on some specially structured test cases to demonstrate the
efficacy of our approach.
Toward this end, note that we can write any Benders cut generated via SP(x ̄)usinga
selectedπ∈optas

```
η≤πTb+
```
```
∑n
```
```
j= 1
```
##### [

```
−πTaj
```
##### ]

```
xj, (9)
```
whereajdenotes thejth column ofA. By a standard definition used in the integer pro-
gramming literature (see Nemhauser and Wolsey 1999 , for example), for ( 9 ) to be “non-
dominated”, or more distinctly, to be _maximal_ , there must not exist anyπ′∈optfor which
π′Tb≤πTband−π′Taj≤−πTaj,∀j= 1 ,...,n, with at least one of these (n+1) in-
equalities being strict. Note that this definition of a “nondominated cut” is not as strict as
that of Magnanti and Wong’s ( 1981 ) definition stated above, and to be clear, we henceforth
refer to this type of cut as being _maximal nondominated_ ,orsimply _maximal_. Observe that
a Pareto-optimal or nondominated cut generated in the sense of Magnanti and Wong ( 1981 )
is also maximal provided that the core pointxˆis positive, but the converse is not necessarily
true.
Consequently, we can view the process of generating maximal cuts as one of determining
a Pareto optimal solution to the multiple objective linear program (see Steuer ( 1986 ), for
example):

```
minimize
```
##### {

```
πTb,−πTa 1 ,...,−πTan:π∈opt
```
##### }

##### . (10)

This can be achieved by designatingxˆ∈Rnto be any _positive weight vector_ (PWV) and
accordingly minimizing the positively weighted sum of the multiple objective functions in
( 10 )asgivenbyπTb+

```
∑n
j= 1 (−π
```
```
Taj)xˆj, subject toπ∈opt, which yields the exact same
```
problem defined in (7a). Hence, ifxˆis a positive core point solution, then the resulting cut
would be both maximal as well as nondominated in the sense of Magnanti and Wong ( 1981 ).
However, our focus in this paper will be on generating maximal nondominated or maximal
cuts via a strategy that is computationally easy to implement. Noting the unit weight ascribed
to the functionπTbin this combined single objective representation, we might likewise
selectxˆj≡1,∀j= 1 ,...,n, assuming a relatively well-scaled problem (we use this in our
computations in the sequel).
Now, in lieu of solving SP(x ̄) first to formulateoptas in (7b), and subsequently solving
(7a), we can combine these two steps by noting that we are essentially considering in the
present context a preemptive priority multiple objective program, where we wish to mini-
mizeπT(b−Ax) ̄ subject toπ∈with the first priority (i.e., to solve the dual to SP(x ̄)),
and among alternative optimal solutions to this problem, we wish to minimizeπT(b−Ax)ˆ
as in (7a). We denote this preemptive priority multiobjective program as follows:

```
minimize
```
##### {

```
πT(b−Ax) ̄ πT(b−Ax)ˆ :π∈
```
##### }

##### . (11)


Note that the approach of Magnanti and Wong ( 1981 ) to generate nondominated cuts via
core pointsxˆcan be afforded the identical interpretation. As shown by Sherali and Soyster
( 1983 ), there exists aμ>0 small enough such that the following combined weighted-sum
problem equivalently solves ( 11 ):

```
minimize
```
##### {

```
πT(b−Ax) ̄ +μπT(b−Ax)ˆ :π∈
```
##### }

##### . (12)

In general, whereas Sherali and Soyster ( 1983 ) prescribe a value ofμ>0 that would
render ( 12 ) _equivalent_ to ( 11 ), the derivation of such a theoretical weight is not typically a
practically convenient task except in certain special cases as described in Sherali ( 1982 )(see
also Corollary 1 below). However, the following result provides an alternative for generating
near-optimal maximal Benders cuts.

**Theorem 1** _Given any_  0 _with_ 0 < 0 <min{, 1 }, _let_ opt( 0 ) _be the set of_  0 _-optimal
solutions to the dual of SP_ (x) ̄ _as given by_

```
opt( 0 )=
```
##### {

```
π∈:πT(b−Ax) ̄ ≤ν[ SP (x) ̄]+ 0
```
##### }

##### , (13)

_and let_

```
μ≡
```
#####  0

```
Mθ
```
```
, where θ= 0 +max{ 0 ,max
```
#### I

#### I

```
{bˆi}} −min{ 0 ,min
i
```
```
{bˆi}}, (14)
```
_and where_ bˆ≡(b−Ax)ˆ, _and_ xˆ _is a positive weight vector_ ( _PWV_ ). _Denote_ f 1 (π )≡πT(b−
Ax) ̄ _and_ f 2 (π )≡πT(b−Ax)ˆ =πTbˆ _as the two objective functions in_ ( 11 ), _and let_ f 1 ∗≡
ν[ _SP_ (x) ̄] _be the minimum value of_ f 1 (π ) _over_ π∈. _Then letting_ π ̃ _be an optimal solution
obtained for_ ( 12 ) _with_ μ _given by_ ( 14 ), _we have the following holding true_ :

(i) _The optimal solution_ π ̃ _to_ ( 12 ) _satisfies_

```
π ̃∈opt( 0 ), i. e .,f 1 (π) ̃ ≤f 1 ∗+ 0. (15)
```
(ii) _The Benders cut_

```
η≤ ̃πTb− ̃πTAx (16)
```
```
is a valid cut , and moreover , if
```
```
π ̃T(b−Ax) ̄ +(− 0 )≥ ̄η, (17)
```
_then_ (x, ̄ y, ̄σ) ̄ _is an_  _-optimal solution to Problem MP_ ( _or MIP_ ).
(iii) _The Benders cut_ ( 16 ) _is maximal with respect to alternative cuts generated off_

```
̃opt( 0 )≡
```
##### {

```
π∈:πT(b−Ax) ̄ =f 1 (π) ̃
```
##### }

##### . (18)

_Proof_ First of all, denotingf2minandf2maxrespectively as the minimum and maximum
values off 2 (π )subject toπ∈, and noting thateTmπ≤M,π≥0,∀π∈,weget

```
f2max≤Mmax{ 0 ,max
i
```
```
{bˆi}} and f2min≥Mmin{ 0 ,min
i
```
```
{bˆi}}.
```
Therefore, noting ( 14 ), we have that

```
f2max−f2min<Mθ,
```

and hence,

```
 0 =μMθ > μ(f2max−f2min), i.e., 0 +μf2min>μf2max. (19)
```
Now to prove Part 1 , suppose on the contrary thatf 1 (π) > f ̃ 1 ∗+ 0. Then, using ( 19 ), we
get

f 1 (π) ̃ +μf 2 (π)>f ̃ 1 ∗+ 0 +μf 2 (π) ̃ ≥f 1 ∗+ 0 +μf2min>f 1 ∗+μf2max≥f 1 (π) ̄ +μf 2 (π), ̄

whereπ ̄is an optimal dual solution to SP(x ̄). This contradicts the optimality ofπ ̃in ( 12 ),
and so ( 15 ) holds true.
Next, consider Part 1. The validity of the Benders cut ( 16 ) follows from the fact that
π ̃∈. Moreover, from ( 15 ), we get that

```
cTx ̄+ ̃πT(b−Ax) ̄ − 0 ≤cTx ̄+f 1 ∗=cTx ̄+ν[SP(x ̄)]≤ν[MP],
```
and socTx ̄+ ̃πT(b−Ax) ̄ − 0 provides a lower bound for Problem MP (or MIP). Hence, if
[cTx ̄+ ̃πT(b−Ax) ̄ − 0 ]+≥cTx ̄+ ̄η, i.e., ( 17 ) holds true (where note that(− 0 )> 0
by assumption), we have that(x, ̄y, ̄ σ) ̄ is an-optimal solution to Problem MP (or MIP).
Finally, ( 16 ) is maximal with respect to ̃opt( 0 )since if there exists aπ′∈ ̃opt( 0 )for
whichπ′Tb≤ ̃πTband−π′Taj≤− ̃πTaj,∀j= 1 ,...,n,with at least one of these(n+ 1 )
inequalities strict, we would have, noting thatx>ˆ 0,

```
f 1 (π′)+μf 2 (π′)=f 1 (π) ̃ +μf 2 (π′)=f 1 (π) ̃ +μ
```
##### [

```
π′Tb+
```
```
∑n
```
```
j= 1
```
```
(−π′Taj)xˆj
```
##### ]

```
<f 1 (π) ̃ +μ
```
##### [

```
π ̃Tb+
```
```
∑n
```
```
j= 1
```
```
(− ̃πTaj)xˆj
```
##### ]

```
=f 1 (π) ̃ +μf 2 (π), ̃
```
which contradicts the optimality ofπ ̃in ( 12 ). 

**Corollary 1** _Suppose that SP_ (x ̄) _has all-integer data with integral dual extreme point solu-
tions_ ( _e_. _g_ ., _under a total unimodularity condition—see_ Bazaraa et al. ( 2010 )). _Then_ π ̃∈opt
_and_ ( 16 ) _is a maximal Benders cut with respect to_ opt.

_Proof_ Under the condition of Corollary 1 , assuming without loss of generality thatπ ̃is
an extreme point optimal solution to ( 12 ), iff 1 (π) ̃ =f 1 ∗,wethenhavef 1 (π) ̃ ≥f 1 ∗+1,
which would contradict ( 15 )since 0 <1. Hence,f 1 (π) ̃ =f 1 ∗or thatπ ̃∈opt. Moreover,
̃opt( 0 )≡optin this case, and so by Part 1 , the Benders cut ( 16 ) is maximal with respect

toopt. 

```
To implement ( 12 ), note that the dual to this problem is given by:
```
```
SP( ̃x ̄ ): MaximizedTy−Mσ (20a)
```
```
subject toDy−emσ≤(b−Ax) ̄ +μ(b−Ax),ˆ (20b)
```
```
(y, σ )≥ 0. (20c)
```
Denotingbˆ≡(b−Ax)ˆ as in Theorem 1 , we note thatbˆis essentially the right-hand-
side of SP(x ̄)whenx ̄is replaced byxˆ. Hence, when formulating the subproblem in this


modified Benders approach, noting the form of ̃ **SP** (x) ̄ in (20a)–(20c), we simply perturb the
right-hand-side of SP(x ̄)byμbˆ,whereμis given by ( 14 ).

_Remark 1_ **Generating Maximal Cuts for Fischetti et al.’s ( 2010 ) Procedure.** Although we
do not explore its computational performance herein, we note that our perturbation technique
can be applied to the procedure of Fischetti et al. ( 2010 ), wherein, given an MIP formulation

```
min
```
##### {

```
cTx+dTy:Ax≥b, T x+Qy≥r, x∈Zn+,y∈Rt+
```
##### }

##### ,

the authors generate a cutπ∗T(r−Tx)≤ηπ 0 ∗,where(π∗,π 0 ∗)solves the problem:

```
max
```
##### {

```
πT(r−Tx∗)−π 0 η∗:πTQ≤π 0 dT,
```
```
∑m
```
```
i= 1
```
```
wiπi+w 0 π 0 = 1 ,(π,π 0 )≥ 0
```
##### }

##### . (21)

Assuming thatη≥0 for simplicity, we can further promote the generation of maximal cuts
in this procedure by augmenting the objective function in ( 21 ) with the termμ[πT(r−Tx)ˆ−
π 0 ], which results in a revised objective functionπTr( 1 +μ)−π 0 (η∗+μ)−πTT(x∗+μx)ˆ,
wherexˆis some PWV, andμ>0 is sufficiently small.

_Remark 2_ **Deriving Standard Feasibility Cuts Devoid of Big-** M **Effects.** We present here
a technique that enables the direct extraction of extreme direction or feasibility cuts, where
we denote by dir( 0 ) the set of extreme directions of the dual feasible region 0 obtained
by deleting the bounding restrictioneTmπ≤Mfrom. Given an incumbent solutionx ̄to

RMP(K), let(y, ̄ σ) ̄ solve SP(x ̄)(or ̃SP(x ̄), as appropriate) with an associated dual solution
π ̄∈vert(). One of the following two cases result:

**Case (i)** .σ ̄=0: In this case, we generate an optimality cut, as before.

**Case (ii)** .σ> ̄ 0, and hence, we haveemTπ ̄=M: In this case, the original subproblem SP(x ̄)
based on formulating MIP without the artificial variableσis infeasible and its dual is un-
bounded. Note that by the structure of the normalization constrainteTmπ≤M,wehavethat

```
π ̄=πk+λkdk, whereπk∈vert(),dk∈dir(),andλk> 0. (22)
```
We want to extractdkfrom ( 22 ) so that instead of imposing the Benders cutη≤ ̄πTb−
π ̄TAx(orη≤π∗Tb−π∗TAx, as in the variant prescribed by Magnanti and Wong), which
would containO(M)coefficients, we can directly impose the standard feasibility cut:

```
dkT(b−Ax)≥ 0. (23)
```
Let

∗
denote the set of active constraints inat the solutionπ ̄(written as equalities),
_except for_ the presently active constrainteTmπ=M. Then, from Bazaraa et al. ( 2010 ), we
have that the system of equations in

∗
yields a one-dimensional ray defined by the required
extreme directiondk. To ease the detection ofdkvia a standard LP package (e.g., CPLEX),
we simply solve the problem

```
SP′(x ̄):min
```
##### {

```
πT(b−Ax) ̄ :π∈
```
```
∗
,eTmπ=M( 1 + 1 )
```
##### }

##### (24)

for some suitable 1 >0 (e.g., 1 = 0 .1).Notethat( 24 ) has a unique feasible solution, and
so the objective function used could actually be taken as null. This unique solution is given


by

```
π∗∗=πk+λ′kdk, whereλ′k>λk, (25)
```
so that from ( 22 ), we have that

```
(π∗∗− ̄π)=(λ′k−λk)dk. (26)
```
Noting ( 23 )and( 26 ), we can impose the feasibility cut

```
(π∗∗− ̄π)T(b−Ax)≥ 0 , (27)
```
where for numerical convenience, we can scale ( 27 ) by dividing by‖π∗∗− ̄π‖or more
simply by‖π∗∗− ̄π‖∞(i.e., by the maximum component ofπ∗∗− ̄π≥0).

In our computational experiments described in Sect. 3 , we provide insights into algorith-
mic performance by comparing the following alternative procedures:

**Algorithm 1** This is the original Benders procedure that iterates between the master pro-
gram RMP(K) and the subproblem SP(x ̄).

**Algorithm 2a** This is the accelerated Benders procedure that adopts Magnanti and Wong’s
( 1981 ) strategy for generating nondominated Benders cuts, with the approximation and up-
dating of core points conducted in accordance with Papadakos ( 2009 ), i.e., we initialize a
core point approximationxˆwith a vector that is feasible to Problem MIP and update the
approximation at each successive iterationkby settingxˆ← 0. 5 (xˆ+ ̄x),wherex ̄is the in-
cumbent solution obtained by solving RMP(K). We add the caveat that the use of alternative
core points might affect the convergence process via alternative cut selections within a sub-
problem iteration, but we set aside this aspect of algorithmic refinement for future research.
Finally, although Magnanti and Wong do not specify whether their prescribed nondominated
cut augments or replaces a standard Benders cut, Saharidis et al. ( 2010 ) interpret their work
in the former context, as do we herein.

**Algorithm 2b** This procedure is identical to Algorithm2a, except that, as proposed herein,
it uses a fixed PWVxˆin lieu of an updated core point (or approximation thereof) to generate
a maximal cut via (7a) to augment each standard Benders cut generated via (3a)–(3c).

**Algorithm 3** This is also identical to Algorithm2a, except that instead of formulatingopt
as in (7b) for solving (7a) as suggested by Magnanti and Wong ( 1981 ), we utilize the com-
plementary slackness condition as follows. Let(y, ̄σ, ̄ ̄s)be an optimal solution to SP(x ̄),
wheresdenotes the vector of slack variables in (3b). Then, we formulateoptas follows:

```
opt≡{π∈:DTjπ=djify ̄j> 0 , ∀j= 1 ,...,p,
```
```
emTπ=Mifσ> ̄ 0 ,
```
```
πi=0ifs ̄i> 0 , ∀i= 1 ,...,m}, (28)
```
wheredjandDjrespectively denote thejth component ofdand thejth column ofDthat
are associated with the variableyj,∀j= 1 ,...,p.


**Algorithm 4a** This is the proposed modified Benders procedure in which we solve the
perturbed subproblem ̃SP(x ̄)asgivenby(20a)–(20c), whereμisgivenby( 14 ), and where
xˆis a positive weight vector (PWV), thereby generating a maximal cut as per Theorem 1
and Corollary 1.

**Algorithm 4b** This procedure is identical to Algorithm4a, except that it generates two cuts
per iteration. After solving (20a)–(20c) and obtaining an optimal dual solutionπ ̃(i.e., op-
timal solution to ( 12 )), we defineδmin≡min{− ̃πTaj,j= 1 ,...,n},δmax≡max{− ̃πTaj,

j= 1 ,...,n}, and accordingly letxˆˆj=− ̃π

Taj−δmin
δmax−δmin +^2 ,∀j=^1 ,...,n(where^2 >0is
small, e.g., 2 = 0 .1). We then solve (20a)–(20c) withxˆˆin lieu ofxˆ. This algorithmic mod-
ification emphasizes coefficients in ( 10 ) that were not sufficiently minimized, similar to the
motivation of the covering cut bundle approach of Saharidis et al. ( 2010 ).

**Algorithm 5** This is identical to Algorithm4a, except that we utilizeμ≡ 10 −^6 in lieu
of ( 14 )(wealsotryμ∈{ 10 −^5 , 10 −^4 , 10 −^3 }). (Note that this generates valid Benders cuts
because we maintain dual feasibility to the original subproblem, and in effect, implicitly
dictates a particular choice of 0 .)

**Algorithm 6** This is identical to Algorithm 5 , except that we generate a scaled feasibility
cut of the form ( 27 ) whenever a subproblem solutionπ ̄ yields Case (ii)(σ> ̄ 0 )of Re-
mark 2. The motivation here is to test the generation of a standard feasibility cut in lieu
of the resulting (optimality) cut whenever the artificial variableσis positive at optimality
in the subproblem solution. (This technique could likewise be applied to any of the other
foregoing algorithms.)

**3 Computational results**

We coded Algorithms 1 – 6 using C++ (Microsoft Visual 2008 version) in concert with the
IBM-ILOG CPLEX 12.1 solver, and we tested them on a set of specially structured test
instances from the literature, the fixed charge network flow problem ( **FCNF** ), with instances
generated as described by Haouari et al. ( 2007 ). (See Costa ( 2005 ) for a comprehensive
survey of Benders decomposition method as applied to the FCNF.) All runs were made on
a computer having an Intel Model T7100 Processor (dual-core with a 1.80 GHz speed) and
4.00 GB of RAM.
The model formulation for this problem is given as follows:

```
FCNF: Minimize
```
##### ∑

```
(i,j )∈A
```
##### [

```
cijxij+dijyij
```
##### ]

##### (29)

```
subject to
```
##### ∑

```
j:(i,j )∈A
```
```
yij−
```
##### ∑

```
j:(j,i)∈A
```
```
yji=Si,∀i∈N, (30)
```
```
yij≤uijxij, ∀(i, j )∈A, (31)
```
```
xij∈{ 0 , 1 }, ∀(i, j )∈A, (32)
```
```
yij≥ 0 , ∀(i, j )∈A, (33)
```
where the problem is defined over an underlying graph having|N|nodes (i∈N)and|A|
arcs ((i, j )∈A), and wherexijis a binary decision variable that takes on a value of 1 if


the arc(i, j )is included in the network design at a fixed charge ofcij(and 0 otherwise),
∀(i, j )∈A,andyijis a continuous decision variable that represents the nonnegative flow
on arc(i, j )at a unit cost ofdij,∀(i, j )∈A. The objective ( 29 ) seeks to minimize the total
cost, and Constraint ( 30 ) assures a balance of flow at each node, withSirepresenting the net
supply at node∑ i(positive for a source node and negative for a demand node),∀i∈N,where

i∈NSi=0. Constraint (^31 ) requires that an arc must be included in the network design
in order to allow flow across it, where the flow is bounded by the arc-specific capacity,uij,
∀(i, j )∈A. For the special case of uncapacitated arc flows, we can setuij=M,∀(i, j )∈A,
using a suitably large value ofMto prevent imposing unintended bounds on flow (e.g.,
M=

##### ∑

i∈Nmax{^0 ,Si}). Finally, Constraints (^32 )–(^33 ) impose the binary and nonnegativity
logical restrictions onxijandyij, respectively, for all(i, j )∈A.
The test instances for FCNF were generated similar to Haouari et al. ( 2007 ). Assuming
a planar graph, the coordinates of the|N|nodes inR^2 were determined independently via
a continuous uniform distribution on[ 1 , 10 ]. The arcs were established by first generating
a random spanning tree to ensure connectedness, and then randomly adding arcs until the
network contained|A|/2 arcs (for even|A|). Each created arc(i, j )was then augmented by
an oppositely directed arc(j, i)to generate the complete arc setA. Lettingθijdenote the
Euclidean distance between nodesi∈Nandj∈N,weletdij=

##### √

θij,∀(i, j )∈A.The
arc capacities were set asuij≡U,∀(i, j )∈A,whereUwas selected randomly among the
values{ 10 , 20 , 40 }via a discrete uniform distribution, and the fixed costs were determined
ascij=θij

##### √

10 U,∀(i, j )∈A. In a departure from the instance generation scheme used
by Haouari et al. ( 2007 ), in order to affect degenerate subproblems, we generated values
forSi,∀i∈N,via a discrete uniform distribution on[− 2 , 2 ]and such that net supply and
net demand are balanced. Finally, we also represented each equality constraint ( 30 )astwo
inequality constraints to enable direct conversion to the form assumed by MIP.
The results obtained are summarized in Table 1. For Algorithms2a,2b,and 3 ,viewing
the formulations in the form of MIP, we initialized either the core point approximation or
the positive weight vector, as applicable, to bexˆj≡1,∀j= 1 ,...,n. Likewise, for Algo-
rithms4a,4b,and 5 , we utilizedxˆj≡1,∀j= 1 ,...,n. Using a relative optimality toler-
ance of= 0 .001, we report for each instance the following information: the number of
nodes (|N|), the number of arcs (|A|), and the total CPU time (Tkseconds) and the corre-
sponding number of iterations required (Ik) for Algorithmkto attain an-optimal solution,
∀k= 1 ,...,5. We report in the rightmost columns of Table 1 the total CPU time (T 6 sec-
onds), the number of iterations (I 6 ), and the number of extreme direction cuts generated (E)
for Algorithm 6 to attain an-optimal solution, as well as the averages of these performance
measures in the final row. We also terminated an algorithm for a given instance if it failed
to attain an-optimal solution within 900 CPU seconds, and Table 1 recordssuchanevent
with a ‘∗’ against the CPU time.
The ordinal ranking of the algorithms in increasing order of average computational effort,
as reported in Table 1 ,is{ 5 , 4 a, 6 , 4 b, 1 , 3 , 2 b, 2 a}, with the respective % reduction in aver-
age computational effort for the best performing method (Algorithm 5 ) over each of the other
algorithms in this list being{ 28 .5%, 35 .5%, 42 .4%, 47 .7%, 58 .0%, 66 .0%, 66 .1%}.How-
ever, note that Algorithm 6 required relatively lesser computational effort for the instances
with(|N|,|A|)∈{( 75 , 250 ), ( 75 , 300 ), ( 100 , 350 )}. Algorithm 5 generated cuts with an av-
erage density of 11.39% over all instances. Limiting consideration to only the nine instances
that each algorithm could solve within 900 CPU seconds, the ordinal ranking among algo-
rithms that do not utilize feasibility cuts remains the same, but Algorithm 6 exhibited the
greatest average computational effort, primarily due to its poor performance on the instance


## Comparison of Algorithms

over 17 Instances of Problem FCNF

|N

||

A

|

Alg.

2a

Alg.

2b

Alg.

4a

Alg.

4b

aAlgorithm terminated at 900 CPU seconds without attaining an

- Ta b l e
- 1 –
   - Alg.
   - Alg.
   - Alg.
   - Alg.
      - T
      - I
      - T
      - I a
      - T a
      - I b
      - T b
      - I
      - T
      - I a
      - T a
      - I b
      - T b
      - I
      - T
      - I
         - E
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
         -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
            -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
               -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                  -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                     -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                        -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                           -
                              -
                              -
                              -
                              -
                              -
                              -
                              -
                              - a
                              -
                              -
                              - a
                              -
                              -
                              -
                              -
                              -
                              -
                              -
                              -
                              -
                              -
                              -
                              -
                              -
                              -
                              -
                              -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                 -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                    -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       -
                                       - a
                                       -
                                          -
                                          -
                                          -
                                          -
                                          -
                                          -
                                          -
                                          - a
                                          -
                                          -
                                          - a
                                          -
                                          -
                                          - a
                                          -
                                          -
                                          -
                                          -
                                          -
                                          -
                                          -
                                          -
                                          -
                                          -
                                          -
                                          -
                                          -
                                             - Averages
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -
                                             -


with(|N|,|A|)=( 50 , 200 ). Although Algorithms2a,2b,and 3 each performed fewer av-
erage iterations than Algorithm 1 , this did not offset the additional computational effort re-
quired by each of these algorithms to solve two linear programs (and thus generate two cuts)
for each subproblem iteration, reinforcing the observation made by Mercier and Soumis
( 2007 ) that the increase in computational effort per iteration when using the Magnanti and
Wong ( 1981 ) cut generation technique is not necessarily offset by the potential reduction in
required iterations. Note also that, for any instance solved to optimality by each of Algo-
rithms 1 – 5 within 900 CPU seconds, Algorithms2a,2b, 3 ,4a,4b,and 5 each required the
same or fewer number of iterations compared to Algorithm 1. The generation of two cuts per
iteration in Algorithm4breduced the number of iterations required to solve each problem
compared to using Algorithm4a, albeit with an average 39% increase in required compu-
tational effort. Nevertheless, Algorithm4boutperformed the basic Benders decomposition
method, as well as the algorithmic variants of the Magnanti and Wong technique. Further-
more, the use of alternative (greater)μ-values for Algorithm 5 turned out to be less effective
than usingμ= 10 −^6 ,wherean-optimal solution was attained within 900 CPU seconds for
only 75% of the instances forμ∈{ 10 −^5 , 10 −^4 , 10 −^3 }, indicating the shortcoming of select-
ing aμ-value independently of parametric data. Also, the incorporation of feasibility cuts
in Algorithm 6 did not improve the performance in comparison to Algorithm 1 , requiring
more iterations than Algorithm 1 to solve five of the twelve instances, and with the addi-
tional computational burden of solving two linear programs for each subproblem iteration
(although the second LP has a unique feasible solution). Nevertheless, Algorithm 6 did affect
an average 32.6% reduction in the number of iterations required compared to Algorithm 1 ,
over all instances for which both algorithms could attain an-optimal solution within 900
CPU seconds. The characterization of instances for which the inclusion of standard feasi-
bility cuts would benefit an algorithm is elusive based on the instances examined herein, yet
there is some evidence of its potential benefit, which merits further examination in a future
study.

**4 Conclusions and recommendations**

In this paper, we have studied the potential of accelerating the convergence of Benders de-
composition method for mixed-integer programming problems via the generation of non-
dominated (or Pareto-optimal) cuts. We proposed an algorithmic strategy that utilizes a pre-
emptively small perturbation of the right-hand-side of the Benders subproblem, with the
magnitude of the perturbation determined either via an instance parameter-based calcula-
tion or a user-input parameter. Our technique represents a fundamental improvement over
the nondominated cut generation scheme set forth by Magnanti and Wong ( 1981 ), in that it
reduces the number of linear programs to be solved in each subproblem iteration and that it
does not require the identification of a core point within the Benders master problem.
In our experiments, we compared the performance of our proposed algorithmic variants
to the basic Benders decomposition implementation and the nondominated cut generation
method of Magnanti and Wong ( 1981 ) (utilizing core point approximations), as well as a
variant of the latter technique that identifies the Pareto-optimality region for cut genera-
tion utilizing complementary slackness conditions. Considering a set of specially structured
problem instances from the literature, we also examined a modification of the algorithm that
preemptively perturbs of the right-hand-side of the Benders subproblem via a user-input pa-
rameter, via the direct incorporation of feasibility cuts in lieu of penalty parameter-based
optimality cuts. Using a fixed-charge network flow problem set, we found that our proposed


maximal cut generation technique, with one cut generated per iteration and utilizing a user-
input parameter to perturb the subproblem, yielded the best performance in terms of the
average computational effort required to attain an-optimal solution, reducing the effort re-
quired by a standard Benders implementation and by Magnanti and Wong’s nondominated
cut strategy by 47.7% and 66.1%, respectively. Moreover, we found our proposed maximal
cut generation method, utilizing an instance-based parameter to perturb the subproblem, to
yield the second-best performance, requiring 26.8% and 52.6% less average computational
effort than the standard Benders implementation and Magnanti and Wong’s nondominated
cut strategy, again with the latter utilizing a core point approximation.
Although the Benders decomposition procedure was originally designed for solving
large-scale mixed-integer programs (MIPs) or large-scale problems having complicating
(i.e., coupling) variables, Geoffrion ( 1972 ) expanded its application to a broader subset of
general convex programming problems with the use of nonlinear Lagrangian duality theory.
We note that a worthwhile extension to our research is an examination of the efficacy of our
algorithmic enhancement to such a generalized Benders decomposition method, particularly
for specially structured separable cases. Furthermore, although the experiments herein serve
as a sufficient demonstration of the relative efficacy for our proposed methodologies, and the
perturbation techniques have been demonstrated to improve solution efficiency in other ap-
plications (e.g., Sherali et al. 2010 ), a computational study of their efficacy on other classes
of test problems and an exhaustive sensitivity analysis of the algorithmic performance dif-
ferentials to variations in user-specified parameters (i.e.,ε,εi,∀i= 0 , 1 ,2, andM)merit
further examination in a future study.

**Acknowledgements** This work is partially supported by the _National Science Foundation_ under Grant No.
CMMI-0969169. The authors also thank the Guest Editor and four referees for their detailed and constructive
comments that have greatly helped improve the presentation of this paper.

**References**

Bazaraa, M. S., Jarvis, J. J., & Sherali, H. D. (2010). _Linear programming and network flows_ (4th edn.).
Hoboken: Wiley.
Benders, J. F. (1962). Partitioning procedures for solving mixed-variables programming problems. _Nu-
merische Mathematik_ , _4_ (1), 238–252.
Codato, G., & Fischetti, M. (2006). Combinatorial Benders’ cuts for mixed-integer linear programming.
_Operations Research_ , _54_ (4), 756–766.
Costa, A. M. (2005). A survey on Benders decomposition applied to fixed-charge network design problems.
_Computers & Operations Research_ , _32_ (6), 1429–1450.
Côté, G., & Laughton, M. A. (1984). Large-scale mixed integer programming: Benders-type heuristics. _Eu-
ropean Journal of Operational Research_ , _16_ (3), 327–333.
Fischetti, M., & Lodi, A. (2003). Local branching. _Mathematical Programming_ , _98_ (1–3), 23–47.
Fischetti, M., Salvagnin, D., & Zanette, A. (2010). A note of the selection of Benders’ cuts. _Mathematical
Programming_ , _124_ (1–2), 175–182.
Geoffrion, A. M. (1972). Generalized Benders decomposition. _Journal of Optimization Theory and Applica-
tions_ , _10_ (4), 237–260.
Geoffrion, A. M., & Graves, G. W. (1974). Multicommodity distribution system design by Benders decom-
position. _Management Science_ , _20_ (5), 822–844.
Haouari, M., Mrad, M., & Sherali, H. D. (2007). Optimum synthesis of discrete capacitated networks with
multi-terminal commodity flow requirements. _Optimization Letters_ , _1_ (4), 341–354.
Magnanti, T. L., & Wong, R. T. (1981). Accelerating Benders decomposition: algorithmic enhancement and
model selection criteria. _Operations Research_ , _29_ (3), 464–484.
Magnanti, T. L., & Wong, R. T. (1990). Decomposition methods for facility location problems. In P. B.
Mirchandani & R. L. Francis (Eds.), _Discrete location theory_ (pp. 209–262). Hoboken: Wiley.
McDaniel, D., & Devine, M. (1977). A modified Benders’ partitioning algorithm for mixed integer program-
ming. _Management Science_ , _24_ (3), 312–319.


Mercier, A., & Soumis, F. (2007). An integrated aircraft routing, crew scheduling, and flight retiming model.
_Computers & Operations Research_ , _34_ (8), 2251–2265.
Mercier, A., Cordeau, J., & Soumis, F. (2005). A computational study of benders decomposition for the
integrated aircraft routing and crew scheduling problem. _Computers & Operations Research_ , _32_ (6),
1451–1476.
Nemhauser, G. L., & Wolsey, L. A. (1999). _Integer and combinatorial optimization_. Hoboken: Wiley.
Papadakos, N. (2008). Practical enhancements to the Magnanti-Wong method. _Operations Research Letters_ ,
_36_ (4), 444–449.
Papadakos, N. (2009). Integrated airline scheduling. _Computers & Operations Research_ , _36_ (1), 176–195.
Rei, W., Cordeau, J.-F., Gendreau, M., & Soriano, P. (2009). Accelerating Benders decomposition by local
branching. _INFORMS Journal on Computing_ , _21_ (2), 333–345.
Saharidis, G. K. D., & Ierapetritou, M. G. (2010). Improving Benders decomposition using maximum feasible
subsystem (MFS) cut generation strategy. _Computers & Chemical Engineering_ , _34_ (8), 1237–1245.
Saharidis, G. K. D., Minoux, M., & Ierapetritou, M. G. (2010). Accelerating Benders method using covering
cut bundle generation. _International Transactions in Operational Research_ , _17_ (2), 221–237.
Sandhu, R., & Klabjan, D. (2004). Integrated airline planning. In _Proceedings of the 44th AGIFORS annual
symposium_ , Singapore.
Santoso, T., Ahmed, S., Goetschalckx, M., & Shapiro, A. (2005). A stochastic programming approach for
supply chain network design under uncertainty. _European Journal of Operational Research_ , _167_ (1),
96–115.
Sherali, H. D. (1982). Equivalent weights for lexicographic multi-objective programs: characterizations and
computations. _European Journal of Operational Research_ , _11_ (4), 367–379.
Sherali, H. D., Bae, K., & Haouari, M. (2010). A Benders decomposition approach for an integrated air-
line schedule design and fleet assignment problem with flight retiming, schedule balance, and demand
recapture. _Annals of Operations Research_ (to appear).
Sherali, H. D., & Soyster, A. L. (1983). Preemptive and nonpreemptive multi-objective programming: rela-
tionships and counterexamples. _Journal of Optimization Theory and Applications_ , _39_ (2), 173–186.
Steuer, R. E. (1986). _Multiple criteria optimization: theory, computation, and application_. Hoboken: Wiley.
Van Roy, T. J. (1983). Cross decomposition for mixed integer programming. _Mathematical Programming_ ,
_25_ (1), 46–63.
Wentges, P. (1996). Accelerating Benders’ decomposition for the capacitated facility location problem. _Math-
ematical Methods of Operations Research_ , _44_ (2), 267–290.
Zakeri, G., Philpott, A. B., & Ryan, D. M. (1998). Inexact cuts in Benders’ decomposition. _SIAM Journal on
Optimization_ , _10_ (3), 643–657.


