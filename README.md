# Prelude

A couple of words about where I'm coming from: I have a background in ML, but also functional programming, programming language theory, DSLs, and all that stuff.  My master's thesis is about a system that tracks Julia IR, using IRTools, of Turing models, to extract a "semantic representation" of what's going on in a model in probabilistic terms.  During that, I started contributing a bit to Turing and DynamicPPL, with a focus on the the model macro and the internal model representation.  All in all, a lot of metaprogramming.

Currently, Turing models are very primitive in this respect: a data structure called `VarInfo` contains a map from variable names to values, the accumulated log-likelihood, and some other metadata.  During my project, I noticed that retrospectively fitting structure onto this is not ideal, and for proper analysis, it would be nice to begin with a better representation from the start.  The two main difficulties were matching of `VarName`s (e.g., when I need `x[1:3][2]`, but the model contains `x[1:10]`), and getting rid of array mutations that shadow actual data dependencies (e.g., one has an array `x`, samples `x[1]`, writes it to `x` with `setindex!`, and then uses `getindex(x, i)` somewhere downstream).  I thought about writing a more versatile kind of "dictionary with variable name keys", but that wouldn't satisfactorily solve all of the issues.

From these difficulties that occured during my quest of extracting Gibbs conditionals, together with the knowledge about DynamicPPL's internals, I developed the following understanding of what an ideal representation of probabilistic models for the purpuse of analysis would be for me.


# An intermediate representation for probabilistic programs with multiple interpretations

What I propose with the "probabilistic IR" kind of turns around the way  things are constructed right now in all the Julia approaches I've seen. Instead of starting from your sampling function/generative function/model, which is evaluated to get out graphs from it, you start from a representation of the model that already is "graphical", and derive evaluators from it. And if that representation looks like Julia IR, it doesn't matter whether the model is dynamic -- you always work on a fixed, full program.

The advantage of this kind of approach, besides solving "compiler domain" problems like the ones I mentioned above, is that it provides a different kind of common abstraction for PPLs.  Recently, people have started writing "bridge code" to allow PPL interaction: there is invented a common interface that multiple PPL systems can be fit under, and then models in each can be used from within the other at evaluation.  This approach is due to the lack of division of a system into an evalator and a model specification part (DynamicPPL is supposed to be a factored out model system, but currently way too specialized to Turing): they always go totheger.  I believe that starting from a common model specification language is in many cases more feasible and general than defining a common interface for evaluators: the latter tends to assume much more about the internals, while model syntax is essentially fixed: the notation of random variables used in model specification by hand, extended through general Julia syntax.

I think this is feasible also from the end-user perspective, because there is a difference between AD and PPL-DSLs: you can't expect every writer of a mathematical function to anticipate it being used in AD code by marking it as `@differentiable` or whatever. But you _can_ expect that from the writer of a probabilistic model, because non-standard evaluation in some form is inherently part of the whole probabilistic programming approach.

## Summary of other Julia PPL approaches

- Turing/DynamicPPL: a model is a function that, applied to arguments, produces an object containing a closure, the "model function".  This closure contains code very similar to the original model code, with the tildes replaced by "tilde functions" that get some metadata (the lhs and rhs of the expression, for example), and chain through the `VarInfo` object.  The closure can then be run with a sampler in different "contexts", which determine how the tilde functions work: to run the model as a generative function, sampling from the prior, evalating the log-joint, or more exotic things necessary for VI etc.
- Gen: what a "model" is is left open in principle, and only defined by the "generated function interface" of eight (?) methods `generate`, `update`, etc.  These allow interaction between submodels: a sampling algorithm can use the interface on any model, and gets back only concrete values or traces, which are in turn also only defined by an abstract interface (so different models might have different optimized versions of traces).  But there are also two built-in "modelling languages", the dynamic one and the static one.  In the dynamic language, a model is a function that just runs the model code as written, logging the traced variables on a tree-dictionary like structure.  The static language is a rather restricted DSL based on normal Julia code, and stored internally as a graph, allowing more efficient (re)evaluation, since the key space of the trace is known in advance.
- Omega: The idea is, IIUC, modelled on the mathematical definition of random variables and probability spaces.  RVs can be created from a common source of randomness, a kind of underlying extensible probability space.  The association between RVs and the underlying events is remembered to model independence structures.  Then, the RVs can be combined in pretty much arbitratry ways.  The interface is a bit more general than in sampling-based languages like Gen and Turing, since one of the goals is to support arbitrary constraints. Additionally, interventions and other functions for causal inference are supported.
- ForneyLab: less a PPL with an interface for modelling -- more like an implementation of factor graphs with a higher-level interface for constructing them.  Key point is the automatic derivation of message passing based inference algorithms from the graphs.
- Soss: a PPL with general model syntax.  Execution is based on complete symbolic analysis of the model, and producing generated functions.
- Stheno: a combiator interface for  Gaussian Processes, supporting a very general model syntax and complex queries.
- Jaynes: An interface to Gen, based on IR analysis: instead of using one of the Gen modelling languages, a regular function with overloaded `rand` methods can be used.  This function is evaluated through IRTools dynamos that produce efficient implementations of the GFI methods.
- Poirot: Another IR-based model specification language that works with normal functions and IR transformations, but in a style similar to Church (explicit `observe` calls etc.).  Puerly experimental for now.

Of these, Gen seems like the best candidate for an abstraction of a "model evaluation interface" for sampling-based approaches.  Omega, Soss, and ForneyLab could make good general evaluators for abstracted models in the IR as well, and Stheno for the fragment of models describing Gaussian Processes (and we could imagine something similar for nonparametric models).  The approaches of Jaynes and Poirot could make a good start for the "front end" -- the way by which a user/modeller writes their model down, and through which it is turned into the intermediate representation.  Jaynes especially also has begun the next step: the transformation from the abstract model to generated evaluation code.


# "Single static sampling form"

As far as I see it, the lowest common denominator (or is it least upper bound?) of all PPL modelling parts consists of 

1. General Julia code: this can most conveniently be represented like normal Julia IR with SSA statements, branches, and blocks
2. "Sampling statements": tildes, or assumptions/observations in Turing parlance, which relate names or values to distributions in an immutable way
3. Variable names, which may be "complex", like containing indexing, fields, link functions, etc., that can be identified/analyzed in a structured way

So my idea was to just combine all that into an IR-like syntax.  This amounts to writing down a directed graphical model with deterministic and stochastic nodes, named random variables, generalized to programs (i.e., you can have dynamic structure due to branching and recursion).

A model in this kind of format then defines an abstract and uninterpreted parametrized density joint density function over its trace (as given through the unified name set) and return value, factorized into primitive statements and blocks.


## Choices of syntax

One of the nice properties of probabilistic programs is that logically, random variables already behave like SSA form -- you can only assign them once. But in something like Turing, assigning to "complex" ones, i.e., arrays, needs to be implemented by mutation of actual data structures, and that annoyed me a lot recently. 

What I would like to have is a SSA language with a semantics in mind that gets rid of this need, by allowing you to use "complex names", that are unified as needed. In essence, this works like a structural equation model in SSA form, extended with control flow, e.g. the following fragment

```julia
n = 1
while n <= N
    {x[n]} ~ Normal({mu[z[n]]})
    n += 1
end
{y} ~ MvNormal({x}, {sigma})
```

being translated to an equivalent of

```julia
1:
  goto 2 (1)
2 (%1):
  %2 = {z[%1]}
  %3 = {mu[%2]}
  {x[%1]} ~ Normal(%3)
  %4 = %1 < N
  br 4 unless %4
  br 3
3:
  %5 = %1 + 1
 br 2 (%5)
4:
  {y} ~ MvNormal({x}, {sigma})
```

(The `{x}` notation I just made up for illustrative purposes.  It has the advantage that Julia can parse it.  I probably I have seen this trick somewhere else before.  Question: do we need this notation? Or could the distinction be based purely on a type system?  At least at the first sight, marking the variables makes the model more clear to me.)

The `x` in block 4 should be uniquely recoverable from the fact that before, we have already specified all the `x[n]`s. (I'm thinking of some kind of unification mechainsm for that).  When you actually want to run this program, you can then *choose* whether to implement it with an array backing, or something like a dictionary/trie-based trace structure.

How this is then interpreted is left abstract. At least (almost) trivially, you can just sample from the prior by putting the variable names into a dictionary, and replacing the tildes with assignments to `rand` calls.  (The complication is that one may have to infer shapes and "unify" variables, like the `x[i]` and `x` above. But there must be at least one general solution to that.)

A specific system can then define its own interpretation or even generate code out of this form. Everything is open to static or dynamic analysis. And an implemention might just accept a fragment of the general variant, e.g., free of stochastic control flow, or with restrictions the indexing of variable names.

Note that there is within the representation no distinction of assumed and observed variables, as in Turing.  This gives more freedom in evaluation, without having to resort to things like the `Prior` context in DynamicPPL. 

We can even have "crazy" models few if any current PPL syntaxes can currently accept:

```julia
{G} ~ DirichletProcess(5.0)
{mu[1:N]} .~ iid({G}, N) # aka `filldist`
{y[1:N]} .~ Normal.({mu})
```

or

```julia
# This could actually be evaluated in Stheno, I guess:
{f} ~ GP(kernel)
{y[1:N]} = {f}.(x[1:N])
```

At least write them.  If anyone cares to invent and implement a transformation that will transform these into a practical form in a general way.

## Normal and tilde assignment

Note that we get a feature for free which are not part of Turing (or most other modelling languages) yet:

```julia
{x} ~ Normal()
{y} = {x} + 1
```

designates `y` as a deterministic, named random variable.  Also:

```julia
{x[1]} = 1
{x[2:N]} ~ Markov({x[1]}, x -> Normal(x), N)
```

Conversely, of course, we can have "constant observations":

```julia
1 ~ Poisson(3.0)
```

or even

```julia
x = something()
f(x) ~ Dirichlet(10.0)
```

(is that ever useful?)

We can also have anonymous random variables; in the IR:

```julia
1:
  {%1} ~ Normal() # instead of {x}
  {y} ~ Normal({%1})
```

Perhaps they can be useful in intermediate transformations, after we have specialized on a query: when `x` is integrated out, we don't need to name it anymore, and don't want to see its name in the resulting chain object.  

Or they are just not important; e.g., sampling from a geometric distribution:

```julia
i = 1
while ({} ∼ Bernoulli(0.5))
  i = i + 1
end
{n} = i
```

We can also directly assign values of random variables to Julia variables:

```julia
x = zeros(N)
x[1] = {x[1]} ~ Normal()
{x[2]} ~ Normal(x[1])
```

As you see, this is getting dangerous, since now `{x[1]}` and `{x[2]}` aren't related anymore in the resulting IR, because in lowered code, `x` was just mutably written to using `setindex!`.

## Lenses, unifiable link functions, and first class names

A crucial part is the semantics of named variables.  The problem can, I think, be generalized to arbitrary "link functions" that can be described through structural and functional lenses:

- Arrays/`getindex`: `x[1:10][1] ~ D`
- Structs/`getproperty`: `foo.bar[1].baz ~ D`
- Bijector link functions: `log(x[1]) ~ D`

What I still need to figure out is how to properly "unify names".  Lenses + copatterns + unification?  See: https://cs.stackexchange.com/questions/129702/semantics-of-write-once-variables-for-complex-data-structures

Another part, altough minor, is to make sure the "write once" semantics: random variables aren't variables that can be "written to" -- they are definitions.  But that should be easier: SSA form is write-once already, in a sense; and a solution based on (co-)pattern matching is also intrinsically not based on memory locations, but unification.

Maybe it will be necessary (or just useful?) to classify names into "static" and "dynamic" ones.  A static name is one that contains only constants.  This can change after partial evaluation: for example, a loop depending on the size of the observed values might become unrolled into a sequence of known size of tildes.  On the other hand, if the loop remains, it must depend on a random variable dependent on the input, and it's body dynamic.  Similar for probabilistic `if`s.  This classification gives a clear definition of dynamicity of (parts of) models: a model is static, if all variables within it are static.

Maybe there is a way to subsume or interpret models as distributions over traces.  For a static model, the key space of the trace is deterministic; for dynamic models, it is not, altough it could consist of an indexed-family-like thing, or "stochastic [dependent sum](https://en.wikipedia.org/wiki/Dependent_type#%7F'%22%60UNIQ--postMath-00000016-QINU%60%22'%7F_type)": `\sum_{n ~ Poisson()} [{x[i]} for i = 1:n]`. Todo: look at trace types.

Another possibility is to use static analysis to extract local dynamic parts into combinators, like Soss and Gen's, or into local Markov kernels (which probably is just collapsing?).

Perhaps it is possible to construct an indexed free monad over statements, whose bind operator performs the name unification at the trace type level.


# Interpretation in Julia IR

## Model compilation as partial specialization

The next step is to perform the interpretation as partial evaluation given the observed data and the sampling algorithm, Mjolnir-style. Then you can "lower" the abstract probabilistic model to concrete Julia IR "optimized" for that specific inference.  If we go back to the above example, supposedly, `N`, the size of `x`, depends on the size of the input. So if we partially evaluate the IR on constant input,  the loop vanishes and a series of `x[1] = ...; ...; x[N] = ...` remains. In that case, “shape” inference should be easy.

Ideally, the "compilation step" is a function from a model, an evaluator, and a query to something executable.  Alternatively (although probably less efficient or more complicated), from model + evalator to (query -> executable).

## Correspondence with Julia surface syntax

The expansion of 

```julia
@model function changepoint(y)
    N = length(y)
    α = 1 / mean(y)
    {λ₁} ~ Exponential(α)
    {λ₂} ~ Exponential(α)
    {τ} ~ DiscreteUniform(1, N)
    
    for n in 1:N
        {y[n]} ~ Poisson({τ} > n ? {λ₁} : {λ₂})
    end
end
```

would probably consiste of just of a constant `changepoint` of an anonymous type, plus a method `getmodel(typeof(changepoint), Tuple{<:Any})` returning the constant

```julia
1: (%1, %2)
  %3 = length(%2)
  %4 = mean(%2)
  %5 = 1 / %4
  {λ₁} ~ Exponential(%5)
  {λ₂} ~ Exponential(%5)
  {τ} ~ DiscreteUniform(1, %3)
  %6 = 1:%3
  %7 = Base.iterate(%6)
  %8 = %7 === nothing
  %9 = Base.not_int(%8)
  br 6 unless %9
  br 2 (%7)
2: (%10)
  %11 = Core.getfield(%10, 1)
  %12 = Core.getfield(%10, 2)
  %13 = {τ} > %11
  br 4 unless %13
  br 3
3:
  br 5 ({λ₁})
4:
  br 5 ({λ₂})
5: (%14)
  %15 = Poisson(%14)
  {y[%11]} ~ %15
  %16 = Base.iterate(%6, %12)
  %17 = %16 === nothing
  %18 = Base.not_int(%17)
  br 6 unless %18
  br 2 (%16)
6:
  return nothing
```

This can then be used in generated functions operating on `changepoint` (e.g., dynamos), that can inspect the model, transform it, and return actual Julia code.  This would suffice for use cases such as sampling from the prior or the joint, or for conversion to a log-probability evaluator.  If things like specialization on the input data are required, then the translation to Julia would have happen at runtime, of course.


## Static analysis

First, there should obviously be an interfact to allow seamless queries of all kinds of dependencies between variables, working with and categorizing static and dynamic dependencies (i.e., blocks depending on variables through conditional branches).  This will allow probabilisic analysis of models, e.g., calculation of Markov blankets and the like, as well as help transformations, such as to factor graphs. 

Then, there is also a load of already known concepts from compiler theory and static analysis that may not be known to PPL folks, but can nevertheless be sucessfully utilized already at the level of the "evalation transformation": for example, a certain partial evalation might be followed by optimization passes, or require some form of dependency information for which algorithms are well known.

There are several possible modes of operation on the IR: 

- internal transformation, such as:
  - specialization on a constant parameter or observation, resulting in a new model in IR form
  - exploitation of probabilistic knowledge, like collapsing or conjugacy exploitation, or "disaggretating" a non-parametric model into something sampleable (e.g., re-representing a Dirichlet process model with a CRP-based one)
  - intervention, or other changes of the probabilistic structure
- static analysis or abstract interpretation only for analysis purposes: e.g., extraction of Gibbs conditionals
- evalation -- the transformation of the IR into executable code (Julia IR for a specific PPL system, for example) or runtime model structure

Example usages:

- Give me a new model that is the Gibbs conditional for varible `X`
- Turn the Gibbs conditional model into a plain Julia generative function (just `rand` calls), to be used as proposal distribution in a Gibbs sampler
- Give me a ForneyLab factor graph for this model, with `Y` fixed to `data`
- Give me an Omega function for this model
- Specialize the model with `Y` fixed to `data`, check that it is static, and derive a Gen GFI implementation for it.
- Transform the model into a form that uses only Soss combinators, and pass it to Soss
- Give me the model after the intervention `do(Z = z)`
- Turn this infinite mixture model into a Chinese Restaurant Process with centers `z`, and give me a Turing model for it

# Interpretation in a probability monad

I think you can interpret the represantion in a monadic way as well, a la Giry monads and the like. Maybe with some fancy nonstandard lambda calculi for the names, and CPS for blocks and jumps.

Oh, and except form the branching, the blokcs can be directly interpreted in a monad:

```
...
let t2 = z[t1] in
let t3 = mu[t2] in
Normal(t3) >>= (\<x[t1]> ->
let t4 = t1 < tN in
...
```

## Bookkeeping names with locally nameless lambdas

Kappa calculus? Explicit substitutions? De Bruijn levels, locally nameless terms representation?


## Blocks as coroutines/CPS functions

I haven't yet found out how to properly deal with blocks and block arguments.  Trivially, each block can be transformed into a function from all arguments and all free SSA variables to it's body, and then branches are tail calls.  However, this doesn't seem the nicest idea.  Maybe a solution exists in [Kelsey, 1995](https://doi.org/10.1145/202530.202532).


# Possible evaluators/semantics

A common query syntax would also be appropriate.  I think this is easier to find than the modelling syntax or the abstract evalator interface.

```julia
@model function foo() ... end

# MCMC sampling
posterior = @P(theta, lambda) | @P(x = data)
m = compile(model, posterior, MH())
chain = sample(m)

# data generation
generative = @P(x)
m2 = compile(model, generative, MH())

# VI
posterior = @P(theta, lambda) | @P(x = data)
m3 = compile(model, posterior, VI())
```

## Causal queries

TODO: see what Omega does.

I don't know if this is actually relevant, I might just have read too much about causality in the last weeks.  Anyway: by separating variable names and tildes, we get for free the possibility of transforming models through interventions: simply change the tilde to equality and replace the rhs by a fixed value.  Example:

```julia
@model colliding()
  x ~ Normal()
  y ~ Normal(mu_y + x)
  z ~ Normal(mu_z + x)
  w ~ Normal(mu_w + alpha * y + beta * z)
end
```

While the query `@P(w) | @P(y = obs_y, z = obs_z)` calculates a _conditional_ 

```
p(w | y, z) = ∫ p(x) p(y | x) p(z | x) p(w | y, z) dx
```

we can also trivially support Pearl's do-notation, `@P(w) | @P(y = obs_y, do(z = obs_z))`, which corresponds to evaluation 
in the interventional model

```julia
@model colliding()
  {x} ~ Normal()
  {y} ~ Normal(mu_y + x)
  {z} = obs_z
  {w} ~ Normal(mu_w + alpha * y + beta * z)
end
```

with factorization

```
p(w | y, ẑ) = ∫ p(x) p(y | x) p(w | y, ẑ) dx
```

This corresponds to my understanding that the `do`-notation would be much better written as a model transformation, always, and not through within probability queries.  Fortunately, with a PPL formulation like this, we always have am explicit model at hand.
