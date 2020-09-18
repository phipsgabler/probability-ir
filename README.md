# Prelude

A couple of words about where I'm coming from: I have a background in ML, but also functional programming, programming language theory, DSLs, and all that stuff.  My master's thesis is about a system that tracks Julia IR, using IRTools, of Turing models, to extract a "semantic representation" of what's going on in a model in probabilistic terms.  During that, I started contributing a bit to Turing and DynamicPPL, with a focus on the the model macro and the internal model representation.  All in all, a lot of metaprogramming.

Currently, Turing models are very primitive in this respect: a data structure called `VarInfo` contains a map from variable names to values, the accumulated log-likelihood, and some other metadata.  During my project, I noticed that retrospectively fitting structure onto this is not ideal, and for proper analysis, it would be nice to begin with a better representation from the start.  The two main difficulties were matching of variable names (e.g., when I need `x[1:3][2]`, but the model contains `x[1:10]`), and getting rid of array mutations that shadow actual data dependencies (e.g., one has an array `x`, samples `x[1]`, writes it to `x` with `setindex!`, and then uses `getindex(x, i)` somewhere downstream).  I thought about writing a more versatile kind of "dictionary with variable name keys", but that wouldn't satisfactorily solve all of the issues.

From these difficulties that occured during my quest of extracting Gibbs conditionals, together with the knowledge about DynamicPPL's internals, I developed the following understanding of what an ideal representation of probabilistic models for the purpuse of analysis would be for me.


# An intermediate representation for probabilistic programs with multiple interpretations

What I propose with the "probabilistic IR" kind of turns around the way  things are constructed right now in all the Julia approaches I've seen. Instead of starting from your sampling function/generative function/model, which is evaluated to get out graphs from it, you start from a representation of the model that already is "graphical", and derive evaluators from it. And if that representation looks like Julia IR, it doesn't matter whether the model is dynamic -- you always work on a fixed, full program.

I think this is feasible also from the end-user perspective, because there is a difference between AD and PPL-DSLs: you can't expect every writer of a mathematical function to anticipate it being used in AD code by marking it as `@differentiable` or whatever. But you _can_ expect that from the writer of a probabilistic model, because non-standard evaluation in some form is inherently part of the whole probabilistic programming approach.



# "Single static sampling form"

As far as I see it, the lowest common denominator of all PPLs consists of 

1. General Julia code: this can most conveniently be represented like normal Julia IR with SSA statements, branches, and blocks
2. "Sampling statements": tildes, or assumptions/observations in Turing parlance, which relate names or values to distributions in an immutable way -- "write once" assignment with special semantic meaning
3. Variable names, which may be "complex", like containing indexing, fields, link functions, etc., that can be identified in a structured way

So my idea was to just combine all that into an IR-like syntax.

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

Note that there is within the representation no distinction of assumed and observed variables, as in Turing.  This gives more freedom in evaluation, without having to resort to things like the "Prior context" in DynamicPPL. 


## Normal and tilde assignment

Note that we get a feature for free which are not part of Turing, yet:

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

It remains the question: would it ever make sense to allow anonymous random variables?

```julia
{%1} ~ Normal() # instead of {x}
{y} ~ Normal({%1})
```

Perhaps they can be useful in intermediate transformations, after we have specialized on a query: when `x` is integrated out, we don't need to name it anymore, and don't want to see its name in the resulting chain object.

## Lenses, unifiable link functions, and first class names

A crucial part is the semantics of named variables.  The problem can, I think, be generalized to arbitrary "link functions" that can be described through structural and functional lenses:

- Arrays/`getindex`: `x[1:10][1] ~ D`
- Structs/`getproperty`: `foo.bar[1].baz ~ D`
- Bijector link functions: `log(x[1]) ~ D`

What I still need to figure out is how to properly "unify names".  Lenses + copatterns + unification?  See: https://cs.stackexchange.com/questions/129702/semantics-of-write-once-variables-for-complex-data-structures

Another part, altough minor, is to make sure the "write once" semantics: random variables aren't variables that can be "written to" -- they are definitions.  But that should be easier: SSA form is write-once already, in a sense; and a solution based on (co-)pattern matching is also intrinsically not based on memory locations, but unification.


# Interpretation in Julia IR

## Model compilation as partial specialization

The next step is to perform the interpretation as partial evaluation given the observed data and the sampling algorithm, Mjolnir-style. Then you can "lower" the abstract probabilistic model to concrete Julia IR "optimized" for that specific inference.  If we go back to the above example, supposedly, `N`, the size of `x`, depends on the size of the input. So if we partially evaluate the IR on constant input,  the loop vanishes and a series of `x[1] = ...; ...; x[N] = ...` remains. In that case, “shape” inference should be easy.

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
