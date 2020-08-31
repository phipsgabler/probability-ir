# An intermediate form for probabilistic programs with three possible representations

What I propose with the "probabilistic IR" kind of turns around the way  things are constructed right now in all the Julia approaches I've seen. Instead of starting from your sampling function/generative function/model, which is evaluated to get out graphs from it, you start from a representation of the model that already is "graphical", and derive evaluators from it. And if that representation looks like Julia IR, it doesn't matter whether the model is dynamic -- you always work on a fixed, full program.

I think this is feasible also from the end-user perspective, because there is a difference between AD and PPL-DSLs: you can't expect every writer of a mathematical function to anticipate it being used in AD code by marking it as @differentiable or whatever. But you can expect that from the writer of a probabilistic model, because non-standard evaluation in some form is inherently part of the whole probabilistic programming approach.

# Single static sampling form

One of the nice properties of probabilistic programs is that logically, random variables already behave like SSA form -- you can only assign them once. But in something like Turing, assigning to "complex" ones, i.e., arrays, needs to be implemented by mutation of actual data structures, and that annoyed me a lot recently. 

What I would like to have is a SSA language with a semantics in mind that gets rid of this need, by allowing you to use "complex names", that are unified as needed. E.g.

```
x = zeros(N)
n = 1
while n <= N
    x[n] ~ Normal(mu[z[n]])
    n += 1
end
y ~ MvNormal(x, sigma)
```

being translated to

```
1:
  goto 2 (1)
2 (%1):
  %2 = <z[%1]>
  %3 = <mu[%2]>
  <x[%1]> ~ Normal(%3)
  %4 = %1 < N
  br 4 unless %4
  br 3
3:
  %5 = %1 + 1
 br 2 (%5)
4:
  <y> ~ MvNormal(<x>, sigma)
```

The `x` in block 4 should be uniquely recoverable from the fact that before, we have already specified all the `x[n]`s. (I'm thinking of some kind of unification mechainsm for that).  When you actually want to run this program, you can then *choose* whether to implement it with an array backing, or something like a dictionary/trie-based trace structure.


## Normal and linear assignment

## Lenses, unifiable link functions, and first class names

- Arrays/`getindex`
- Structs/`getproperty`
- Bijector link functions


# Interpretation in Julia IR

## Model compilation as partial specialization

The next step is to perform the interpretation as partial evaluation given the observed data and the sampling algorithm, Mjolnir-style. Then you can "lower" the abstract probabilistic model to concrete Julia IR "optimized" for that specific inference.

## Correspondence with Julia surface syntax


# Interpretation in a probability monad

I think you can interpret the represantion in a monadic way as well, a la Giry monads and the like. Maybe with some fancy nonstandard lambda calculi for the names, and CPS for blocks and jumps.

## Bookkeeping names with locally nameless lambdas

Kappa calculus? Explicit substitutions? De Bruijn levels, locally nameless terms representation?

## Blocks as coroutines/CPS functions
