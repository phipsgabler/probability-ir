# An intermediate representation for probabilistic programs with multiple interpretations

What I propose with the "probabilistic IR" kind of turns around the way  things are constructed right now in all the Julia approaches I've seen. Instead of starting from your sampling function/generative function/model, which is evaluated to get out graphs from it, you start from a representation of the model that already is "graphical", and derive evaluators from it. And if that representation looks like Julia IR, it doesn't matter whether the model is dynamic -- you always work on a fixed, full program.

I think this is feasible also from the end-user perspective, because there is a difference between AD and PPL-DSLs: you can't expect every writer of a mathematical function to anticipate it being used in AD code by marking it as @differentiable or whatever. But you can expect that from the writer of a probabilistic model, because non-standard evaluation in some form is inherently part of the whole probabilistic programming approach.



# Single static sampling form

As far as I see it, the lowest common denominator of all PPLs consists of 

1. General Julia code: this can most conveniently be represented like normal Julia IR with SSA statements, branches, and blocks
2. "Sampling statements": tildes, or assumptions/observations in Turing parlance, which relate names or values to distributions
3. Variable names, which may be "complex", like containing indexing, fields, link functions, etc.

So my idea was to just combine all that into an IR-like syntax.

One of the nice properties of probabilistic programs is that logically, random variables already behave like SSA form -- you can only assign them once. But in something like Turing, assigning to "complex" ones, i.e., arrays, needs to be implemented by mutation of actual data structures, and that annoyed me a lot recently. 

What I would like to have is a SSA language with a semantics in mind that gets rid of this need, by allowing you to use "complex names", that are unified as needed. In essence, this works like a structural equation model in SSA form, extended with control flow, e.g.

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

How this is then interpreted is left abstract. At least (almost) trivially, you can just sample from the prior by putting the variable names into a dictionary, and replacing the tildes with assignments to `rand` calls.  (The complication is that one may have to infer shapes and "unify" variables, like the `x[i]` and `x` above. But there must be at least one general solution to that.)

A specific system can then define its own interpretation or even generate code out of this form. Everything is open to static or dynamic analysis. And an implemention might just accept a fragment of the general variant, e.g., free of stochastic control flow, or with restrictions the indexing of variable names.


## Normal and linear assignment

## Lenses, unifiable link functions, and first class names

- Arrays/`getindex`
- Structs/`getproperty`
- Bijector link functions

Necessary to "unify names".  Copatterns?  See: https://cs.stackexchange.com/questions/129702/semantics-of-write-once-variables-for-complex-data-structures


# Interpretation in Julia IR

## Model compilation as partial specialization

The next step is to perform the interpretation as partial evaluation given the observed data and the sampling algorithm, Mjolnir-style. Then you can "lower" the abstract probabilistic model to concrete Julia IR "optimized" for that specific inference.  If we go back to the above example, supposedly, `N`, the size of `x`, depends on the size of the input. So if we partially evaluate the IR on constant input,  the loop vanishes and a series of `x[1] = ...; ...; x[N] = ...` remains. In that case, “shape” inference should be easy.

## Correspondence with Julia surface syntax


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
