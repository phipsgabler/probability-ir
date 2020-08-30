# An intermediate form for probabilistic programs with three possible representations


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
