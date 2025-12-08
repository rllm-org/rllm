sympy_issues = [
    ## issue 1
    """Title: TypeError when substituting zeros matrix in Mul

```py
>>> Mul(zeros(2), y, evaluate=False).subs(y, 0)
0  # instead of zeros(2)
>>> (x + y).subs({{x: zeros(2), y: zeros(2)}})
Traceback (most recent call last):
...
TypeError: cannot add <class 'sympy.matrices.immutable.ImmutableDenseMatrix'> and <class 'sympy.core.numbers.Zero'>
```

Getting these error messages
""",
    ## issue 2
    """Title: Wrong result for an integral over complex exponential with a Diracdelta function

I ask Sympy for the complex integral

∫02πexp⁡(−imϕ)δ(ϕ−ϕ0)dϕ,

where m is an integer and δ is the Diracdelta distribution. For ϕ0=0, the above integral yields 0 with SymPy although it should be 1 (or 1/2 depending on the definition of the Delta function if the integral starts at the argument of the δ). For 0<ϕ0<2π, the SymPy result seems correct.

Interestingly, I obtain the (correct) result of 1/2 for ϕ0=2π but zero again for ϕ0=4π. Here is my code:



```
import sympy as sp
# The SymPy version is 1.13.2

phi = sp.symbols(r'\phi', real=True)
m = sp.symbols('m', integer=True)

# This yields 0; it should be 1/2 (phi0 = 0)
sp.integrate(sp.exp(-sp.I * m * phi) * sp.DiracDelta(phi), (phi, 0, 2 * sp.pi))

# This is correct (phi0 = pi/2)
sp.integrate(sp.exp(-sp.I * m * phi) * sp.DiracDelta(phi - sp.pi/2), (phi, 0, 2 * sp.pi))

# This is correct too (phi0 = 2pi)
sp.integrate(sp.exp(-sp.I * m * phi) * sp.DiracDelta(phi - 2 * sp.pi), (phi, 0, 2 * sp.pi))

# Wrong again (phi0 = 4pi)
sp.integrate(sp.exp(-sp.I * m * phi) * sp.DiracDelta(phi - 4 * sp.pi), (phi, 0, 2 * sp.pi))
```
""",
    ## issue 3
    """Title: floor() method on non-primitive causes RecursionError

MWE:

```
from sympy import Matrix, floor

floor(Matrix([1,1,0]))
```

It seems that this ought to gracefully throw an error when called on something non-number like this.
""",
    ### issue 4
    """
```
>>>from sympy import Symbol, ask, Q
>>>i = Symbol('i', integer=True)
>>>i2 = Symbol('i2', integer=True)
>>>ask(Q.rational(i ** i2))
True
```

0^-2 is a counterexample as this value is not defined (because of division by zero).

Currently, the old assumptions consider integers to the power of integers as possibly not rational. Or at least there exists a test to ensure this is the case for the old assumptions.

""",
    ### issue 5
    """This is coming from #15770.

SymPy can create floats that are unequal and compare unequal but have the same repr:

```
In [65]: s1 = integrate(2*exp(1.6*x)*exp(x))

In [66]: s2 = 0.769230769230769*exp(2.6*x)

In [67]: s1        
Out[67]: 2.6⋅x
0.769230769230769⋅ℯ     

In [68]: s2        
Out[68]: 2.6⋅x
0.769230769230769⋅ℯ     

In [69]: repr(s1) == repr(s2)
Out[69]: True

In [70]: s1 == s2  
Out[70]: False

In [7]: s1 - s2    
Out[7]: 2.6⋅x
2.22044604925031e-16⋅ℯ     
```

The repr of the float should ideally be sufficient to recreate the same float or at least the same number even if the precision is different.

""",
]
