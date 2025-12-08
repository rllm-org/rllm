old_default_issues = """
Example 1:

[ISSUE]
Title: TypeError when substituting zeros matrix in Mul

```py
>>> Mul(zeros(2), y, evaluate=False).subs(y, 0)
0  # instead of zeros(2)
>>> (x + y).subs({{x: zeros(2), y: zeros(2)}})
Traceback (most recent call last):
...
TypeError: cannot add <class 'sympy.matrices.immutable.ImmutableDenseMatrix'> and <class 'sympy.core.numbers.Zero'>
```

Getting these error messages
[/ISSUE]

Example 2:

[ISSUE]
Title: Wrong result for an integral over complex exponential with a Diracdelta function

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
[/ISSUE]

Example 3:

[ISSUE]
Title: floor() method on non-primitive causes RecursionError

MWE:

```
from sympy import Matrix, floor

floor(Matrix([1,1,0]))
```

It seems that this ought to gracefully throw an error when called on something non-number like this.

[/ISSUE]

Example 4:

[ISSUE]
I'm using the ImageFilter.Kernel filter with a 3x3 kernel that should shift the image down by one pixel, but instead it is shifting the image one pixel up.

I'm assuming the order of weights in the kernel is left-right, top-bottom because, AFAIK, the documentation doesn't specify the order and I assume it uses the same ordering/coordinates as the image itself.

So, this is either a bug in the code, or a bug in the documentation.

What did you expect to happen?
The image should be shifted one pixel down.

What actually happened?
The image is shifted one pixel up.

```
from PIL import Image, ImageFilter

shift_down = (
    0, 1, 0,
    0, 0, 0,
    0, 0, 0,
)

original_image = Image.open("images/example.png", mode="r")
filtered_image = original_image.filter(ImageFilter.Kernel((3, 3), shift_down))
filtered_image.save("filtered.png", format="PNG")
```
[/ISSUE]

Example 5:

[ISSUE]

The return type of RequestHandler.get_argument depends on a default being provided or not, accounted for using appropriate overloads. Unfortunately, the same isn't true for RequestHandler.get_body_argument and RequestHandler.get_query_argument, which simply declare Optional[str].

Perhaps those need similar overloads, or they could be moved over to RequestHandler._get_argument, called by all...? I'm not sure if most type checkers would infer the rest from that.

[/ISSUE]
"""
