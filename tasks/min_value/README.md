## Optimization (1 POINT)

`scipy` `optimize` `minimize`

### Task

Implement function `find_minimum` that finds the (local) 
minimum value and its location for a given function f(x).
Use 0 as the initial guess.

```python
>>> def f(x):
...     return (x - 2) ** 2 + 1
>>> find_minimum(f)
(1.0000000000000007, 1.9999999746237656)
```