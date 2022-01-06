# algorithms
<hr>
A small collection of functions.
<br><br>





```python
import algo

# arbitrary
algo.beep(t=2)                     # audio beep for t seconds
algo.map_palette(q, palette)       # map palette to quantitative axis
algo.flatten(lst)                  # flatten nested list(s)

# building prior distributions
algo.find_normal(L, U, bulk=0.99, precision=4)
algo.find_studentt(ν, L, U, bulk=0.95, precision=4)
algo.find_gamma(L, U, Lppf=0.005, Uppf=0.995, bulk=None, precision=4)
algo.find_exponential(U, Uppf=0.99, precision=4)
algo.find_weibull(L, U, Lppf=0.005, Uppf=0.995, bulk=None, precision=4)
algo.find_lognormal(L, U, bulk=0.90, precision=4)
algo.find_invgamma(L, U, Lppf=0.005, Uppf=0.995, bulk=None, precision=4)
algo.find_pareto(ymin, U, Uppf=0.99, precision=4)
algo.find_cauchy(L, U, bulk=0.90, precision=4)
```



normal, gamma, invgamma, weibull, lognormal
