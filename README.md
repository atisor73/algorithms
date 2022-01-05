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

# distributions
algo.find_normal(L, U, bulk=0.99, precision=4)
algo.find_exponential(U, Uppf=0.99, precision=4)
algo.find_invgamma(L, U, Lppf=0.005, Uppf=0.995, bulk=None, precision=4)
```

