

# Quantum random walks

Based on the following reference papers about quantum random walk, this notebook implements the calculation and animation of classic and quantum random walk of different graphs and various coin operators:
- [Julia Kempe “Quantum random walks - an introductory overview”](https://arxiv.org/abs/quant-ph/0303081)
- [Neil Shenvi, Julia Kempe, K. Birgitta Whaley "A Quantum Random Walk Search Algorithm"](https://arxiv.org/abs/quant-ph/0210064)

apart from examples from the these papers the jupyter notebook also features some other variations of quantum random walk to explore as well as additional anylsis to understand the dynamics

all content flows into the jupyter notebook "Quantum random walks demo", all calculations are in util functions so that this notebook just contains demo and analysis, animations are partly saved as gif and embedded in notebook

all calculations are done in plain numpy, using [einstein sum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) for the 3-dimensional matrix multiplications of quantum random walk, where a transition is done in each dimension of the coin space  

disclaimer: the notebook does not serve as intro to quantum random walks, since no background information from the papers is repeated here, so the papers above are recommended to be read along with the notebook. Also is the terminology here rather sloppy and not an accurate description of the underlying tensor spaces

animations are done with plotly as well as matplot, latter based on [this post](https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c). A util function is provided to do similar calls as for plotly animation for the matplot animation, with pandas df input.
matplot animation has disadvantage that it writes gif, which is not conveniently embedded/refreshed in jupyter. Plotly has disadvantage that line graph is buggy exect if spline interpolation is used, here i use the bar plot instead.


### Content of _Quantum random walks demo_ notebook:
1. Classical random walk on the circle
2. Quantum Random walk on circle
3. Hypercube graph: Classical & Quantum random walk
4. Quantum walk search algorithm