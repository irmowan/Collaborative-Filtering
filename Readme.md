## Implementation of Collaborative Filtering

[Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) is a technique used by some [recommender systems](https://en.wikipedia.org/wiki/Recommender_system).

[This repository](https://github.com/irmowan/MovieLens) is the Python implementation of Collaborative Filtering.

### Usage

*Run:*

`> python main.py`

*Notice:*

Python Version: 3.5.1

Required modules: Numpy, Pandas, Matplotlib

Need to download the dataset first and put it in the `dataset/` folder.

### Dataset

[MovieLens](http://grouplens.org/datasets/movielens/), 100K dataset

### Report

[推荐系统的协同过滤算法实现和浅析](https://github.com/irmowan/MovieLens/blob/master/report/Report.pdf) is the pdf version of report.

### File tree

```
Python files:
├── main.py			# Main python file including training and testing.
├── predict.py		# Predict functions.
├── utils.py		# Some useful functions, including calculating.
├── var.py			# Define global variables.

Jupyter Notebook files:
├── Cross Validation.ipynb	# Main file of Collartive Filtering.
├── TopK.ipynb		# File to choose K in Top-K algorithm.
├── alpha.ipynb		# File to choose alpha in model blending.
├── MovieLens.ipynb	# Early version file for data cleanning.
├── CF.ipynb		# Early version file about Collarative Filtering.

Others:
├── LICENSE			# MIT LICENSE
├── dataset			# ignored. You can get it from GroupLens Website.
│   ├── ml-100k		# 100K MovieLens dataset
├── papers			# ignored. Papers have been cited in report.
├── report
│   ├── Report.tex	# Raw Tex file. Using XeTeX as the engine.
│   ├── Report.bib	# References.
│   ├── Report.pdf	# Exported pdf report.
│   ├── Plot		# Plot folder. Including Echarts.
│   ├── K-figure.png		# Pictures included in the report.
│   ├── alpha-figure.png
│   └── rating-pie.png
```

### License

[MIT LICENSE](https://github.com/irmowan/MovieLens/blob/master/LICENSE)

### Author

[Irmo](https://github.com/irmowan)

2016.6
