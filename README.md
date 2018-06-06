# postexperiment
postprocessor for experimental (event based) data.


## Installation
Currently, `postexperiment` only supports python3.

**Users** can install from the latest master using:  
`pip install --user git+https://github.com/skuschel/postexperiment.git`


**Developers** or people who often update should clone the repository  
`git clone git@github.com:skuschel/postexperiment.git`
and then install in development mode via  
`./setup.py develop --user`  
To update just run `git pull` in the directory of the clone.

**Finally** check if the installation succeeded by running:
```python
import postexperiment as pe
print(pe.__version__)
```


## Questions, freature requests, bug reports

 Please use the github [issue tracker](https://github.com/skuschel/postexperiment/issues).
