# Cluster tweets using K-Means

## Libraries

Following needs to be installed

```python
pip3 install numpy scikit nltk requests ndjson

python

# and now in the interpretor install nltk packages
import nltk
nltk.download()
```

## Commands to run

```python
# To make use of defaults
python tweets-custer

(or)

# All input options are optional but they have to be sent in order
python tweets-custer K <path/to/tweets.json> <path/>to/seeds> <max_iterations> <output_filename>


```

## Preprocessing

* Removed hashtags
* Removed @mentions
* Removed twitter short urls
* Removed unecessary line feeds, carriage returns, spaces and punctuations

### Sum of Squared Errors(SSE) - 8.9265106
