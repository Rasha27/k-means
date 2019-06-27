import ndjson
import requests
import re
from string import punctuation
from nltk.corpus import stopwords
import sys

stop_words = set(stopwords.words('english'))

class KMeans:
  def __init__(self, tweetsPath, seedsPath, K, maxIterations, outputPath):
    data = self.getTweets(tweetsPath)

    # centroids for each cluster
    # centroids[i] gives i th cluster centroid
    self.centroids = self.getInitialSeeds(seedsPath)

    self.tweetsIdDict = self.preprocessing(data)
    # print(data[0])
    # construct matrix tweet_count X (k + 2)
    self.data = data
    self.K = K
    self.maxIterations = maxIterations
    self.outputPath = outputPath
    self.rows = len(data)

  # Removes common punctuation marks from sentences
  # so that they wont influence jaccard distance
  def strip_punctuation(self, s):
    return ''.join(c for c in s if c not in punctuation)

  # remove @mentions, hashtags and links before clustering
  # Tokenize sanitized tweet content and store it as word set
  def sanitize(self, raw_text):
    # convert lower
    sanitized_text = raw_text.lower()

    regex_str = [
      # Remove replies and tags
      r'(?:@[\w_]+)',
      # hashtags
      r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",
      # remove links as they wont add
      r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
    ]

    sanitized_text = re.sub(r'('+'|'.join(regex_str)+')', ' ', sanitized_text)

    tokens = []
     # break sentence into words
    for word in sanitized_text.split(' '):
      word = word.rstrip().lstrip()
      word = self.strip_punctuation(word)
      if (word != ''):
        tokens.append(word)
    return tokens

  # sanitizes and creates a tweet dictionary
  def preprocessing(self, data):
    tweetsIdDict = {}
    for index, tweet in enumerate(data):
      words = self.sanitize(tweet['text'])
      tweetsIdDict[tweet['id']] = {
        'text': tweet['text'],
        'words': set(words),
        'id': tweet['id']
      }
      data[index]['words'] = set(words)
    return tweetsIdDict

  # Computes jaccard distance for two given word sets
  def jaccardDistance(self, x, y):
    # jDistance = 1 - (x ∩ y)/(x ∪ y)
    intersection = len(x.intersection(y))
    union = len(x.union(y))
    jDistance = 1 - (float(intersection) / union)
    return round(jDistance, 4)

  def getTweets(self, path):
    # load from file-like objects
    response = requests.get(path)
    items = response.json(cls=ndjson.Decoder)
    # data that we need (text, id)
    data = []
    for item in items:
      data.append({
        'text': item['text'],
        'id': item['id']
      })

    return data

  def getInitialSeeds(self, path):
    # get initial seeds
    seeds = []
    response = requests.get(path)
    seeds = [int(i) for i in response.text.replace('\n', '').split(',')]
    return seeds

  def updateCentroids(self, cluster):
    newCentroid = 0
    # as distance wont go beyond 1
    minDistance = 1
    for value in cluster:
      if (minDistance > value['distance']):
        minDistance = value['distance']
        newCentroid = value['tweetId']

    return newCentroid

  # Cluster tweets based on the jaccard distance
  def clusterify(self):
    counter = 0
    centroids = self.centroids
    K = self.K
    while (counter < self.maxIterations):
      counter = counter + 1
      clusters = {}
      # construct matrix tweet_count x K
      distanceMatrix = [[0 for x in range(K)] for y in range(self.rows)]

      # iterate and use jaccard distance and assign centroid
      for i in range(self.rows):
        for j in range(K):
          distanceMatrix[i][j] = self.jaccardDistance(self.data[i]['words'], self.tweetsIdDict[centroids[j]]['words'])

      # assignments of each tweet to its cluster
      # assignments[i] gives cluster centroid for that tweet
      assignments = [0 for x in range(self.rows)]

      for index, row in enumerate(distanceMatrix):
        assignments[index] = centroids[row.index(min(row))]

      centroidIdxDict = {}
      for idx, tweetId in enumerate(centroids):
        centroidIdxDict[tweetId] = idx

      # uniqueCentroids = set(assignments)
      newCentroids = []
      for centroid in set(assignments):
        cluster = [{ 'index': index, 'tweetId': tweet['id'], 'distance': distanceMatrix[index][centroidIdxDict[centroid]] } for index, tweet in enumerate(self.data) if assignments[index] == centroid]
        newCentroid = self.updateCentroids(cluster)
        newCentroids.append(newCentroid)
        clusters[newCentroid] = cluster

      # if (set(centroids) == set(newCentroids)):
      #   print('Converged after %d iterations' % counter)
      #   break

      centroids = newCentroids

    f = open(self.outputPath, 'w')

    clusterCount = 0
    for centroid in set(assignments):
      tweetIds = []
      rawTweets = []
      clusterCount += 1
      for index, tweet in enumerate(self.data):
        if assignments[index] == centroid:
          # print(tweet['text'])
          tweetIds.append(tweet['id'])
          rawTweets.append(tweet['text'])

      f.write(str(clusterCount) + ' ' + ','.join(str(x) for x in tweetIds) + '\n')
      print('Cluster No: ', clusterCount)
      print('-'.join('' for i in range(50)))
      print('\n'.join(x for x in rawTweets))
      print('\n\n\n')

    f.close()
    self.centroids = centroids
    self.clusters = clusters
    return

  # Compute sum of square errors
  def sse(self):
    sum = 0
    for centroid in self.centroids:
      # get the cluster
      cluster = self.clusters[centroid]
      for tweet in cluster:
        distance = self.jaccardDistance(self.data[tweet['index']]['words'], self.tweetsIdDict[centroid]['words'])
        sum += (distance ** 2)
    print('Sum of Squared Errors (SSE): %f' % sum)
    f = open(self.outputPath, 'a')
    f.write('\n\nSum of Squared Errors (SSE): %f' % sum)
    f.close()
    return sum

if __name__ == "__main__":
  defaultSeedsPath = 'http://www.utdallas.edu/~axn112530/cs6375/unsupervised/InitialSeeds.txt'
  defaultTweetsPath = 'http://www.utdallas.edu/~axn112530/cs6375/unsupervised/Tweets.json'
  defaultOutputFile = 'tweet-kmeans-output.txt'
  defaultK = 25
  K = sys.argv[1] if len(sys.argv) > 1 else defaultK
  tweetsPath = sys.argv[2] if len(sys.argv) > 2 else defaultTweetsPath
  seedsPath = sys.argv[3] if len(sys.argv) > 3 else defaultSeedsPath
  maxIterations = sys.argv[4] if len(sys.argv) > 4 else 1000
  outputPath = sys.argv[5] if len(sys.argv) > 5 else defaultOutputFile

  k_means = KMeans(tweetsPath, seedsPath, K, int(maxIterations), outputPath)
  k_means.clusterify()
  k_means.sse()

