# FastTextRank

Ahora soporta español con NLTK: ver `FastTextRank/FastTextRankSpanish.py`.

## Español

Para resumen y extracción de palabras clave en español, usa:

```python
from FastTextRank.FastTextRankSpanish import FastTextRankSpanish, FastTextRankSpanishKeywords

texto = "La inteligencia artificial es una tecnología revolucionaria..."
extractor = FastTextRankSpanish()
resumen = extractor.summarize(texto, 3)

extractor_kw = FastTextRankSpanishKeywords()
palabras_clave = extractor_kw.extract_keywords(texto, 5)
```

# 快速文本摘要及关键词提取
从中文文本中提取摘要及关键词，并对算法时间复杂度进行了修改，计算图最大权节点的时间复杂度由o（n^2）降低到了o（n）。在有限的测试文本上（10篇文章），其运行速度相比于textrank4zh这个包快了8倍。[算法原理见知乎文章](https://zhuanlan.zhihu.com/p/41241390)
## 安装
Numpy>=1.14.5
gensim>=3.5.0
pip install FastTextRank==1.1

## 使用
详情请见./FastTextRank/test文件夹<br/>
KeyWord.py：提取关键字示例<br/>
Sentence.py：提取摘要示例<br/>

## 额外
如有优化点，欢迎pull requests<br/>
如有问题，欢迎提issues<br/>

# FastTextRank
Extract abstracts and keywords from Chinese text, use *optimized iterative algorithms* to improve running **speed**, and *selectively use word vectors* to improve **accuracy**.
## PageRank
PageRank is a website page ranking algorithm from Google.<br/>
PageRank was originally used to calculate the importance of web pages. The entire www can be seen as a directed graph, and the node is a web page.<br/>
This algorithm can caculate all node's importance by their connections.<br/>
* My algorithm changed the iterative algorithm to make the algorithm much faster, it costs 10ms per article, on the mean while TextRank4ZH costs 80ms on my data.<br/>
* My algorithm also use word2vec to make the abstract more accurate, but it will cost more time to run the algorithm. Using word2vec costs 40ms per article on the same traning data.

## FastTextRank4Sentence
### Introduction
1. Cut article into sentence
2. Calculate similarity between sentences:
   * Using word vectors' cosine similarity
   * Using two sentences' common words
3. Build a graph by sentences' similarity
4. Caculate the importance of each sentence by improved iterative algorithm
5. Get the abstract
### API
* use_stopword: boolean, default True
* stop_words_file: str, default None.
The stop words file you want to use. If it is None, you will use this package's stop words.
* use_w2v: boolean, default False
If it is True, you must input passing dict_path parameter.
* dict_path: str, default None.
* max_iter:maximum iteration round
* tol: maximum tolerance error

## FastTextRank4Word

### Introduction
1. Cut artile into word
2. Calculate similarity between word: 
   If two words are all in window distance, then the graph's side of this two word add 1.0. Window is set by user.
3. Build a graph by word' similarity
4. Caculate the importance of each word by improved iterative algorithm
5. Get the key word

### API
* use_stopword=boolean, default True
* stop_words_file=str, default None.
The stop words file you want to use. If it is None, you will use this package's stop words.
* max_iter=maximum iteration round
* tol=maximum tolerance error
* window=int, default 2
The window to determine if two words are related