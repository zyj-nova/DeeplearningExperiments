### 疫情微博情感分析

#### 数据集说明

训练数据和测试数据原始数据为json数据，已经处理为csv格式。每条数据包含id、content、label字段。

不同数值的label对应的类别如下：

```python
class_mapping = {'happy':0, 'sad':1, 'neural':2, 'fear': 3, 'angry':4, 'surprise':5}
```

#### 分词与词向量训练

首先将数据中所有content采用jieba分词工具进行分词，之后采用gensim工具包对分词好的语料库进行词向量的预训练，也可以使用别人已经训练好的词向量。

```python
from gensim.models import Word2Vec
import pandas as pd
import jieba

train_text = pd.read_csv('./virus_train.csv',encoding='utf-8')
test_text = pd.read_csv('./virus_test.csv',encoding='utf-8')

res = []
for t in train_text.content:
    tmp = [word.strip() for word in jieba.cut(t)]
    res.append(tmp)
    
for t in test_text.content:
    tmp = [word.strip() for word in jieba.cut(t)]
    res.append(tmp)

model = Word2Vec(res, size=300, window=6,min_count=2)
model.wv.save_word2vec_format('./weibo.vector.txt',binary=False)

```

模型参考：https://github.com/bentrevett/pytorch-sentiment-analysis

