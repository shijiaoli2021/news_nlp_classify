# news_nlp_classify

## 1.赛题目标

这是一个阿里天池的自然语言学习赛。通过对新闻文本数据的特征提取或学习完成对文本分类，共提供数据集大小为20万条数据供训练使用，每条数据包含文本数据“text”和标签数据“label”，提供5万条数据进行线上测试，分类类别数量为14个。所有数据均通过数字形式进行脱敏处理。

## 2.数据与解决方案分析

提供的新闻文本数据集中，所提供的文本数据长度分布跨度极大，最小长度小于10（3），最大长度大于5万个字符。

对于长度不一致文本数据第一反应是对文本进行特征提取，通过构建特征工程得到相同长度的特征向量，随后采用机器学习的方式拟合特征达到分类效果。但是，题目所提供的数据为脱敏数据，每个字符均用数字来表示，这其实一定程度上限制了人为特征工程的构建，例如按照标点符号分句，筛选过滤停用词等等（可以通过数据统计进行一定的分析猜测，但并不保证准确），所以，在特征工程构造方面，最常见的特征构造回到了每个文本的出现词频上，例如采用TF-TDF（词频-逆词频）的方法来构建。

在深度学习方面，目前有两类实现方式：1）卷积、池化、序列预测模型等操作学习文本中词语间关系，例如fasttext，textcnn等，还可以使用LSTM，RNN等序列预测模型。2）采用预训练模型学习文本语义，再通过预训练模型进行微调实现文本分类的目标。这两个方法都会遇到同一个问题：文本的长度长短不一，这导致做深度学习入参长度不一致问题。因此，要做深度学习，首先需要解决长度分布的问题。常见的解决方法有截断、补齐、拆分等，在这个赛题中使用textcnn训练时采用了拆分文本的方法，大部分文本长度都是大于100的，因此，我们设置拆分长度为L（例如128），将长文本按照L拆分为多个小文本，对应相同标签，当文本长度不能被L整除时，从末尾向前截取L长度文本作为最后一个子文本，对于本来就小于L的文本，进行填充处理。对于预训练模型微调，采用文档截断、填充方式进行（例如截断最大长度3000）。

## 4.分类模型

### 4.1 TF-IDF

采用TF-TDF作为特征工程，使用主流分类器进行拟合，例如：岭回归分类器、支持向量机、决策树、随机森林等，采用岭回归分类f1值能达到0.9-0.92左右。

```
# run code
cd ./machine_learning/tfidf

python tf_idf_classify.py
```

### 4.2 fasttext

fasttext通过构建对应词向量与窗口词向量，采用简单的均值、池化、全连接等处理来训练，f1在0.7左右。

```
# run code

cd ./deeplearning/fasttext_lr

python fasttext_tr.py 
```

### 4.3 textcnn

首先，依据拆分数量将文本进行拆分多个子文本构建单词、多词词典，对文本实现卷积操作,首先对文本进行拆分处理，存储为npz训练文件，测试时对于大于拆分长度的文本，拆分为多个相同长度的子文本投入模型得到预测结果，统计预测数量最多的为预测结果,测试f1值在0.85-0.9。

```
# run code
cd ./deeplearning/textcnn

# preprocess data
python run.py --mode data_preprocess

# train
python run.py --mode train 

# test
python run.py --mode test
```

### 4.4 textcnn + attention

加上注意力机制两者效果相差不大，没啥区别，有时候甚至原始的textcnn效果更好，本文在textcnn中新写了textcnnplus来实现，训练代码时只需要替换对应run.py中import的模型即可。

### 4.5 textcnn + RNN

## 4.6 预训练模型+微调

#### 4.6.1预训练

目前借鉴bert训练模式，由于只需要学习文本语义，于是减少了NSP训练任务，只保留了 MLM任务。同时，对于预训练时，采取的截断文本长度为3000，由于资源有限，预训练了mini-bert（L=4， hidden_dim=256）。对预训练模型以batch=4，训练共40万步。

```
#1.pretrain mini-bert just for mlm
cd bert_pretrain
python run.py
```

#### 4.6.2 微调

1.预训练后采用最简单的全连接进行分类任务微调。再用预训练模型进行全连接微调，作为文本分类模型。共训练20万步，f1值能达到0.943-0.947左右。

```
# run code
# prompt classify
cd bert_app
python run.py
```

2.stacking：微调后的bert作为输入接入机器学习完成分类。

```
# run code
# stacking
cd bert_app
python run.py --mode app_stacking --model_on_path ./checkpoints/checkpoint1/trained_model.pth
```

3.lora,使用lora进行模型微调，对此，构建了一套lora适配器和相关工具方法lora_util，适配原有模型，训练时只训练lora部分参数。

```
#run code 
# lora
cd bert_app
python run.py --fine_tuning_for_lora True
```
