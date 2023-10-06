# AI model optimization

*How to build your own sentiment analysis solution based on Machine Learning?*

💡 Access the full presentation [here](https://noti.st/eleapttn/C1RRRI/a-quel-point-devons-nous-optimiser-nos-modeles-dia).

## Introduction

The aim here is to propose several **sentiment analysis models** and see which solution is the most relevant.

**USE CASE:** analyzing reviews sentiment on a clothing e-commerce site

*You can find out the dataset on [Hugging Face hub](https://huggingface.co/datasets/saattrupdan/womens-clothing-ecommerce-reviews)*

The repository contains the following elements:
- a [notebook](https://github.com/eleapttn/project-model-optimization-sentiment-analysis/blob/main/notebook-train-lstm-sentiment-analysis.ipynb) to learn how to train an **LSTM** model for sentiment analysis
- a [notebook](https://github.com/eleapttn/project-model-optimization-sentiment-analysis/blob/main/notebook-train-bert-sentiment-analysis.ipynb) to learn how to fine-tune a **BERT** model for sentiment analysis
- a [notebook](https://github.com/eleapttn/project-model-optimization-sentiment-analysis/blob/main/notebook-test-lettria-sentiment-analysis.ipynb) to test an "on-shelf" solution **Lettria**
- a [Python code](https://github.com/eleapttn/project-model-optimization-sentiment-analysis/blob/main/evaluate-sentiment-analysis-models.py) to get the global comparaison of these models

## Overview of the models evaluation results

- Metrics calculation for model **LSTM**:

Confusion Matrix
```
 [[  7   8  19   1   5]
 [  6  11  31   5   9]
 [  6  11  60  19  31]
 [  0   4  31  44 128]
 [  0   2  16  36 510]]
```

Accuracy: `0.63`

Macro Precision: `0.44`
Macro Recall: `0.39`
Macro F1-score: `0.40`

- Metrics calculation for model **BERT**:

Confusion Matrix
```
 [[  9  17   8   3   3]
 [ 10  14  30   4   4]
 [  9  13  68  26  11]
 [  0   5  33  75  94]
 [  4   4  11  82 463]]
```

Accuracy: `0.63`

Macro Precision: `0.44`
Macro Recall: `0.43`
Macro F1-score: `0.44`

- Metrics calculation for model **Lettria**:

Confusion Matrix
```
 [[ 30   5   2   1   2]
 [ 18  32  11   1   0]
 [ 18  47  49  12   1]
 [  1  10  40 116  40]
 [  0   3  17 142 402]]
```

Accuracy: `0.63`

Macro Precision: `0.50`
Macro Recall: `0.59`
Macro F1-score: `0.53`

- Models evaluation summary:

```
      metric  model_lettria  model_bert  model_lstm
0   accuracy       0.629000    0.629000    0.632000
1  precision       0.503853    0.439738    0.444379
2     recall       0.585022    0.433896    0.388335
3   f1 score       0.528562    0.435056    0.396866
```
