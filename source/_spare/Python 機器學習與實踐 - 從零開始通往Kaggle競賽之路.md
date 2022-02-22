---
title: Python 機器學習與實踐 - 從零開始通往 Kaggle 競賽之路
tags:
---

## 簡介篇
監督學習 Supervised Learning
關注對事物未知表現的預測
分類問題 Classification
回歸問題 Regression

無監督學習 Unsupervised Learning
傾向於對事物本身特性的分析
數據降維 Dimensionality Reduction
聚類問題 Clustering

特徵 Feature
反映數據內在規律的信息

特徵向量 Feature Vector
描述一個數據樣本

標記/目標 Label/Target
表現形式取決於監督學習的種類

## 基礎篇 監督學習經典模型
二分類 Binary Classification
多類分類 Multiclass Classification
多標籤分類 Multi-lable Classification

線性分類器 Linear Classifiers
邏輯斯蒂回歸模型 Ligistic Regression
隨機梯度上升算法 Stochastic Gradient Ascend http://cs229.stanford.edu/notes/cs229-notes1.pdf

準確性 Accuracy
召回率 Recall
精確率 Precision

支持向量機分類器 Support Vector Classifier

樸素貝葉斯 Naive Bayes
樸素貝葉斯分類器的構造基礎是貝葉斯理論

K 近鄰算法 K Neighbor Classifier
近朱者赤，近墨者黑
可以用來對生物物種進行分類
屬於無參數模型 Nonparametric model

決策樹
描述非線性關係的不二之選
決策樹節點 node 代表數據特徵
節點下的分支代表對應特徵值的分類
決策樹葉子節點 leaf 則顯示模型的決策結果
信息摘 Information Gain
基尼不純性 Gini Impurity

集成分類模型 Ensemble
綜合考量多個分類器的預測結果
主要分為兩種
一種是利用相同的訓練數據同時搭建多個獨立的分類模型，然後以少數服從多數的原則做出最終分類決策
另一種是按造一定次序搭建多個分類模型

隨機森林分類器 Random Forest Classifier
在相同訓練數據上同時搭建多顆決策樹 Decision Tree

梯度提升決策樹 Gradient Tree Boosting

工業界為了追求更強勁的預測性能，經常使用隨機森林分類模型作為基線系統 Baseline System


回歸預測
期待預測的目標是連續變量

線性回歸器
隨機梯度下降 Stochastic Gradient Descend
最簡單、易用的回歸模型，再不清楚特徵之間關係的前提下，仍然可以使用線性回歸模型作為大多數科學試驗的基線系統 Baseline system

支持向量機回歸
可以通過配置不同的核函數來改變模型效能
可多嘗試幾種配置，進而獲得更好的預測性能

K 近鄰回歸
屬於無參數模型 Nonparametric model
計算方法非常直觀，深受廣大初學者的喜愛

回歸樹
樹模型可以解決非線性特徵的問題
樹模型不要求對特徵標準化盒統一量化
樹模型可以直觀地輸出決策過程，使預測結果具有可解釋性
缺點
容易因為模型搭建過於複雜而喪失對新數據預測的精度
預測穩定性較差

集成回歸模型
雖然集成模型在訓練過程中要耗費更多的時間
但往往可以提供更高的表現性能和更好的穩定性



## 基礎篇 無監督學習經典模型
無監督學習 Unsupervised Learning 著重發現數據本身的分佈特點
不需要對數據進行標記
可以幫助我們發現數據的“群落”
也可以尋找“離群”的樣本


數據聚類
是無監督學習的主流應用之一

K 均值算法 K-means
最為經典並且易用的聚類模型
算法執行過程
1. 隨機布設 K 個特徵空間內的點作為初始的聚類中心
2. 根據每個數據的特徵向量，從 K 個聚類中心中尋找距離最近的一個，並把該數據標記為從屬於這個聚類中心
3. 在所有數據都標記過後，重新對 K 個聚類中心做計算
4. 執行一輪後，若所屬的聚類中心沒有變化，則迭代停止，否則回到步驟2


特徵降維
我們經常會在實際項目中遭遇特徵維度非常之高的訓練樣本
我們無法用肉眼觀測超過三個維度的特徵
主成分分析 Principal Component Analysis



## 進階篇
NLTK：自然語言處理
Word2Vec：詞向量技術
XGBoost：提供強大預測能力
Tensorflow：Google發布的用於深度學習的框架

特徵抽取
