# 毕业项目：为 Arvato Financial Services 金融服务公司实现一个顾客分类报告

优达学城数据科学家纳米学位毕业项目。

## 使用开发环境

* python 3.7.4
* conda 4.8.3
* numpy 1.18.1
* pandas 1.0.2
* sklearn 0.22.1
* scikitplot 0.3.7

## 文件说明

* imgs 博文使用的图片
* terms_and_conditons 条款声明
* kaggle_result.csv 提交kaggle的结果文件
* rfc_result.csv 通过随机森林的预计结果，也提交过kaggle
* Avato_Project_workbook_zh.ipynb 工程记录实现代码
* notice.md 博文原稿
* Arvato_Project_Workbook_zh.html 笔记本导出的html
* DIAS Attributes - Values 2017.xlsx Arvato提供的数据说明文件
* DIAS Information Levels - Attributes 2017.xlsx Arvato提供的数据说明文件

## 工程使用方法

代码下载后，可以通过Jupyter Notebook直接打开Avato_Project_workbook_zh.ipynb。由于这个项目使用的数据是Arvato Financial Services为优达学城纳米学位数据科学家提供的，如果你有优达学城毕业项目工作区才可以获得数据。这里我把数据放在了本地的'C:\Data\avator_data\'中，如果要使用不同的文件路径，请在Avato_Project_workbook_zh中的第二步进行相应修改。

```python
workspace_path = {'windows': 'C:\\Data\\avator_data\\', 
                  'udactity': '../../data/Term2/capstone/arvato_data/'}
drive_path = workspace_path['windows']
```

我使用的conda环境没有默认安装scikitplot。需要通过手动安装，具体方法参阅[Conda下scikit-plot安装](https://anaconda.org/conda-forge/scikit-plot)

## 项目结果

这个项目的最后结果有两个部分

* jupyter notebook 在[Github](https://github.com/lqiang79/udacity_DSND_arvato)
* 博文发表在[Medium](https://medium.com/@lqiang79/%E5%AE%A2%E6%88%B7%E5%9C%A8%E5%93%AA%E9%87%8C-%E6%95%B0%E6%8D%AE%E5%91%8A%E8%AF%89%E4%BD%A0-e27dd615c4c1)

## 参阅资料

* [特征选择(Feature Selection)方法汇总](https://zhuanlan.zhihu.com/p/74198735)
* [Feature Selection with sklearn and Pandas](https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b)
* [Categorical Data](https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63)
* [What is One Hot Encoding? Why And When do you have to use it?](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f)
* [How to Use ROC Curves and Precision-Recall Curves for Classification in Python](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)
* [理解主成分分析 (PCA)](https://zhuanlan.zhihu.com/p/37810506)
* [First steps with Scikit-plot](https://scikit-plot.readthedocs.io/en/stable/Quickstart.html)