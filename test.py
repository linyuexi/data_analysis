import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

train = pd.read_csv(
    r'/Users/panzixu_mac/Downloads/数据分析/八大项目实战/资料/慕课数据分析中级项目数据/第三章/互联网金融信贷数据分析/input/cs-training.csv')
train.drop('Unnamed: 0', axis=1, inplace=True)
test = pd.read_csv(
    r'/Users/panzixu_mac/Downloads/数据分析/八大项目实战/资料/慕课数据分析中级项目数据/第三章/互联网金融信贷数据分析/input/cs-test.csv')
test.drop('Unnamed: 0', axis=1, inplace=True)
train.rename(
    columns={'SeriousDlqin2yrs': '未来两年可能违约', 'RevolvingUtilizationOfUnsecuredLines': '可用信贷额度比例',
             'age': '年龄',
             'NumberOfTime30-59DaysPastDueNotWorse': '逾期30-59天的笔数', 'DebtRatio': '负债率',
             'MonthlyIncome': '月收入',
             'NumberOfOpenCreditLinesAndLoans': '信贷数量', 'NumberOfTimes90DaysLate': '逾期90天+的笔数',
             'NumberRealEstateLoansOrLines': '固定资产贷款数',
             'NumberOfTime60-89DaysPastDueNotWorse': '逾期60-89天的笔数',
             'NumberOfDependents': '家属数量'}, inplace=True)
test.rename(columns={'SeriousDlqin2yrs': '未来两年可能违约', 'RevolvingUtilizationOfUnsecuredLines': '可用信贷额度比例',
                     'age': '年龄',
                     'NumberOfTime30-59DaysPastDueNotWorse': '逾期30-59天的笔数', 'DebtRatio': '负债率',
                     'MonthlyIncome': '月收入',
                     'NumberOfOpenCreditLinesAndLoans': '信贷数量', 'NumberOfTimes90DaysLate': '逾期90天+的笔数',
                     'NumberRealEstateLoansOrLines': '固定资产贷款数',
                     'NumberOfTime60-89DaysPastDueNotWorse': '逾期60-89天的笔数',
                     'NumberOfDependents': '家属数量'}, inplace=True)
plt.figure(figsize=(20, 20), dpi=300)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
for n, i in enumerate(train.columns):
    plt.subplot(4, 3, n + 1)
    plt.title(i)
    plt.grid(linestyle='--')
    train[i].hist(color='grey', alpha=0.5)
plt.show()

plt.figure(figsize=(20, 20), dpi=300)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
for n, i in enumerate(train.columns):
    plt.subplot(4, 3, n + 1)
    plt.title(i, )
    plt.grid(linestyle='--')
    train[[i]].boxplot(sym='.')
plt.show()

plt.figure(figsize=(10, 5), dpi=300)
sns.heatmap(train.corr(), cmap='Reds', annot=True)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
train['未来两年可能违约'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=axes[0])
axes[0].set_title('未来两年可能违约')
sns.countplot(x='未来两年可能违约', data=train, ax=axes[1])
axes[1].set_title('未来两年可能违约')
plt.show()

from sklearn.feature_selection import mutual_info_classif


def plot_distirbutions_discrete(feature):
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(train[feature], kde=True, label='Train', stat='density', discrete=True, color='steelblue', alpha=0.6,
                 ax=axes[0])
    sns.histplot(test[feature], kde=True, label='Test', stat='density', discrete=True, color='gold', alpha=0.25,
                 ax=axes[0])
    axes[0].legend()
    sns.boxplot(x='未来两年可能违约', y=feature, data=train, ax=axes[1], palette=['seagreen', 'tan'])
    X = train[[feature]].dropna()
    MI = mutual_info_classif(X, train.loc[X.index, '未来两年可能违约'], discrete_features=True, random_state=0)
    axes[1].set_title('Distribution depending on the SeriousDlqin2yrs\n-> MI Score:' + str(round(MI[0], 7)))
    plt.show()


def plot_distributions_continuous(feature):
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.kdeplot(train[feature].apply(np.log1p), label='Train', color='steelblue', alpha=0.5, shade=True, edgecolor='k',
                ax=axes[0])
    sns.kdeplot(test[feature].apply(np.log1p), label='Test', color='gold', alpha=0.3, shade=True, edgecolor='k',
                ax=axes[0])
    sns.boxplot(x='未来两年可能违约', y=train[feature], data=train, ax=axes[1], palette=['seagreen', 'tan'])
    X = train[[feature]].dropna()
    MI = mutual_info_classif(X, train.loc[X.index, '未来两年可能违约'], random_state=0)
    axes[1].set_title('Distribution depending on the SeriousDlqin2yrs\n-> MI Score:' + str(round(MI[0], 7)))
    plt.show()
