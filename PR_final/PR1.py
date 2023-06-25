import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(1)

# 距离计算函数（2范数）
def Distance(x1,x2):
    s = 0
    for i in range(len(x1)):
        s += np.power((np.abs(x1[i] - x2[i])), 2)
    return np.power(s, 0.5)

# 展示二维高斯分布
def Show(data_path):
    data = np.array(scio.loadmat(data_path)['data']).T
    data,w = PCA(data, 2)
    #print(data.shape)
    sns.kdeplot(data[:, 0], data[:, 1], shade=True)
    plt.show()


# PCA降维函数
def PCA(data, d_new):
    n, d = data.shape[0], data.shape[1]
    Data = np.array(data)
    # 所有样本中心化
    for j in range(d):
        Data[:, j] -= np.full(n, np.mean(Data[:, j]))
    # 分解协方差矩阵,取前d_new大特征值对应的特征向量作为投影矩阵W
    c = np.linalg.eig(np.matmul(Data.T, Data) / (n - 1))
    index = np.argsort(-c[0])  # 特征值降序排序
    W = (c[1])[:, index[0:d_new]]
    Data_pca, d = np.matmul(Data, W), d_new
    return Data_pca, W

# PCA分析特征值比例
def PCA_analysis(data_path):
    data = np.array(scio.loadmat(data_path)['data']).T
    n, d = data.shape[0], data.shape[1]
    Data = np.array(data)
    for j in range(d):
        Data[:, j] -= np.full(n, np.mean(Data[:, j]))
    c = np.linalg.eig(np.matmul(Data.T, Data) / (n - 1))
    np.sort(-c[0])  # 特征值降序排序
    lamda_sum_proportion = np.zeros(d)  # 第i个元素表示前i大特征值所占比例和
    lamda_sum_proportion[0] = c[0][0]
    for i in range(1, d):
        lamda_sum_proportion[i] = lamda_sum_proportion[i - 1] + c[0][i]
    lamda_sum_proportion /= np.sum(c[0])

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(d), lamda_sum_proportion)
    plt.xlabel('num of dimensions', fontsize=15)
    plt.ylabel('lamda_sum_proportion', fontsize=15)
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(d), c[0])
    plt.xlabel('dimension', fontsize=15)
    plt.ylabel('lamda', fontsize=15)
    plt.show()


# PCA可视化（2维,无类别）
def PCA_visualize(data_path):
    data = np.array(scio.loadmat(data_path)['data']).T
    Data_pca, W = PCA(data, 2)
    plt.scatter(Data_pca[:, 0], Data_pca[:, 1], s=2)
    plt.show()

def train(label_data_path, label_path, unlabel_data_path):
    # 数据导入
    label_data = np.array(scio.loadmat(label_data_path)['data']).T       # 有标签样本
    unlabel_data = np.array(scio.loadmat(unlabel_data_path)['data']).T   # 无标签样本
    label = np.array(scio.loadmat(label_path)['label'][0], dtype='int')  # 标签

    ############################# 临时 ############################
    random_train_list = np.random.choice(np.arange(1500), num_for_train, False)
    random_test_list = np.delete(np.arange(1500), random_train_list)
    label_data = label_data[random_train_list]
    label = label[random_train_list]
    #################################################################

    data = np.r_[label_data, unlabel_data]  # 有标签在前，无标签在后
    n = data.shape[0]  # 总样本数
    d = data.shape[1]  # 维数
    L_label = len(label_data)      # 有标签样本数
    L_unlabel = len(unlabel_data)  # 无标签样本数
    l = np.zeros(3)   # 确定每一类有标记样本数目
    for i in label:
        l[i] += 1

    # min-max归一化
    for j in range(d):
        Max, Min = max(data[:, j]), min(data[:, j])
        for i in range(n):
            data[i][j] = (data[i][j] - Min) / (Max - Min)

    # PCA降维
    data, W = PCA(data, d_new)
    d = d_new

    # 半监督的高斯混合聚类
    # 初始化参数
    μ = []  # 均值
    Σ = [np.eye(d), np.eye(d), np.eye(d)]  # 协方差矩阵都初始化为单位矩阵
    a = [1/3, 1/3, 1/3]   # 混合系数初始化为全部相同
    γ = np.zeros((n, 3))  # 后验概率矩阵

    # GMM对初值敏感，用k-means++的初始化方法确定初始均值
    # 在有标签样本中初始化，使得第i个高斯分量的类别是i
    group = [[], [], []]  # group[i]表示i类的样本集
    for i in range(L_label):
        group[label[i]].append(i)
    # 1.选第一个中心
    min_distance = []  # 每个0类样本到1,2类样本的最小距离
    for i in group[0]:
        distance = []
        for j in np.r_[group[1], group[2]]:
            distance.append(Distance(data[i], data[j]))
        min_distance.append(min(distance))
    i_first = group[0][np.argmax(min_distance)]  # 取离1,2类样本最近距离最远的0类样本为第一个均值
    μ.append(data[i_first])

    # 2.再找两个中心
    for t in range(2):
        min_distance = []
        for i in group[t]:
            tmp = []
            for CenterPoint in μ:
                tmp.append(Distance(data[i], CenterPoint))
            min_distance.append(min(tmp))
        i_next = group[t][np.argmax(min_distance)]  # 取离现有中心最近距离最远的样本作为新中心
        μ.append(data[i_next])

    # EM算法迭代，直到满足条件
    time = 0
    while 1:
        time += 1
        print('time:', time)

        # E步：计算后验概率矩阵γ, γ[j][i]表示样本j由第i个高斯分量产生的概率
        for j in range(n):
            p_list = []
            for i in range(3):
                step1 = 1 / np.power(2 * np.pi, d / 2) / np.sqrt(np.abs(np.linalg.det(Σ[i])))
                step2 = np.matmul(data[j] - μ[i], np.linalg.inv(Σ[i]))
                step3 = -0.5 * np.matmul(step2, data[j] - μ[i])
                p_list.append(a[i] * step1 * np.exp(step3))
            γ[j] = p_list / np.sum(p_list)

        # M步：更新均值、协方差矩阵、均值
        μ_new, Σ_new, a_new = [], [], []
        for i in range(3):
            sum1, sum2, sum3, sum6 = 0, 0, 0, 0
            sum4, sum5 = np.zeros((d, d)), np.zeros((d, d))
            # 1.更新均值和混合系数
            for j in range(n):
                if j >= L_label:  # 无标签
                    sum1 += γ[j][i]
                    sum2 += γ[j][i] * data[j]
                if j < L_label and label[j] == i:  # 有标签且类别为i
                    sum3 += data[j]
            μ_new.append((sum2 + sum3) / (sum1 + l[i]))
            a_new.append((sum1 + l[i]) / n)

            # 2.更新协方差矩阵
            for j in range(n):
                if j >= L_label:  # 无标签
                    err = np.mat(data[j] - μ_new[i])
                    sum4 += γ[j][i] * np.matmul(err.T, err)
                if j < L_label and label[j] == i: # 有标签且类别为i
                    err = np.mat(data[j] - μ_new[i])
                    sum5 += np.matmul(err.T, err)
            Σ_new.append((sum4 + sum5) / (sum1 + l[i]))

        # 均值不再变化或超过最大迭代轮数就结束迭代，否则更新参数
        if (np.array(μ) == np.array(μ_new)).all() or time >= MaxTime:
            break
        else:
            μ, Σ, a = μ_new, Σ_new, a_new
    # EM迭代结束

    # 确定样本的簇标记λ,并把样本归入各个分量中
    λ = np.zeros(n)  # 每个样本的簇标记
    Cluster = [[], [], []]  # 簇，每个元素是一个簇（样本编号集）
    for j in range(n):
        λ[j] = np.argmax(γ[j])
        Cluster[int(λ[j])].append(j)

    # 利用有标签样本以多数原则确定高斯分量的类别
    category = []  # category[i]代表高斯分量i的类别
    for i in range(3):
        num = np.zeros(3)
        for j in Cluster[i]:
            if j < L_label:
                num[label[j]] += 1
        category.append(np.argmax(num))

    return Min, Max, W, μ, Σ, a, category, random_test_list  ########## 临时 ########
    # return Min, Max, W, μ, Σ, a, category


def test(test_data_path, test_label_path, model):
    Min, Max, W, μ, Σ, a, category, random_test_list = model  ########## 临时 ########
    # Min, Max, W, μ, Σ, a, category = model

    data = np.array(scio.loadmat(test_data_path)['data']).T
    label = np.array(scio.loadmat(test_label_path)['label'][0], dtype='int')

    ############################ 临时 ####################################
    data_train = np.array(scio.loadmat('data_train.mat')['data']).T
    label = np.array(scio.loadmat('label_train.mat')['label'][0], dtype='int')
    data = data_train[random_test_list]
    label = label[random_test_list]
    ##################################################################

    n = data.shape[0]  # 测试样本数
    d = data.shape[1]  # 维数

    # min-max归一化
    for j in range(d):
        for i in range(n):
            data[i][j] = (data[i][j] - Min) / (Max - Min)

    # 用训练集的投影矩阵W来给测试集数据PCA降维
    for j in range(d):
        data[:, j] -= np.full(n, np.mean(data[:, j]))
    data, d = np.matmul(data, W), d_new

    # 给测试样本分类
    label_predict = []
    for j in range(n):
        p_list = []  # 计算每个样本由各个高斯分量生成的概率，取argmax
        for i in range(3):
            step1 = 1 / np.power(2 * np.pi, d / 2) / np.sqrt(np.abs(np.linalg.det(Σ[i])))
            step2 = np.matmul(data[j] - μ[i], np.linalg.inv(Σ[i]))
            step3 = -0.5 * np.matmul(step2, data[j] - μ[i])
            p_list.append(a[i] * step1 * np.exp(step3))
        label_predict.append(category[np.argmax(p_list)])

    # 对比真实label得出准确率
    num_correct = 0
    for i in range(n):
        num_correct += (label_predict[i] == label[i])
    acc = num_correct / n
    print('准确率：', acc)

    # 作图展示测试集上分类结果
    color = ['red', 'blue', 'green']
    Group = [[], [], []]
    for i in range(n):
        Group[label_predict[i]].append(i)
    for i in range(3): # 第i类
        plt.scatter(data[Group[i]][:, 0], data[Group[i]][:, 1], c=color[i], s=2, label=i) # 展示某两维
    plt.title('classification result on test set', fontsize=15)
    plt.legend()
    plt.show()

    return acc



if __name__ == '__main__':
    # 参数
    MaxTime = 20    # 最大迭代轮数(仅不收敛时用）
    d_new = 30      # PCA降到d_new维
    num_for_train = 300  # 1500个标签样本中选这么多来训练，其余当测试集
    # PCA_analysis('data_train.mat') # 分析特征值的比例
    #Show('data_train.mat')  # 展示高斯分布
    #
    np.random.seed(5)
    model = train('data_train.mat', 'label_train.mat', 'data_unsupervise.mat')  # 训练
    acc = test('data_test.mat', 'label_test.mat', model)  # 先用有标签训练集测试
    #PCA_visualize('data_test.mat')

     # 3倍交叉验证降维数 # 取300个标签样本训练
    # num_for_train = 300
    # avr_acc_list = []
    # for d_new in np.arange(2,100,2):
    #     print(d_new)
    #     acc_sum = 0
    #     for i in range(3):
    #         np.random.seed(d_new * i)
    #         model = train('data_train.mat', 'label_train.mat', 'data_unsupervise.mat')
    #         acc_sum += test('data_train.mat', 'label_train.mat', model)
    #     avr_acc_list.append(acc_sum / 3)
    # plt.plot(np.arange(2,100,2), avr_acc_list)
    # plt.xlabel('num of dimensions',fontsize=12)
    # plt.ylabel('average accuracy',fontsize=12)
    # plt.show()
