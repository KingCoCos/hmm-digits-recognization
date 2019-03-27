#作者:金聪聪
#版本:1.0
#时间:2019.03
#介绍:基于LBG算法的矢量量化,目前只对一维数据进行测试过,二维数据按理应该是可以的
#参考:https://blog.csdn.net/zouxy09/article/details/9153255 和 https://blog.csdn.net/ihaveadream511/article/details/76100496?utm_source=blogxgwz4

import numpy as np
import math
#函数功能介绍:用LBG算法训练得到码本,再对数据进行矢量量化
#输入:训练数据,码矢数量,误差阈值
#输出:矢量量化后的数据集,码矢集
def lbg_train_vq(data, cv_num, e_th):
    #初始码矢数量为1,令码本cvec为所有数据平均值,此时的失真度distor会很大(用欧式距离来衡量)
    rows,cols = data.shape
    cvec = (np.sum(data,axis=1) / cols).reshape(1,-1) #axis = 1,每一行都加起来，最后就一列
    distor =  np.sum((data - cvec)**2) / (rows * cols)
    lnum = int(math.log2(cv_num)) + 1


    #两个迭代器
    #第一个迭代器i表示的是码矢个数的增加,i是2的指数->1,2,4,8,……
    #第二个迭代器while表示的是对于每次分裂出来的码矢集,我们都要重新对数据进行聚类,求得该数量下的最佳码矢集
    for i in range(1,lnum):
        rerr = 10000 
        cvec = np.concatenate((cvec * (1 + e_th),cvec * (1 - e_th)), axis = 1)

        while rerr > e_th:
            eudists_set = np.sum((data - cvec[:,0].reshape(-1,1))**2,axis=0) / rows #axis = 0,每一列都加起来，最后就一行

            #该迭代求取所有码矢与所有训练样本之间的欧式距离
            #存放在eudists_set数组中,行:码矢数,列:训练集数.
            a = 2**i
            for j in range(1,2**i):
                eud =  np.sum((data - cvec[:,j].reshape(-1,1))**2,axis=0) / rows
                eudists_set = np.row_stack((eudists_set,eud))
            new_cvec = np.zeros(cvec.shape)
            new_distor = 0
            match_index = np.zeros(cols)#每个样本匹配的码矢索引
            match_num = np.zeros(cvec.shape[1])#各个码矢匹配样本的数量
            
            #该迭代求得各样本与当前码矢集合cvec中的最佳匹配码矢,并求取新的码矢集合new_cvec
            for k in range(cols):
                dis_col = eudists_set[:,k].reshape(-1,1) #使其为一列,方便后续处理
                #与样本k距离最短的码矢min_index
                min_index = np.where(dis_col == np.min(dis_col))[0][0] 
                match_index[k] = min_index 
                #数据聚类,求和取平均得到新的码矢
                new_cvec[:,min_index] = new_cvec[:,min_index] + data[:,k]
                match_num[min_index] = match_num[min_index] + 1
            new_cvec = new_cvec / match_num

            #该迭代用于计算新的失真度
            for m in range(cols):
                new_distor = new_distor + np.sum((new_cvec[:,int(match_index[m])] - data[:,m])**2)
            new_distor = new_distor / (rows*cols)
            rerr = (distor - new_distor) / distor
            #在下一次迭代前,需要覆盖之前的数据
            distor = new_distor
            cvec = new_cvec
    return match_index,new_cvec

#从txt文件中读取数据(字符串转换为数组)
#输入:文件名
#输出:narray类型的数组
def data_read(filename):
    try:
        f_read = open(filename,"r")
    except:
        print("no such file named %s" %filename)
    all_data = f_read.readlines()
    groups = len(all_data) 
    tdata = []
    for i in range(groups):
        line_strlist = all_data[i].split()#生成['1','0',……]
        line_intlist = []
        for j in range(len(line_strlist)):
            line_intlist.append(int(line_strlist[j]))  
        tdata.append(line_intlist)
    tdata = np.array(tdata)
    f_read.close()
    print(tdata.shape)
    return tdata

#从txt文件中读取数据(数组转换为字符串)
#输入:文件名,数组array
#输出:无
def array_write2file(filename,arr):
    f_write = open(filename,"w")
    rows,cols = arr.shape
    for i in range(rows):
        for j in range(cols):
            int2str = str(arr[i,j])
            int2str = int2str + '\t'
            f_write.write(int2str)
        f_write.write('\n')
    return

#对数据进行矢量量化
#输入:待处理数据data,码本cv
#输出:矢量量化后的数据vq_data
def data_vq(data,cv):
    rows,cols = data.shape
    cv_rows,cv_cols = cv.shape

    eudists_set = np.sum((data - cv[:,0].reshape(-1,1))**2,axis=0) / rows #axis = 0,每一列都加起来，最后就一行
    #该迭代求取所有码矢与所有训练样本之间的欧式距离
    #存放在eudists_set数组中,行:码矢数,列:训练集数.
    for i in range(1,cv_cols):
        eud =  np.sum((data - cv[:,i].reshape(-1,1))**2,axis=0) / rows
        eudists_set = np.row_stack((eudists_set,eud))

    cv_data = np.zeros(cols)#每个样本匹配的码矢索引
    match_num = np.zeros(cv.shape[1])#各个码矢匹配样本的数量

    #该迭代求得各样本与当前码矢集合cvec中的最佳匹配码矢
    for i in range(cols):
        dis_col = eudists_set[:,i].reshape(-1,1) #使其为一列,方便后续处理
        #与样本k距离最短的码矢min_index
        min_index = np.where(dis_col == np.min(dis_col))[0][0] 
        cv_data[i] = min_index 
    return cv_data

if __name__ == "__main__":
    #训练数据得到码本,同时进行矢量量化
    # read_filename = 'data/LBG_VQ_train_data.txt'
    # write_filename = 'data/LBG_VQ_generate_data.txt'
    # train_data = data_read(read_filename).reshape(1,-1)
    # vq_index,code_vectors = lbg_train_vq(train_data,16,.01) 
    # vq_index = vq_index.astype(int).reshape(-1,16) #float数组转换为int,并且转换为每行16个元素
    # array_write2file(write_filename,vq_index)
    # print(code_vectors)

    #用得到的码本对测试数据也进行矢量量化
    code_vectors = np.array([[99.86949021,47.14514768,76.00867375,18.02663262,88.48307631,33.05575117,
                            61.48763205,5.0140124,94.47273115,40.01940771,68.94972989,10.96276392,
                            82.44150268,25.55801342,53.98833404,0.13622126]])
    read_filename = 'data/test_data.txt'
    write_filename = 'data/VQ_test_data.txt'
    test_data = data_read(read_filename).reshape(1,-1)
    vq_test_index = data_vq(test_data,code_vectors)
    vq_test_index = vq_test_index.astype(int).reshape(-1,16)
    array_write2file(write_filename,vq_test_index)

