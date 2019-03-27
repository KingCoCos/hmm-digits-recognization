import numpy as np
from hmmlearn import hmm
import pickle
import time

def train_data_create(filename):
    f_read = open(filename,"r")
    all_data = f_read.readlines()
    groups = len(all_data) 
    tdata = []
    for i in range(groups):
        line_strlist = all_data[i].split()#生成['1','0',……]
        line_intlist = []
        for j in range(len(line_strlist)-1):#减一是因为最后一列不是观测序列，是label
            line_intlist.append(int(line_strlist[j]))  
        tdata.append(line_intlist)
    tdata = np.array(tdata)
    return tdata

if __name__ == "__main__":
    n_states = 4
    data_filename = ['data/LBG_VQ/train0.txt','data/LBG_VQ/train1.txt','data/LBG_VQ/train2.txt','data/LBG_VQ/train3.txt','data/LBG_VQ/train4.txt','data/LBG_VQ/train5.txt','data/LBG_VQ/train6.txt','data/LBG_VQ/train7.txt','data/LBG_VQ/train8.txt','data/LBG_VQ/train9.txt']
    best_models_set = []
    for i in range(10):
        #多维观测序列
        scores_set = []
        onemodel_parms_set = []
        model = hmm.MultinomialHMM(n_components = n_states, n_iter = 500, tol = 0.01)
        O = train_data_create(data_filename[i])#观测序列
        print("##########",i,i,i,"##########")
        print(O.shape)
        for j in range(10):
            #进行十次训练，取得分最高的模型，减少收敛至局部极大值的影响
            model.fit(O)
            model_parms = {}
            model_parms['pi'] = model.startprob_
            model_parms['A'] = model.transmat_
            model_parms['B'] = model.emissionprob_
            onemodel_parms_set.append(model_parms)
            scores_set.append(model.score(O))

        max_index = scores_set.index(max(scores_set))
        print("数字%d的最佳模型索引%d"%(i,max_index))
        best_models_set.append(onemodel_parms_set[max_index])
        print("找到数字%d的最佳模型"%(i))
        save_model = hmm.MultinomialHMM(n_components = n_states)
        save_model.startprob_ = onemodel_parms_set[max_index]['pi']
        save_model.transmat_ = onemodel_parms_set[max_index]['A']
        save_model.emissionprob_ = onemodel_parms_set[max_index]['B']
        date = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        output_name ='models/' + date + '-model' + str(i) 
        output_file = open(output_name,'wb')
        s = pickle.dump(save_model, output_file)
        output_file.close()
        
    print(best_models_set)#这里出bug了,无法把整个列表打印出来,不知道为什么
    for i in range(len(best_models_set)):
        print(best_models_set[i])
    print()
            # print('初始概率=',model.startprob_) 
            # print('状态转移矩阵=',model.transmat_)
            # print('观测概率=',model.emissionprob_)
            # print('得分=',model.score(O))
