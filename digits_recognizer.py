import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt#约定俗成的写法plt
import pickle


def test_data_read(filename):
    f_read = open(filename,"r")
    all_data = f_read.readlines()
    groups = len(all_data) 
    tdata = []
    for i in range(groups):
        line_strlist = all_data[i].split()#生成['1','0',……]
        line_intlist = []
        tlabel = []
        for j in range(len(line_strlist)):#因为最后一列是label
            if j < len(line_strlist) - 1:
                line_intlist.append(int(line_strlist[j])) 
            else:
                tlabel.append(int(line_strlist[j])) 
        tdata.append(line_intlist)
    tdata = np.array(tdata)
    return tdata,tlabel

def models_creator(input_models_name):
    input_file = open(input_models_name[0], 'rb')
    model0 = pickle.load(input_file)
    input_file.close()

    input_file = open(input_models_name[1], 'rb')
    model1 = pickle.load(input_file)
    input_file.close()
    
    input_file = open(input_models_name[2], 'rb')
    model2 = pickle.load(input_file)
    input_file.close()

    input_file = open(input_models_name[3], 'rb')
    model3 = pickle.load(input_file)
    input_file.close()

    input_file = open(input_models_name[4], 'rb')
    model4 = pickle.load(input_file)
    input_file.close()

    input_file = open(input_models_name[5], 'rb')
    model5 = pickle.load(input_file)
    input_file.close()
    
    input_file = open(input_models_name[6], 'rb')
    model6 = pickle.load(input_file)
    input_file.close()

    input_file = open(input_models_name[7], 'rb')
    model7 = pickle.load(input_file)
    input_file.close()
    
    input_file = open(input_models_name[8], 'rb')
    model8 = pickle.load(input_file)
    input_file.close()

    input_file = open(input_models_name[9], 'rb')
    model9 = pickle.load(input_file)
    input_file.close()
    return model0,model1,model2,model3,model4,model5,model6,model7,model8,model9

if __name__ == "__main__":
    models_num = 10
    test_filename = ['data/LBG_VQ/test0.txt','data/LBG_VQ/test1.txt','data/LBG_VQ/test2.txt','data/LBG_VQ/test3.txt','data/LBG_VQ/test4.txt','data/LBG_VQ/test5.txt','data/LBG_VQ/test6.txt','data/LBG_VQ/test7.txt','data/LBG_VQ/test8.txt','data/LBG_VQ/test9.txt']
    models_files = ['models/2019-03-27-model0','models/2019-03-27-model1','models/2019-03-27-model2','models/2019-03-27-model3','models/2019-03-27-model4','models/2019-03-27-model5','models/2019-03-27-model6','models/2019-03-27-model7','models/2019-03-27-model8','models/2019-03-27-model9']
    model0,model1,model2,model3,model4,model5,model6,model7,model8,model9 = models_creator(models_files)

    #读取测试集数据0的数据
    accuracy_set = []
    for i in range(len(test_filename)):
        obesers,labels = test_data_read(test_filename[i])
        rows,cols = obesers.shape
        ten_models = []

        maxprob_dig = []
        for j in range(rows):
            scores = []
            s0 = model0.score(obesers[j].reshape(1,-1))#改成obesers[i].reshape(-1,1)也是对的，神奇
            s1 = model1.score(obesers[j].reshape(1,-1))
            s2 = model2.score(obesers[j].reshape(1,-1))
            s3 = model3.score(obesers[j].reshape(1,-1))
            s4 = model4.score(obesers[j].reshape(1,-1))
            s5 = model5.score(obesers[j].reshape(1,-1))
            s6 = model6.score(obesers[j].reshape(1,-1))
            s7 = model7.score(obesers[j].reshape(1,-1))
            s8 = model8.score(obesers[j].reshape(1,-1))
            s9 = model9.score(obesers[j].reshape(1,-1))
            scores = [s0,s1,s2,s3,s4,s5,s6,s7,s8,s9]
            digits = scores.index(max(scores))
            maxprob_dig.append(digits)
        # print(maxprob_dig)
        count = 0
        for k in range(len(maxprob_dig)):
            if maxprob_dig[k] == i:
                count = count + 1
        print(count / len(maxprob_dig))



