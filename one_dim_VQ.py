import xlrd
import xlsxwriter

#一维数据的矢量量化
def vector_quantization(data,data_len,data_range,nums_dim):
    category = range(nums_dim)  #符号化的状态类别
    delta = int((data_range + 1)/ nums_dim)
    for i in range(data_len - 1):
        for j in category:
            if data[i] in range(j*delta,(j+1)*delta+1):
                data[i] = j
                break
    return data

fname_read = "测试数据.xlsx"
fname_write = "VQ10测试数据.xlsx"
excfile = xlrd.open_workbook(fname_read)
workbook= xlsxwriter.Workbook(fname_write)#创建一个excel文件
sheet_name = ['0','1','2','3','4','5','6','7','8','9']
add_sheet_name = ['0_VQ','1_VQ','2_VQ','3_VQ','4_VQ','5_VQ','6_VQ','7_VQ','8_VQ','9_VQ']
dataset = []
for i in range(10):     
    try:
        data_sheet = excfile.sheet_by_name(sheet_name[i])
        VQdata_sheet = workbook.add_worksheet(add_sheet_name[i])
    except:
        print("no sheet in %s" %add_sheet_name[i])
    #获取行数和列数
    rows = data_sheet.nrows
    cols = data_sheet.ncols
    data = []
    for j in range(rows):
        row_data = data_sheet.row_values(j)
        row_data = vector_quantization(row_data,cols,101,10)
        for k in range(cols):
            VQdata_sheet.write(j,k,row_data[k])
workbook.close()