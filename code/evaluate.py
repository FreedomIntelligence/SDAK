#! python3
# -*- encoding: utf-8 -*-
'''
@description : 
@Author :  yxfan 
@Contact :  yxfansuda@stu.suda.edu.cn
@Time :  2023/08/27 23:24:58
'''
import json


def parser_anaswr_one_cateory(data):
    
    new_data = {}
    for da in data:
        if da['label_id'] in new_data:
            print(da)
        new_data[da['label_id']] = da
    
    group_list = [ ]
    for key, value in new_data.items():
        if 'pos' in key:
            index = key.split('_')[-1]
            group_list.append([new_data['pos_{}'.format(index)], new_data['neg_{}'.format(index)]])

    positive = 0
    for pos_da, neg_da in group_list:
        if pos_da['label']=='support' and (('正确' in pos_da['predict_0'][:5] and '不正确' not in pos_da['predict_0'][:5])  or '是的' in pos_da['predict_0'][:3]  or '是正确的' in pos_da['predict_0'][:10] ):
            if neg_da['label']=='refute' and ('错误' in neg_da['predict_0'][:3] or '不正确'in neg_da['predict_0'][:10] or '这个说法是错误的' in neg_da['predict_0'][:10] or '这个说法是不正确的' in neg_da['predict_0'][:10] 
                                              or '不完全正确' in neg_da['predict_0'][:10] or '不是' in neg_da['predict_0'][:5]) :
                positive += 1  
                
    return positive, len(group_list)


def follow_rate(data):
    
    new_data = {}
    for da in data:
        if da['label_id'] in new_data:
            print(da)
        new_data[da['label_id']] = da
        
    group_list = [ ]
    for key, value in new_data.items():
        if 'pos' in key:
            index = key.split('_')[-1]
            group_list.append([new_data['pos_{}'.format(index)], new_data['neg_{}'.format(index)]])
            
    following = 0
    
    for pos_da, neg_da in group_list:
        if (('正确' in pos_da['predict_0'][:5] and '不正确' not in pos_da['predict_0'][:5])  or '是的' in pos_da['predict_0'][:3]  or '是正确的' in pos_da['predict_0'][:10] \
        or '错误' in pos_da['predict_0'][:3] or '不正确'in pos_da['predict_0'][:10] or '这个说法是错误的' in pos_da['predict_0'][:10] or '这个说法是不正确的' in pos_da['predict_0'][:10] \
                                                or '不完全正确' in pos_da['predict_0'][:10] or '不是' in pos_da['predict_0'][:5]) and \
            (('正确' in neg_da['predict_0'][:5] and '不正确' not in neg_da['predict_0'][:5])  or '是的' in neg_da['predict_0'][:3]  or '是正确的' in neg_da['predict_0'][:10] \
        or '错误' in neg_da['predict_0'][:3] or '不正确'in neg_da['predict_0'][:10] or '这个说法是错误的' in neg_da['predict_0'][:10] or '这个说法是不正确的' in neg_da['predict_0'][:10] \
                                                or '不完全正确' in neg_da['predict_0'][:10] or '不是' in neg_da['predict_0'][:5]):
            following +=1
    
    return following
                

def output_excel(source_file, des_file):
    import json   
    with open(source_file, 'r', encoding='utf8') as fr:
        lines = fr.readlines()
    datas = [json.loads(line.strip()) for line in lines]
    import xlwt
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet('Sheet1')
    sheet1.write(0,0,'type')
    sheet1.write(0,1,'positive')
    sheet1.write(0,2,'following')
    sheet1.write(0,3,'total')
    for index, da in enumerate(datas):
        print(da.items())
        for key, values in da.items():
            sheet1.write(index+1, 0, key)
            sheet1.write(index+1, 1, values[0])
            sheet1.write(index+1, 2, values[1])
            sheet1.write(index+1, 3, values[2])
    workbook.save(des_file)

def parser_anaswr_all_category(source_file, des_result_file):
    with open(source_file,'r',encoding='utf8')as fr:
        lines = fr.readlines()
    
    category_data_dic = {}
    results_by_categoty = {}
    data = [json.loads(line.strip()) for line in lines]
    for da in data:
        if da['type'] in category_data_dic:
            category_data_dic[da['type']].append(da)
        else:
            category_data_dic[da['type']] = [da]
    total_positive = 0
    total_following = 0
    total = 0
    for type, data in category_data_dic.items():
        temp_positive, temp_total = parser_anaswr_one_cateory(data)
        following =  follow_rate(data)
        total_positive += temp_positive
        total_following += following
        total += temp_total
        results_by_categoty[type] = [temp_positive,following,temp_total]
    
    print('total_positive',total_positive)
    print('total', total )
    with open(des_result_file,'a+',encoding='utf8')as fw:
        for key, value in results_by_categoty.items():
            print(key, value)
            fw.write(json.dumps( {key: value}, ensure_ascii=False)+'\n')

def parser_anaswr_all_category_chatGPT(source_file, des_result_file):
    with open(source_file,'r',encoding='utf8')as fr:
        lines = fr.readlines()
    
    category_data_dic = {}
    results_by_categoty = {}
    data = [json.loads(line.strip()) for line in lines]
    for da in data:
        if da['type'] in category_data_dic:
            category_data_dic[da['type']].append(da)
        else:
            category_data_dic[da['type']] = [da]
    total_positive = 0
    total = 0
    for type, data in category_data_dic.items():
        temp_positive, temp_total = parser_anaswr_one_cateory(data)
        total_positive += temp_positive
        total += temp_total
        results_by_categoty[type] = [temp_positive,temp_total]
    
    print('total_positive',total_positive)
    print('total', total )
    with open(des_result_file,'a+',encoding='utf8')as fw:
        for key, value in results_by_categoty.items():
            print(key, value)
            fw.write(json.dumps( {key: value}, ensure_ascii=False)+'\n')
             
if __name__ =='__main__':
    Dir = 'huatuo-7b-0917-baichuan-config'
    model_name_list = [
                        'baichuan-13b-chat',
                       'baichuan2-13b-chat',
                       'huatuo-chat',
                       'bentsao',
                       'bianque-v2',
                       'chatglm-med',
                       'chatmed-consult',
                       'doctorglm',
                       'chatglm2',
                       'medicalgpt',
                       'qizhen-cama-13b',
                       ]
   #first    
    for model_name in model_name_list[2:3]:
        parser_anaswr_all_category(source_file='./{}/{}-temp-default/modelans.jsonl'.format(Dir, model_name),
                               des_result_file = './{}/{}-temp-default/result_summary.jsonl'.format(Dir, model_name))
        
        output_excel(source_file='./{}/{}-temp-default/result_summary.jsonl'.format(Dir, model_name),
                     des_file='./{}/{}-temp-default/result_summary.xls'.format(Dir, model_name))
    