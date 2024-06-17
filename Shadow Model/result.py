import json


file1 = 'D:/iot/识别/模型工具/DL模型/Model_Based/dataset/data/test_data.json'
file2 = 'D:/iot/识别/模型工具/DL模型/Model_Based/test/DevTag.json'
def count_same_values_vendor(file1, file2):
    data1 = []
    data2 = []
    # 读取第一个JSON文件
    with open(file1, 'r') as f1:
        for line in f1:
            json_data = json.loads(line)
            value = json_data['vendor']
            data1.append(value)
    #print(data1)
    # 读取第二个JSON文件
    with open(file2, 'r') as f2:
        for line in f2:
            json_data = json.loads(line)
            value = json_data['vendor']
            data2.append(value)


    # 统计相同key值对应的value相同的数量

    count = 0

    for elem1, elem2 in zip(data1, data2):
        if elem1 == elem2:
            count += 1
    return count

def count_same_values_product(file1, file2):
    data1 = []
    data2 = []
    # 读取第一个JSON文件
    with open(file1, 'r') as f1:
        for line in f1:
            json_data = json.loads(line)
            value = json_data['product']
            data1.append(value)
    #print(data1)
    # 读取第二个JSON文件
    with open(file2, 'r') as f2:
        for line in f2:
            json_data = json.loads(line)
            value = json_data['product']
            data2.append(value)
    count = 0

    for elem1, elem2 in zip(data1, data2):
        if elem1 == elem2:
            count += 1
    return count
def count_same_values_type(file1, file2):
    data1 = []
    data2 = []
    # 读取第一个JSON文件
    with open(file1, 'r') as f1:
        for line in f1:
            json_data = json.loads(line)
            value = json_data['device_type']
            data1.append(value)
    #print(data1)
    # 读取第二个JSON文件
    with open(file2, 'r') as f2:
        for line in f2:
            json_data = json.loads(line)
            value = json_data['device_type']
            data2.append(value)
    count = 0
    for elem1, elem2 in zip(data1, data2):
        if elem1 == elem2:
            count += 1
    return count
# 调用函数统计数量
result_vendor = count_same_values_vendor(file1, file2)
result_product = count_same_values_product(file1, file2)
result_type = count_same_values_type(file1, file2)
print("识别正确的vendor数量：", result_vendor)
print("识别正确的product数量：", result_product)
print("识别正确的type数量：", result_type)