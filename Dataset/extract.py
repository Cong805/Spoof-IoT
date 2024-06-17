import json

input_file = 'http.txt'  # 输入的TXT文件路径
output_file = 'camera_http1.txt'  # 输出的JSON文件路径
target_string = '"device_type": "camera"'  # 目标字符串
#target_string1 = '"device_type": "Camera"'

lines_with_target = []

with open(input_file, 'r') as input_file, open(output_file, 'w') as output_file:
    for line in input_file:
        if target_string in line:
            output_file.write(line)
