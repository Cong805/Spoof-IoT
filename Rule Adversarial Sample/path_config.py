# coding: utf-8

dataset = 'dataset-dt'  # 数据集目录
train_path = dataset + '/data/train.txt'  # 模型训练集
train_pro_path = dataset + '/data/train_pro.txt'  # 处理过的训练集
val_path = dataset + '/data/valid.txt'  # 模型验证集
val_pro_path = dataset + '/data/valid_pro.txt'  # 处理过的验证集
test_path = dataset + '/data/test.txt'  # 模型测试集
test_pro_path = dataset + '/data/test_pro.txt'  # 处理过的测试集
class_path = dataset + '/data/class.txt'  # 类别集合
vocab_path = dataset + "/data/vocab.pkl"  # 训练集对应的词表
embed_path = dataset + "/data/embedding_banner"  # 训练集对应的词向量矩阵
all_path = dataset + "/data/all.txt"  # 所有数据

train_split_dir = dataset + r'/data/train_split'
raw_data_dir = dataset + r'/data/raw_data'

data_white = dataset + r'/data/white'
raw_hotwords_dir = data_white + r'/raw_hotwords'
hotwords_dir = data_white + r'/hotwords'
fs_hotwords_dir = data_white + r'/fs_hotwords'  # 按词频排序的
processed_data_dir = data_white + r'/processed_data'

data_black = dataset + r'/data/black'
raw_hotwords_dir_b = data_black + r'/raw_hotwords'
hotwords_dir_b = data_black + r'/hotwords'
fs_hotwords_dir_b = data_black + r'/fs_hotwords'  # 按词频排序的
processed_data_dir_b = dataset + r'/processed_data'
sim_vocab_path = data_black + r"/word_cos_sim.json"
pretrain_path = dataset + "/pretrained/index_wv.pkl"  # 预训练词向量
html_tag_pretrain_path = dataset + "/pretrained/html_tag_index_wv.pkl"

user_test_path = '针对规则/testdata'  # 测试数据目录
user_test_data_path = user_test_path + '/dt_test.txt'  # 测试数据
user_test_pre_data_path = user_test_path + '/pre.txt'
user_test_result_path = user_test_path + '/testword.txt'  # 测试数据初始词表
user_test_rd_result_path = user_test_path + '/rd_testword.txt'  # 测试数据去重后词表

modelpath = 'models_dicts'  # 模型参数路径
#dt_cnn_modelpath = 'models_dicts/device_type_TextCNN.ckpt'
dt_cnn_modelpath =  dataset + r'/saved_dict/TextCNN.ckpt'
v_cnn_modelpath =  dataset + r'/saved_dict/TextCNN.ckpt'
dt_rnnatt_modelpath = dataset + r'/saved_dict/TextRNN_Att.ckpt'
dt_rcnn_modelpath = 'models_dicts/device_type_TextRCNN.ckpt'
#p_cnn_modelpath = 'models_dicts/product_TextCNN.ckpt'
p_cnn_modelpath = 'dataset-p/saved_dict/TextCNN.ckpt'
# hotwords = 'hotwords' # 词表目录
# original_hotwords_path = hotwords + '/o_hotwords.txt' # 初始词表
# remove_duplicate_hotwords_path = hotwords + '/rd_hotwords.txt' # 去重后的词表
# remove_duplicate_hotwords_confidence_path = hotwords + '/rd_conf_hotwords.txt' #置信度方法去重词表
# b_original_hotwords_path = hotwords + '/o_b_hotwords.txt' # batch初始词表
# b_remove_duplicate_hotwords_path = hotwords + '/rd_b_hotwords.txt' # batch去重后的词表
# b_remove_duplicate_hotwords_confidence_path = hotwords + '/rd_b_conf_hotwords.txt' #batch置信度方法去重词表
#
# #白盒
# att_hotwords_path = hotwords + '/conf_att_hotwords.txt' #总词表
# camera_words = hotwords + '/camera_uatt_hotwords.txt' #camera类词表
# gateway_words = hotwords + '/gateway_uatt_hotwords.txt' #gateway类词表
# modem_words = hotwords + '/modem_uatt_hotwords.txt' #camera类词表
# printer_words = hotwords + '/printer_uatt_hotwords.txt' #printer类词表
# router_words = hotwords + '/router_uatt_hotwords.txt' #router类词表
# switch_words = hotwords + '/switch_uatt_hotwords.txt' #switch类词表
#
# #黑盒
# black_hotwords_path = hotwords +'/conf_rcnn_hotwords.txt' #总词表
# b_camera_words = hotwords + '/camera_uconfrcnn_hotwords.txt' #camera类词表
# b_gateway_words = hotwords + '/gateway_uconfrcnn_hotwords.txt' #gateway类词表
# b_modem_words = hotwords + '/modem_uconfrcnn_hotwords.txt' #camera类词表
# b_printer_words = hotwords + '/printer_uconfrcnn_hotwords.txt' #printer类词表
# b_router_words = hotwords + '/router_uconfrcnn_hotwords.txt' #router类词表
# b_switch_words = hotwords + '/switch_uconfrcnn_hotwords.txt' #switch类词表
