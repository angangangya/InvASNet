# Audio_confg
MAX_WAV_VALUE = 32768

# Super parameters
clamp = 2.0
channels_in = 2
# learning rate
lr = 1.0e-4
epochs = 100  # 迭代次数 50000
weight_decay = 1e-5  # 权重衰减，默认是0
init_scale = 0.01
# 使用多个GPU加速训练，[0,1]，要保证batchsize大于GPU的个数
device_ids = [0]

# Super loss
lamda_reconstruction = 2  # s
lamda_guide = 1  # c
lamda_low_frequency = 1
lamda_analyzer = 1

# Train:
batch_size = 4
betas = (0.5, 0.999)  # 用于计算梯度(一阶矩)以及梯度平方(二阶矩)的运行平均值的系数，默认是[0.9,0.999]
weight_step = 5  # 每训练200个epoch，更新一次参数
gamma = 0.5  # 更新lr的乘法因子

# Val:
batchsize_val = 1  # 2
shuffle_val = False
val_freq = 1

# Dataset
# music_training_files = './music_train.csv'
# speech_training_files = './speech_train.csv'
music_training_files = './music_val.csv'
speech_training_files = './speech_val.csv'
music_val_files = './music_test.csv'
speech_val_files = './speech_test.csv'

music_testing_files = './music_test.csv'
speech_testing_files = './speech_test.csv'
# music_testing_files = './GTZAN_rock.csv'
# speech_testing_files = './TIMIT.csv'

# music_val_files = './music_val.csv'
# speech_val_files = './speech_val.csv'

# Display and logging:
loss_display_cutoff = 2.0  # cut off the loss so the plot isn't ruined
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False

# Saving checkpoints:
MODEL_PATH = '/media/l228/数据/zxh/HSA_updwnsp_dwt/models/'  # 保存模型参数
checkpoint_on_error = True
SAVE_freq = 1


TEST_PATH = '/media/l228/数据/zxh/HSA_updwnsp_dwt/audio_dis/'
# 保存测试图的地址
# TEST_PATH = '/media/l228/数据/zxh/Dataset/ISA/'
TEST_PATH_cover = TEST_PATH + 'cover/'
TEST_PATH_secret = TEST_PATH + 'secret/'
TEST_PATH_steg = TEST_PATH + 'steg/'
TEST_PATH_secret_rev = TEST_PATH + 'secret-rev/'


# Load:
suffix_load = ''
train_next = False

trained_epoch = 0

pretrain = False  # 使用提前训练好的模型或者不
PRETRAIN_PATH = '/media/l228/数据/zxh/HSA_updwnsp_dwt/models/'
suffix_pretrain = 'model_checkpoint_00001'

# new train
new_train = True


# d
z_path = '/media/l228/数据/zxh/HSA_updwnsp_dwt/output_z/'
g_path = '/media/l228/数据/zxh/HSA_updwnsp_dwt/output_g/'