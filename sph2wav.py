from sphfile import SPHFile
import glob
import os


if __name__ == "__main__":
    path = 'E:/Audio_Data/data/lisa/data/timit/raw/TIMIT/TRAIN/*/*/*.WAV'
    sph_files = glob.glob(path)  # 获取路径所有的目录和文件
    print(len(sph_files), "train utterences")
    f = open("./train_files.csv", 'w')
    for i in sph_files:
        sph = SPHFile(i)
        filename = i.replace(".WAV", "_train.WAV")
        sph.write_wav(filename)
        os.remove(i)  # 删除转换前的元音频文件
        f.write(filename.replace("\\", "/"))
        f.write("\n")
    f.close()

    path = 'E:/Audio_Data/data/lisa/data/timit/raw/TIMIT/TEST/*/*/*.WAV'
    sph_files_test = glob.glob(path)
    print(len(sph_files_test), "test utterences")
    f = open("./test_files.csv", 'w')
    for i in sph_files_test:
        sph = SPHFile(i)
        filename = i.replace(".WAV", "_test.WAV")
        sph.write_wav(filename)  # 新的音频文件以_.WAV结尾
        os.remove(i)  # 删除转换前的元音频文件
        f.write(filename.replace("\\", "/"))
        f.write("\n")
    f.close()
    print("Completed")

