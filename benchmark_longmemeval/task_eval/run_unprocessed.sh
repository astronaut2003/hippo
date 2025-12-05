# 测试所有样本
python benchmark_unprocessed_by_qa.py

# 指定模型
python benchmark_unprocessed_by_qa.py --model deepseek-v3:671b

# 只测试 QA_0 到 QA_10
python benchmark_unprocessed_by_qa.py --start 0 --end 10

# 只测试 QA_100 到 QA_199
python benchmark_unprocessed_by_qa.py --start 100 --end 199

# 从 QA_50 开始测试到末尾
python benchmark_unprocessed_by_qa.py --start 50

# 从头测试到 QA_99
python benchmark_unprocessed_by_qa.py --end 99