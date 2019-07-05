# VAE_CF_git
 参考文献 “Variational Autoencoders for Collaborative Filtering” 的源码：https://github.com/dawenl/vae_cf

 分为四个部分:
 数据处理 VAE_CF_pre.py 
 模型搭建 VAE_CF_model.py 
 模型训练 VAE_CF_train.py 
 预测 VAE_CF_main.py

 处理数据格式 {user_id,item_id,rat}
 
 测试数据逻辑：
 select 
 did ,video_id, rating1, rating2,dt 
 from 
 rating_data_user_movie 
 where 
 (dt='2019-06-5' or dt='2019-06-15'or dt='2019-06-20')
 size（75M）
 
 评估函数：DCG@R、Recall@R

 结果：
 1，训练代数在30代往后，评估函数变化很缓慢，趋于稳定
 2，数据处理不当，容易出现评估函数分母为0而为nan的情况
 
