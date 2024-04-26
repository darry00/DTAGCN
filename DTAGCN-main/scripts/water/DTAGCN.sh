export CUDA_VISIBLE_DEVICES=2

model_name=MVMSGNet

#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_96_192 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --target 'OT' \
#  --seq_len 96 \
#  --pred_len 192 \
#  --e_layers 1 \
#  --d_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 32 \
#  --d_ff 64 \
#  --top_k 5 \
#  --dropout 0.1 \
#  --conv_channel 32 \
#  --skip_channel 32 \
#  --batch_size 32 \
#  --node_dim 10 \
#  --reduction 8 \
#  --U_max 4 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_96_336 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --target 'OT' \
#  --seq_len 96 \
#  --pred_len 336 \
#  --e_layers 1 \
#  --d_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 32 \
#  --d_ff 64 \
#  --top_k 5 \
#  --dropout 0.1 \
#  --conv_channel 32 \
#  --skip_channel 32 \
#  --batch_size 32 \
#  --node_dim 10 \
#  --reduction 8 \
#  --U_max 4 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_96_720 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --target 'OT' \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 1 \
#  --d_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 16 \
#  --d_ff 64 \
#  --top_k 5 \
#  --dropout 0.1 \
#  --conv_channel 32 \
#  --skip_channel 32 \
#  --batch_size 32 \
#  --node_dim 10 \
#  --reduction 8 \
#  --U_max 4 \
#  --itr 1
#
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh2.csv \
#  --model_id ETTh2_96_96 \
#  --model $model_name \
#  --data ETTh2 \
#  --features M \
#  --target 'OT' \
#  --seq_len 96 \
#  --pred_len 96 \
#  --e_layers 1 \
#  --d_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 16 \
#  --d_ff 64 \
#  --top_k 5 \
#  --dropout 0.1 \
#  --conv_channel 32 \
#  --skip_channel 32 \
#  --batch_size 32 \
#  --node_dim 10 \
#  --reduction 8 \
#  --U_max 4 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh2.csv \
#  --model_id ETTh2_96_192 \
#  --model $model_name \
#  --data ETTh2 \
#  --features M \
#  --target 'OT' \
#  --seq_len 96 \
#  --pred_len 192 \
#  --e_layers 1 \
#  --d_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 16 \
#  --d_ff 64 \
#  --top_k 3 \
#  --dropout 0.1 \
#  --conv_channel 32 \
#  --skip_channel 32 \
#  --batch_size 32 \
#  --node_dim 10 \
#  --reduction 8 \
#  --U_max 4 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh2.csv \
#  --model_id ETTh2_96_336 \
#  --model $model_name \
#  --data ETTh2 \
#  --features M \
#  --target 'OT' \
#  --seq_len 96 \
#  --pred_len 336 \
#  --e_layers 1 \
#  --d_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 16 \
#  --d_ff 64 \
#  --top_k 3 \
#  --dropout 0.1 \
#  --conv_channel 32 \
#  --skip_channel 32 \
#  --batch_size 32 \
#  --node_dim 10 \
#  --reduction 8 \
#  --U_max 4 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh2.csv \
#  --model_id ETTh2_96_720 \
#  --model $model_name \
#  --data ETTh2 \
#  --features M \
#  --target 'OT' \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 1 \
#  --d_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 16 \
#  --d_ff 64 \
#  --top_k 3 \
#  --dropout 0.1 \
#  --conv_channel 32 \
#  --skip_channel 32 \
#  --batch_size 32 \
#  --node_dim 10 \
#  --reduction 8 \
#  --U_max 4 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_96_96 \
#  --model $model_name \
#  --data ETTm1 \
#  --features M \
#  --target 'OT' \
#  --seq_len 96 \
#  --pred_len 96 \
#  --e_layers 1 \
#  --d_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 32 \
#  --d_ff 64 \
#  --top_k 5 \
#  --dropout 0.1 \
#  --conv_channel 32 \
#  --skip_channel 32 \
#  --batch_size 32 \
#  --node_dim 10 \
#  --reduction 8 \
#  --U_max 4 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_96_192 \
#  --model $model_name \
#  --data ETTm1 \
#  --features M \
#  --target 'OT' \
#  --seq_len 96 \
#  --pred_len 192 \
#  --e_layers 1 \
#  --d_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 16 \
#  --d_ff 64 \
#  --top_k 3 \
#  --dropout 0.1 \
#  --conv_channel 16 \
#  --skip_channel 32 \
#  --batch_size 32 \
#  --node_dim 10 \
#  --reduction 8 \
#  --U_max 4 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_96_336 \
#  --model $model_name \
#  --data ETTm1 \
#  --features M \
#  --target 'OT' \
#  --seq_len 96 \
#  --pred_len 336 \
#  --e_layers 1 \
#  --d_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 16 \
#  --d_ff 64 \
#  --top_k 3 \
#  --dropout 0.1 \
#  --conv_channel 16 \
#  --skip_channel 32 \
#  --batch_size 32 \
#  --node_dim 10 \
#  --reduction 8 \
#  --U_max 4 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_96_720 \
#  --model $model_name \
#  --data ETTm1 \
#  --features M \
#  --target 'OT' \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 1 \
#  --d_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 16 \
#  --d_ff 64 \
#  --top_k 3 \
#  --dropout 0.1 \
#  --conv_channel 16 \
#  --skip_channel 32 \
#  --batch_size 32 \
#  --node_dim 10 \
#  --reduction 8 \
#  --U_max 4 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_96 \
#  --model $model_name \
#  --data ETTm2 \
#  --features M \
#  --target 'OT' \
#  --seq_len 96 \
#  --pred_len 96 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 32 \
#  --d_ff 64 \
#  --top_k 5 \
#  --dropout 0.1 \
#  --conv_channel 32 \
#  --skip_channel 32 \
#  --batch_size 32 \
#  --node_dim 10 \
#  --reduction 8 \
#  --U_max 4 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_192 \
#  --model $model_name \
#  --data ETTm2 \
#  --features M \
#  --target 'OT' \
#  --seq_len 96 \
#  --pred_len 192 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 16 \
#  --d_ff 64 \
#  --top_k 3 \
#  --dropout 0.1 \
#  --conv_channel 32 \
#  --skip_channel 32 \
#  --batch_size 32 \
#  --node_dim 10 \
#  --reduction 8 \
#  --U_max 4 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_336 \
#  --model $model_name \
#  --data ETTm2 \
#  --features M \
#  --target 'OT' \
#  --seq_len 96 \
#  --pred_len 336 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 16 \
#  --d_ff 64 \
#  --top_k 3 \
#  --dropout 0.1 \
#  --conv_channel 32 \
#  --skip_channel 32 \
#  --batch_size 32 \
#  --node_dim 10 \
#  --reduction 8 \
#  --U_max 4 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_720 \
#  --model $model_name \
#  --data ETTm2 \
#  --features M \
#  --target 'OT' \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --d_model 16 \
#  --d_ff 64 \
#  --top_k 3 \
#  --dropout 0.1 \
#  --conv_channel 32 \
#  --skip_channel 32 \
#  --batch_size 32 \
#  --node_dim 10 \
#  --reduction 8 \
#  --U_max 4 \
#  --itr 1

#python -u run.py \
#    --is_training 1 \
#    --root_path ./dataset/exchange_rate/ \
#    --data_path exchange_rate.csv \
#    --model_id exchange_96_96 \
#    --model $model_name \
#    --data custom \
#    --features M \
#    --freq h \
#    --target 'OT' \
#    --seq_len 96 \
#    --label_len 48 \
#    --pred_len 96 \
#    --e_layers 2 \
#    --d_layers 1 \
#    --factor 3 \
#    --enc_in 8 \
#    --dec_in 8 \
#    --c_out 8 \
#    --des 'Exp' \
#    --d_model 64 \
#    --d_ff 128 \
#    --top_k 3 \
#    --dropout 0.2 \
#    --conv_channel 16 \
#    --skip_channel 32 \
#    --batch_size 32 \
#    --reduction 8 \
#    --U_max 4 \
#    --itr 1
#
#python -u run.py \
#    --is_training 1 \
#    --root_path ./dataset/exchange_rate/ \
#    --data_path exchange_rate.csv \
#    --model_id exchange_96_192 \
#    --model $model_name \
#    --data custom \
#    --features M \
#    --freq h \
#    --target 'OT' \
#    --seq_len 96 \
#    --label_len 48 \
#    --pred_len 192 \
#    --e_layers 2 \
#    --d_layers 1 \
#    --factor 3 \
#    --enc_in 8 \
#    --dec_in 8 \
#    --c_out 8 \
#    --des 'Exp' \
#    --d_model 64 \
#    --d_ff 128 \
#    --top_k 5 \
#    --node_dim 30 \
#    --conv_channel 16 \
#    --skip_channel 32 \
#    --batch_size 32 \
#    --reduction 8 \
#    --U_max 4 \
#    --itr 1
#
#python -u run.py \
#    --is_training 1 \
#    --root_path ./dataset/exchange_rate/ \
#    --data_path exchange_rate.csv \
#    --model_id exchange_96_336 \
#    --model $model_name \
#    --data custom \
#    --features M \
#    --freq h \
#    --target 'OT' \
#    --seq_len 96 \
#    --label_len 48 \
#    --pred_len 336 \
#    --e_layers 2 \
#    --d_layers 1 \
#    --factor 3 \
#    --enc_in 8 \
#    --dec_in 8 \
#    --c_out 8 \
#    --des 'Exp' \
#    --d_model 64 \
#    --d_ff 128 \
#    --top_k 5 \
#    --node_dim 30 \
#    --conv_channel 16 \
#    --skip_channel 32 \
#    --batch_size 32 \
#    --reduction 8 \
#    --U_max 4 \
#    --itr 1
#
#python -u run.py \
#    --is_training 1 \
#    --root_path ./dataset/exchange_rate/ \
#    --data_path exchange_rate.csv \
#    --model_id exchange_96_720 \
#    --model $model_name \
#    --data custom \
#    --features M \
#    --freq h \
#    --target 'OT' \
#    --seq_len 96 \
#    --label_len 48 \
#    --pred_len 720 \
#    --e_layers 2 \
#    --d_layers 1 \
#    --factor 3 \
#    --enc_in 8 \
#    --dec_in 8 \
#    --c_out 8 \
#    --des 'Exp' \
#    --d_model 64 \
#    --d_ff 128 \
#    --top_k 5 \
#    --conv_channel 16 \
#    --skip_channel 32 \
#    --batch_size 32 \
#    --reduction 8 \
#    --U_max 4 \
#    --itr 1

#python -u run.py \
#    --is_training 1 \
#    --root_path ./dataset/weather/ \
#    --data_path weather.csv \
#    --model_id weather_96_96 \
#    --model $model_name \
#    --data custom \
#    --features M \
#    --freq h \
#    --target 'OT' \
#    --seq_len 96 \
#    --label_len 48 \
#    --pred_len 96 \
#    --e_layers 2 \
#    --d_layers 1 \
#    --factor 3 \
#    --enc_in 21 \
#    --dec_in 21 \
#    --c_out 21 \
#    --des 'Exp' \
#    --d_model 64 \
#    --d_ff 128 \
#    --top_k 5 \
#    --conv_channel 32 \
#    --skip_channel 32 \
#    --batch_size 32 \
#    --train_epochs 3 \
#    --reduction 8 \
#    --U_max 4 \
#    --itr 1
#
#python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#    --model_id weather_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --freq h \
#   --target 'OT' \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des 'Exp' \
#   --d_model 64 \
#   --d_ff 128 \
#   --top_k 5 \
#   --conv_channel 32 \
#   --skip_channel 32 \
#   --batch_size 32 \
#   --reduction 8 \
#   --U_max 4 \
#   --itr 1
#
#
#python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#    --model_id weather_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --freq h \
#   --target 'OT' \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des 'Exp' \
#   --d_model 64 \
#   --d_ff 128 \
#   --top_k 5 \
#   --conv_channel 32 \
#   --skip_channel 32 \
#   --batch_size 32 \
#   --reduction 8 \
#   --U_max 4 \
#   --itr 1
#
#pred_len=720
#python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#    --model_id weather_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --freq h \
#   --target 'OT' \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des 'Exp' \
#   --d_model 64 \
#   --d_ff 128 \
#   --top_k 5 \
#   --conv_channel 32 \
#   --skip_channel 32 \
#   --batch_size 32 \
#   --reduction 8 \
#   --U_max 4 \
#   --itr 1

# python -u run.py \
#    --is_training 1 \
#    --root_path ./dataset/traffic/ \
#    --data_path traffic.csv \
#    --model_id traffic_96_96 \
#    --model $model_name \
#    --data custom \
#    --features M \
#    --freq h \
#    --target 'OT' \
#    --seq_len 96 \
#    --label_len 48 \
#    --pred_len 96 \
#    --e_layers 2 \
#    --d_layers 1 \
#    --factor 3 \
#    --enc_in 200 \
#    --dec_in 200 \
#    --c_out 200 \
#    --des 'Exp' \
#    --d_model 512 \
#    --d_ff 512 \
#    --top_k 5 \
#    --conv_channel 16 \
#    --skip_channel 32 \
#    --node_dim 100 \
#    --batch_size 32 \
#    --reduction 8 \
#    --U_max 4 \
#    --itr 1
#
#python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#    --model_id traffic_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --freq h \
#   --target 'OT' \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 200 \
#   --dec_in 200 \
#   --c_out 200 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --top_k 5 \
#   --conv_channel 16 \
#   --skip_channel 32 \
#   --node_dim 100 \
#   --batch_size 32 \
#   --reduction 8 \
#   --U_max 4 \
#   --itr 1
#
#
#python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --freq h \
#   --target 'OT' \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 200 \
#   --dec_in 200 \
#   --c_out 200 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --top_k 5 \
#   --conv_channel 16 \
#   --skip_channel 32 \
#   --node_dim 100 \
#   --batch_size 32 \
#   --reduction 8 \
#   --U_max 4 \
#   --itr 1
#
#
#python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --freq h \
#   --target 'OT' \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 200 \
#   --dec_in 200 \
#   --c_out 200 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --top_k 5 \
#   --conv_channel 16 \
#   --skip_channel 32 \
#   --node_dim 100 \
#   --batch_size 32 \
#   --reduction 8 \
#   --U_max 4 \
#   --itr 1

#for pred_len in 96 192 336 720
#do
#  python -u run.py \
#      --is_training 1 \
#      --root_path ./dataset/Flight/ \
#      --data_path Flight.csv \
#      --model_id Flight_96'_'$pred_len \
#      --model $model_name \
#      --data custom \
#      --features M \
#      --freq h \
#      --target 'UUEE' \
#      --seq_len 96 \
#      --label_len 48 \
#      --pred_len $pred_len \
#      --e_layers 2 \
#      --d_layers 1 \
#      --factor 3 \
#      --enc_in 7 \
#      --dec_in 7 \
#      --c_out 7 \
#      --des 'Exp' \
#      --itr 1 \
#      --d_model 16 \
#      --d_ff 32 \
#      --top_k 5 \
#      --conv_channel 32 \
#      --skip_channel 32 \
#      --node_dim 100 \
#      --reduction 8 \
#      --U_max 4 \
#      --batch_size 32
#done

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/WATER/ \
  --data_path water.csv \
  --model_id water_k_1 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 32 \
  --d_ff 64 \
  --top_k 5 \
  --dropout 0.1 \
  --num_kernels 1 \
  --conv_channel 32 \
  --skip_channel 32 \
  --batch_size 32 \
  --node_dim 10 \
  --reduction 8 \
  --U_max 4 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/WATER/ \
  --data_path water.csv \
  --model_id water_k_3 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 32 \
  --d_ff 64 \
  --top_k 5 \
  --dropout 0.1 \
  --num_kernels 3 \
  --conv_channel 32 \
  --skip_channel 32 \
  --batch_size 32 \
  --node_dim 10 \
  --reduction 8 \
  --U_max 4 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/WATER/ \
  --data_path water.csv \
  --model_id water_k_5 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 32 \
  --d_ff 64 \
  --top_k 5 \
  --dropout 0.1 \
  --num_kernels 5 \
  --conv_channel 32 \
  --skip_channel 32 \
  --batch_size 32 \
  --node_dim 10 \
  --reduction 8 \
  --U_max 4 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/WATER/ \
  --data_path water.csv \
  --model_id water_k_7 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 32 \
  --d_ff 64 \
  --top_k 5 \
  --dropout 0.1 \
  --num_kernels 7 \
  --conv_channel 32 \
  --skip_channel 32 \
  --batch_size 32 \
  --node_dim 10 \
  --reduction 8 \
  --U_max 4 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/WATER/ \
  --data_path water.csv \
  --model_id water_k_51 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 32 \
  --d_ff 64 \
  --top_k 5 \
  --dropout 0.1 \
  --num_kernels 51 \
  --conv_channel 32 \
  --skip_channel 32 \
  --batch_size 32 \
  --node_dim 10 \
  --reduction 8 \
  --U_max 4 \
  --itr 1


