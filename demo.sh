#Train
python main.py  --scale 8 --save EDSR-PP_0110 --reset --chop_forward  --pre_train_1 /home/kky/model_5.pt 

# Test your own images
#python main.py --scale 8 --data_test MyImage --test_only --save_results --pre_train_1 /data01/kky/experiment/EDSRX8_1021/model/model_134.pt  --chop_forward



