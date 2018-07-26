python main.py --root_path=/home/lupeng/editing_back/expr4 \
--obj=shoes --random=True --large=True --part=True \
--lambda_rec=10 --lambda_constrain=2 \
--gen_triplet=False --lambda_triplet=0 \
--image_size=64 --num_workers=4 --batch_size=16 \
--conv_dim=64 --num_epochs=100 --num_epochs_decay=20 \
--num_iters=1000 --model_save_step=1 \
--pretrained_model=65
