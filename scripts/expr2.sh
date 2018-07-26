python main.py --root_path=$HOME/editing_back/expr2 \
--obj=shoes --random=True --large=False \
--lambda_rec=10 --lambda_constrain=2 \
--gen_triplet=False --lambda_triplet=0 \
--image_size=128 --num_workers=4 --batch_size=16 \
--num_epochs=90 --num_epochs_decay=20 \
--num_iters=1000 --model_save_step=2 \
--pretrained_model=70
