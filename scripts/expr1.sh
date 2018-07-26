python main.py --root_path=expr/expr1 \
--obj=shoes --data_root=$HOME/dataset/pix2pix/edges2shoes \
--random=True \
--lambda_rec=10 --lambda_constrain=2 \
--gen_triplet=False --lambda_triplet=0 \
--image_size=128 --num_workers=4 --batch_size=16 \
--num_epochs=40 --num_epochs_decay=20 \
--num_iters=1000 --model_save_step=2 
