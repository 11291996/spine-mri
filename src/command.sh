data_dir=/data/datasets/spine/gtu/train
#this is the default value in this server
save_dir=/data/model/
model_dir=/data/model/pix2pix/spine/gtu/G_t1_t2.pth
diffusion_model_dir=/data/model/diffusion/spine/gtu/model_t1_t2.pth

#training codes save the model in the model directory
/home/jaewan/spine-diff/spine-venv/bin/python3 -m src.pix2pix --data_dir $data_dir --save_dir $save_dir
/home/jaewan/spine-diff/spine-venv/bin/python3 -m src.cyclegan --data_dir $data_dir
/home/jaewan/spine-diff/spine-venv/bin/python3 -m src.gan_score --model_dir $model_dir
/home/jaewan/spine-diff/spine-venv/bin/python3 -m src.diffusion --model_dir $data_dir
/home/jaewan/spine-diff/spine-venv/bin/python3 -m src.diffusion_score --model_dir $diffusion_model_dir