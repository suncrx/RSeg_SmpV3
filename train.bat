REM usage:
python train.py --data ./data/waters_nj.yaml --checkpoint_file D:/dlwater/train_data/wat_nj_rgb/out/trained_models/unet_resnet50/unet_resnet50.ckp --arct unet --encoder resnet50 --img_sz 256 --epochs 2 --batch_size 8 --lr 0.0001 --momentum 0.9 --loss dice --aug 1 --resume 0 --save_period 5

python train.py --data ./data/waters_nj_12bands.yaml --checkpoint_file '' --arct unet --encoder resnet50 --img_sz 256 --epochs 2 --batch_size 8 --lr 0.0001 --momentum 0.9 --loss dice --resume 0 --save_period 1 

REM python train.py --data ./data/vehicles.yaml --checkpoint_file D:/GeoData/DLData/vehicle_seg/out/unet.ckp --arct mxsegnet --encoder resnet34 --img_sz 256 --epochs 20 --batch_size 4 --lr 0.0001 --momentum 0.9 --loss dice --checkpoint True --sub_size 1

REM python train.py --data ./data/saltern.yaml --checkpoint_file D:/GeoData/DLData/saltern/10bands/out/mxsegnet_resnet34.ckp --arct mxsegnet --encoder resnet34 --img_sz 512 --epochs 5 --batch_size 4 --lr 0.0001 --momentum 0.9 --loss dice --checkpoint True --sub_size 1