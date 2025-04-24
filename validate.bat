REM python validate.py --model_file "D:/GeoData/DLData/Waters/WaterTiles/out/unet_resnet34_best.pt" --img_dir "D:/GeoData/DLData/Waters/WaterTiles" --img_sz 256 --conf 0.5
  
REM python validate.py --model_file "D:/GeoData/DLData/Waters/WaterTiles/out/unetplusplus_resnet34_best.pt" --img_dir "D:/GeoData/DLData/Waters/WaterTiles" --img_sz 256 --conf 0.5
  
REM python validate.py --model_file "D:/GeoData/DLData/Waters/WaterTiles/out/linknet_resnet34_best.pt" --img_dir "D:/GeoData/DLData/Waters/WaterTiles" --img_sz 256 --conf 0.5
  
python validate.py --model_file "D:\dlwater\train_data\wat_nj_rgb\out\trained_models\unet_resnet50\unet_resnet50_best.pt" --img_dir "D:/dlwater/train_data/wat_nj_rgb/val/images" --img_sz 256 --conf 0.5

#python validate.py --model_file="D:\dlwater\train_data\wat_nj_nirg256\out\trained_models\unet_resnet50\unet_resnet50_best.pt" --img_dir="D:\dlwater\train_data\wat_nj_nirg256/val/images" --img_sz=256 --conf=0.5

 
REM python validate.py --model_file "D:\GeoData\DLData\Vehicle\Vehicle_seg\out\unet_resnet34_best.pt" --img_dir "D:\GeoData\DLData\Vehicle\vehicle_seg" --conf 0.5
  
REM python validate.py --model_file "D:/GeoData/DLData/saltern/10bands/out/mxsegnet_resnet34_best.pt" --img_dir "D:/GeoData/DLData/saltern/10bands" --conf 0.5 --img_sz 512 --plot True

