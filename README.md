# Model training

To run the script using accelerate use command:

```bash
accelerate launch script/main2.py  --data_dir <training_data_path> --output_dir <output_path> --num_epochs <number_of_epochs>
```

Without accelerate:

```bash
python script/main2.py --data_dir <training_data_path> --output_dir <output_path> --num_epochs <number_of_epochs>
```
# Model retraining

## no accelerate

```bash
python script/main2.py --data_dir "/sekhemet/scratch/kamkal/Augm/data_v3_prostate_32slices_34_plus_100/" --output_dir /home/kamkal/scratch/diffusion-ct/diff_mod_out/ct_400/data_134_1/ --num_epochs 3 --checkpoint_path /home/kamkal/scratch/diffusion-ct/diff_mod_out/ct_400/data_134_1/models/model_3999 --save_image_epochs 1 --save_model_epochs 1
```

## with accelerate

```bash
accelerate launch script/main2.py --data_dir "/sekhemet/scratch/kamkal/Augm/data_v3_prostate_32slices_34_plus_100/" --output_dir /home/kamkal/scratch/diffusion-ct/diff_mod_out/ct_400/data_134_1/ --num_epochs 3 --checkpoint_path /home/kamkal/scratch/diffusion-ct/diff_mod_out/ct_400/data_134_1/models/model_3999 --save_image_epochs 1 --save_model_epochs 1
```

# Inference

```bash
python script/generate.py --model_path /home/kamkal/scratch/diffusion-ct/diff_mod_out/ct_400/data_134_1/models/model_3499 --output_dir /home/kamkal/scratch/diffusion-ct/diff_model_gen/ct_400_data_134_1 --num_samples 1 --image_size 400
```


