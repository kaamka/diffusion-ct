# Model training

To run the script using accelerate use command:

```bash
accelerate launch script/main.py  --data_dir <training_data_path> --output_dir <output_path> --num_epochs <number_of_epochs>
```

Without accelerate:

```bash
python script/main.py --data_dir <training_data_path> --output_dir <output_path> --num_epochs <number_of_epochs>
```

