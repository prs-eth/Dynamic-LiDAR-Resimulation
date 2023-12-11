# Preprocessing script for WaymoDynamic Dataset 
### We provide a preprocess script for the waymo v2 perception dataset for dynamic neural lidar simulations
To obtain the dataset, you need to first
### 1. Download the v2 perception dataset from [waymo open dataset](https://waymo.com/intl/en_us/open/download/)
make sure it has following structure, if you have one context with the name of `10061305430875486848_1080_000_1100_000`
```
waymo
└── training
    ├── lidar
    │   └── 10061305430875486848_1080_000_1100_000.parquet
    ├── lidar_box
    │   └── 10061305430875486848_1080_000_1100_000.parquet
    ├── lidar_calibration
    │   └── 10061305430875486848_1080_000_1100_000.parquet
    ├── lidar_pose
    │   └── 10061305430875486848_1080_000_1100_000.parquet
    └── vehicle_pose
        └── 10061305430875486848_1080_000_1100_000.parquet
```
you can download as many contexts as you want, our preprocess handle one context at a time

### 2. install requirements
We use python 3.10.10, python version under 3.9 would not work afaik

```
conda create -n waymov2 python=3.10.10
conda activate waymov2
python3 -m pip install gcsfs waymo-open-dataset-tf-2-11-0==1.5.1
pip install torch==1.13.1 torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu117
pip install open3d==0.16.0
pip install numba

```

### 3. launch the preprocess script
specify the `context_name`  as well as the `dataset_dir` in preprocess_waymo.py, e.g.
```
dataset_dir = './waymo/training' ##where you download in the first step

context_name = '10061305430875486848_1080_000_1100_000'
```
Then launch the script
```
python preprocess_waymo.py
```
It will generates a folder called processed_data_dynamic in your PWD, which as following structure:
```
processed_data_dynamic
└── 10061305430875486848_1080_000_1100_000
    ├── normals.npy
    ├── object_ids_per_frame.npy
    ├── objects_id_2_anchors.npy
    ├── objects_id_2_corners.npy
    ├── objects_id_2_dynamic_flag.npy
    ├── objects_id_2_frameidx.npy
    ├── objects_id_2_tsfm.npy
    ├── objects_id_types_per_frame.npy
    ├── range_images1.npy
    ├── range_images2.npy
    ├── ray_dirs.npy
    ├── ray_object_indices.npy
    ├── ray_origins.npy
    └── valid_normal_flags.npy
```

## Preprocessed data
We preprocessed [5 scenes](https://mega.nz/file/RvcUxKJT#rYS7MUkkEgBYDbjZf4KXpn0YFfEQGMF2XOk3Q6gPQoI) in advance if you would like to skip running the script on your end.



        
