source /data/kechen/miniconda3/etc/profile.d/conda.sh
conda activate musicldm_env
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --texts sample_file.txt --seed 78949
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --texts sample_file.txt --seed 467
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --texts sample_file.txt --seed 878
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --texts sample_file.txt --seed 908
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --texts sample_file.txt --seed 2
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --texts sample_file.txt --seed 72467
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --texts sample_file.txt --seed 4397
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --texts sample_file.txt --seed 95678
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --texts sample_file.txt --seed 379
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --texts sample_file.txt --seed 9489
