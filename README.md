## Official Repo for ICLR2026: Generalizing Linear Autoencoder Recommenders with Decoupled Expected Quadratic Loss
## For use put data parallel to this folder.

## dataset could be downloaded from:
https://drive.google.com/file/d/14UVzyJ74s85NPJCHUO3d5g0i7LWpVuDB/view?usp=sharing
Before use, please put data in to corresponding dirctory according to config file under /config

## Example usage：
```
python run_parallel.py --config_path "./config/ML20M/strong_generalization/EDLAE_b_geq_0_diag0_L2const.yaml" --gpu 0 --isValidationPhase False --key 98765  --note ML20M
```



