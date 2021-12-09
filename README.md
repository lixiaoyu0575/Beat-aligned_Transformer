# Beat-aligned Transformer

This reposity contains the code for the paper "BaT: Beat-aligned Transformer for Electrocardiogram Classification" on the [Physionet/CinC Challenge 2020 dataset](https://physionetchallenges.org/2020).

The dependencies are listed in the requirements.txt. 

To run this code, you need to download and organize the challenge data, and set corresponding paths in the config json file. Then, utilize preprare_segments.py to generate heartbeat segments and resample ratios, and run:
```
python main_beat_aligned.py -c train_beat_aligned_swin_transformer.json -d 0 -s 1
python main_beat_aligned.py -c train_swin_transformer.json -d 1 -s 1
``` 
c for the config json file path, d for the gpu device index, and s for the random seed.

