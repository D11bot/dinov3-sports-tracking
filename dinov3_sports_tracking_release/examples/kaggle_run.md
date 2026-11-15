# How I run the project on Kaggle

I use Kaggle for GPU inference and keep the SoccerNet images in my Kaggle working folder. I upload the Python project as a Kaggle Input. Kaggle may unpack it into a folder automatically, so I first check the location of `run_experiment.py`.

I copy the project into `/kaggle/working/` because this folder lets me create output files. I keep my Hugging Face token in Kaggle Secrets as `HF_TOKEN`.

```python
from pathlib import Path
import shutil
import os
from kaggle_secrets import UserSecretsClient

roots = list(Path('/kaggle/input').rglob('run_experiment.py'))
assert roots, 'I could not find run_experiment.py in Kaggle Input'
project_dir = Path('/kaggle/working/dinov3_sports_tracking_release')
shutil.copytree(roots[0].parent, project_dir, dirs_exist_ok=True)
os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
print(project_dir)
```

I run the tests first:

```python
%cd /kaggle/working/dinov3_sports_tracking_release
!python -m pytest -q tests
```

I already extracted the first 120 frames of SNMOT-061 in my Kaggle session. This is the command I use for the full experiments:

```python
!python run_experiment.py \
    --sequence-dir /kaggle/working/SNMOT-061 \
    --frames 120 \
    --exclude-ids 1 \
    --cache /kaggle/working/cache/SNMOT-061.npz \
    --output /kaggle/working/outputs/SNMOT-061 \
    --experiment all
```

For SNMOT-060, I use `--exclude-ids 18` after inspecting the ball track in that specific sequence. For another sequence, I inspect the small objects and decide which tracks belong in my player tracking experiments.

I keep the same feature cache path when I rerun an experiment. The script reuses the saved embeddings for unchanged crops and extracts new embeddings when I add box jitter or false positives.

I compare the new CSV files with the results saved in `results/reference/` to check reproducibility. I also record my model, sequence, random seed and tracking settings with each run.
