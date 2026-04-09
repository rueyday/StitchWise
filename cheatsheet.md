# 🚀 Great Lakes + YOLOv8 Cheatsheet

## 1. Transferring Files (Local Machine -> Cluster)
*Run this from your laptop terminal to send files to the cluster.*

**Zip your project folder:**
```bash
tar -czf stitchwise.tar.gz -C ~/kane_stuff StitchWise
```
**Send it via Rsync (Pro-tip: faster than scp and resumes if disconnected):**
```bash
rsync -aP stitchwise.tar.gz kaichi@greatlakes-xfer.arc-ts.umich.edu:
```

---

## 2. Cluster Setup & Extraction
*Run this on the Great Lakes terminal.*

**Log in:**
```bash
ssh kaichi@greatlakes.arc-ts.umich.edu
```
*(Optional but recommended)* **Move to Scratch for faster unzipping/training:**
```bash
# Move zip to your high-performance scratch space
mv ~/stitchwise.tar.gz /scratch/eecs504s001w26_class/kaichi/
cd /scratch/eecs504s001w26_class/kaichi/
```
**Unzip the project:**
```bash
tar -xzf stitchwise.tar.gz
cd StitchWise
```

---

## 3. Environment Setup (First Time Only)
*Set up Python and YOLOv8 on the cluster.*

```bash
# Load Anaconda module
module load python3.10-anaconda

# Create a virtual environment
python -m venv venv

# Activate it
source venv/bin/activate

# Install YOLOv8 requirements
pip install --upgrade pip
pip install ultralytics opencv-python-headless pyyaml
```
*(Note: Every time you log back in, you must run `module load python3.10-anaconda` and `source venv/bin/activate` before training).*

---

## 4. The Slurm Job Script (`train_job.sh`)
*This is the ticket you submit to the scheduler. Create it using `nano train_job.sh` (or directly in VS Code).*

```bash
#!/bin/bash
#SBATCH --job-name=stitchwise_yolo
#SBATCH --account=eecs504s001w26_class
#SBATCH --partition=gpu
#SBATCH --time=08:00:00       # Max time allowed for this account
#SBATCH --nodes=1
#SBATCH --gpus=1              # Max GPUs allowed for this account
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x-%j.log

# Load modules and activate environment
module purge
module load python3.10-anaconda

# Make sure you are in the right folder!
cd /scratch/eecs504s001w26_class/kaichi/StitchWise
source venv/bin/activate

# Run the Python script
python scripts/train_production.py
```

---

## 5. Job Management Commands
*How to submit, monitor, and kill your training jobs.*

**Submit your job to the queue:**
```bash
sbatch train_job.sh
```
**Check your job status (Pending = PD, Running = R):**
```bash
sq kaichi
```
**Cancel a job (if you made a mistake):**
```bash
scancel <JOB_ID>
```
**Watch the live YOLO training output:**
```bash
# Replace with your actual log file name
tail -f stitchwise_yolo-1234567.log
```
*(Press `Ctrl+C` to exit the live log view).*

---

## 6. VS Code Remote SSH
*Skip the terminal text editors and code directly on the cluster.*

1. Open VS Code command palette (`F1` or `Ctrl+Shift+P`).
2. Select **Remote-SSH: Connect to Host...**
3. Select `kaichi@greatlakes.arc-ts.umich.edu`.
4. Enter password, then type `1` for Duo push.
5. Click **Open Folder** and navigate to your `StitchWise` directory (e.g., `/scratch/eecs504s001w26_class/kaichi/StitchWise`).

***

You've got the full pipeline locked down now. Let me know how that 100-epoch production run goes!