# Detecting Backdoor Attacks in Black-Box Neural Networks through Hardware Performance Counters

---

## 📑 Overview
The repository presents a defense framework against backdoor attacks in Deep Neural Networks (DNNs), which is operational even when the network internals are unknown (i.e., black-box scenario). It utilizes Hardware Performance Counters (HPCs) to monitor the microarchitectural behavior of the DNN during inference. By applying Gaussian Mixture Models to the collected HPC data, the framework detects unusual patterns that indicate whether an input is legitimate or maliciously altered (backdoor).


## 🖥️  System Requirements
- **Operating System**: Linux (Tested on Ubuntu 18.04.6 LTS).
- **Processor**: Intel (Tested on Intel i7-9700).
- **Python Version**: Python 3.10.9 (Confirmed compatibility).
- **CUDA Toolkit**: Optional for GPU acceleration (Tested with version 11.5, V11.5.119).

## 🛠️ Installation Guide
- Ensure the availability of the `perf` tool in the system.

- Set up a dedicated Python virtual environment and install required dependencies.
```
python -m venv <environment_name>
source <environment_name>/bin/activate
pip install -r requirements.txt
```

## 🚀 Step-by-Step Execution Guide
1. **Model Training with Backdoor Integration**:
Train a ResNet18 model on the CIFAR10 dataset, embedding a subtle cross-pattern backdoor trigger in the bottom right corner of images. Use `--target=<target_class>` to specify the target class for the backdoor attack. The resulting backdoored model is saved as `best_model_resnet18.pth`.
```
python backdoor_training.py --target=<target_class>
```

2. **Image Preparation**:
Generate two sets of CIFAR-10 images: unaltered (saved in `benign_images`) and modified with the backdoor trigger (saved in `backdoor_images`).
```
python save_images.py
```

3. **Performance Counter Data Collection**:
Collect microarchitectural performance counter data using the `perf` tool while the model processes both image sets. This requires superuser access. Data is logged as `perf_benign.log` and `perf_backdoor.log`.
```
./profile_script.sh <environment_name>
```

4. **Data Processing**:
Convert the logged performance counter data into a structured JSON format, resulting in `hpc_data_benign.json` and `hpc_data_backdoor.json`.
```
python process_hpc_log.py
```

5. **Anomaly Detection and Model Evaluation**:
Construct Gaussian Mixture Models from benign performance counter statistics, then apply these model to detect anomalies using the backdoor performance counter statistics. The framework's detection capability is quantified using `accuracy` and `F1-score` metrics.
```
python classify_backdoor.py
```

---

## 📚 Cite Us
If you find our work interesting and use it in your research, please cite our paper describing:

Manaar Alam, Yue Wang, and Michail Maniatakos, "_Detecting Backdoor Attacks in Black-Box Neural Networks through Hardware Performance Counters_", DATE 2024.

### BibTex Citation
```
@inproceedings{DBLP:conf/date/AlamWM24,
  author       = {Manaar Alam and
                  Yue Wang and
                  Michail Maniatakos},
  title        = {{Detecting Backdoor Attacks in Black-Box Neural Networks through Hardware Performance Counters}},
  booktitle    = {Design, Automation {\&} Test in Europe Conference {\&} Exhibition,
                  {DATE} 2024, Valencia, Spain, March 25-27, 2024},
  publisher    = {{IEEE}},
  year         = {2024}
}
```

---

## 📩 Contact Us
For more information or help with the setup, please contact Manaar Alam at: alam.manaar@nyu.edu