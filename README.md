# **25fall-ee641-group9-final**
Group members: Vivin Thiyagarajan , Yi Fan

Setup environment

(1) Create your environment
conda create -n ee641G9 python=3.11

(2) Activate this environment
conda activate ee641G9

(3) Install requirements
pip install -r requirements.txt

(4) Request model access to llama3-8b through hugging face
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct 

(5)Weight & bias client log in(for visualization)
Rember to modify wandb's ENITY and PROJECT to your project group

# **Experiment**

Run under the root directory:
python -m experiment.main

to access llama3-8b, log in to your hugging face :
huggingface-cli login
