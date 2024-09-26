<!DOCTYPE html>
<html>
<head>

</head>
<body>
<h2>Installation guide</h2>  
<p>This is how the framework can be downloaded and configured to run the experiments</p>
<h5>Using Docker</h5>
<ul>
  <li>Download and install Docker from <a href="https://www.docker.com/">https://www.docker.com/</a></li>
  <li>Run the following command to "pull Docker Image" from Docker Hub: <code>docker pull shefai/intent_aware_recomm_systems</code>
  <li>Clone the GitHub repository by using the link: <code>https://github.com/Faisalse/IDS4NR.git</code>
  <li>Move into the <b>IDS4NR</b> directory</li>
  
  <li>Run the command to mount the current directory <i>IDS4NR</i> to the docker container named as <i>IDS4NR_container</i>: <code>docker run --name IDS4NR_container  -it -v "$(pwd):/IDS4NR" -it shefai/intent_aware_recomm_systems</code>. If you have the support of CUDA-capable GPUs then run the following command to attach GPUs with the container: <code>docker run --name IDS4NR_container  -it --gpus all -v "$(pwd):/IDS4NR" -it shefai/intent_aware_recomm_systems</code></li> 
<li>If you are already inside the runing container then run the command to navigate to the mounted directory <i>IDS4NR</i>: <code>cd /IDS4NR</code> otherwise starts the "IDS4NR_container"</li>
<li>Finally, follow the given instructions to run the experiments for each model </li>
</ul>  
<h5>Using Anaconda</h5>
  <ul>
    <li>Download Anaconda from <a href="https://www.anaconda.com/">https://www.anaconda.com/</a> and install it</li>
    <li>Clone the GitHub repository by using this link: <code>https://github.com/Faisalse/IDS4NR.git</code></li>
    <li>Open the Anaconda command prompt</li>
    <li>Move into the <b>IDS4NR</b> directory</li>
    <li>Run this command to create virtual environment: <code>conda create --name IDS4NR python=3.8</code></li>
    <li>Run this command to activate the virtual environment: <code>conda activate IDS4NR</code></li>
    <li>Run this command to install the required libraries for CPU: <code>pip install -r requirements_cpu.txt</code>. However, if you have support of CUDA-capable GPUs, 
        then run this command to install the required libraries to run the experiments on GPU: <code>pip install -r requirements_gpu.txt</code></li>
  </ul>
</p>

<h5>IDS4NR and baseline models</h5>
<ul>
<li>Run this command to reproduce the experiments for the IDS4NR_NCF and baseline models on the MovieLens dataset: <code>python run_experiments_IDS4NR_baselines_algorithms.py --dataset MovieLens --model NCF</code>  </li>

<li>Run this command to reproduce the experiments for the IDS4NR_NCF and baseline models on the Beauty dataset: <code>python run_experiments_IDS4NR_baselines_algorithms.py --dataset Beauty --model NCF</code>  </li>

<li>Run this command to reproduce the experiments for the IDS4NR_NCF and baseline models on the Music dataset: <code>python run_experiments_IDS4NR_baselines_algorithms.py --dataset Music --model NCF</code>  </li>

<li>Run this command to reproduce the experiments for the IDS4NR_LFM and baseline models on the MovieLens dataset: <code>python run_experiments_IDS4NR_baselines_algorithms.py --dataset MovieLens --model LFM</code>  </li>

<li>Run this command to reproduce the experiments for the IDS4NR_LFM and baseline models on the Beauty dataset: <code>python run_experiments_IDS4NR_baselines_algorithms.py --dataset Beauty --model LFM</code>  </li>

<li>Run this command to reproduce the experiments for the IDS4NR_LFM and baseline models on the Music dataset: <code>python run_experiments_IDS4NR_baselines_algorithms.py --dataset Music --model LFM</code>  </li>
</ul>


</body>
</html>  

