[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
![Python][python-shield]



<br />
<p align="center">
  <h3 align="center">Final project for Multiagent Learning course at SGH.</h3>

  <p align="center">
    Below find the information on the setup, reference links and creator notes.
    <br />
    <br />
  </p>
</p>



## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Report](#report)
* [Presentation](#presentation)
* [References](#references)
* [License](#license)



## About the Project

The purpose of this project was to implement a Deep Q Learning Agent, analize it's performance on [PLE's FlappyBird](https://pygame-learning-environment.readthedocs.io/en/latest/user/games/flappybird.html) environment given various hyperparameters setups and present the final work to the course group. 

Algorithm implementation with brief descriptions in the docstrings and appropriate references can be found in the `agent` folder. In order to run the model follow the instuctions from [Getting Started](#getting-started) and [Usage](#usage) sections.

Results of the analysis alongside with the theory behind the Deep Q Learning were outlined in a separate report file. See [Report](#report) section for more details.



## Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

1. Make sure that you have [Anaconda](https://www.anaconda.com/) installed.

### Installation
 
1. Clone the repository.
```sh
git clone https://github.com/ppawlo97/mw-summer-2020.git
```
2. Create separate virtual environment for the project.
```sh
conda create --name=mw_dqn python=3.7
```
3. Switch to the created environment.
```sh
conda activate mw_dqn
```
4. Install the dependencies.
```sh
pip install -r requirements.txt
```
5. Since PLE cannot be downloaded via pip, it should be installed directly through [PLE's GitHub](https://github.com/ntasfi/PyGame-Learning-Environment). Following PLE's documentation:
```sh
git clone https://github.com/ntasfi/PyGame-Learning-Environment
```
And then:
```sh
pip install -e .
```



## Usage

1. From the project's root directory, checkout to the `master` branch, if you are not already on it.
```sh
git checkout master
```
2. Remember to always switch to the right virtual environment.
```sh
conda activate mw_dqn
```
3. In order to inspect available parameters and set defaults:
```sh
python run_flappybird.py --help
```
4. Run the agent with the selected parameters - for example custom number of episodes.
```sh
python run_flappybird.py --episodes=300
```



## Report

The report in the form of Jupyter Notebook with dependent data files was put inside the `report` folder. Since the conducted analysis demanded specific libraries, additional requirements have been listed inside `Report_MW.ipynb` in order to keep the implementation and report dependecies separate.

To reproduce the data used throughout the hyperparameter analysis, run:
```sh
python run_param_search.py
```
from the project's main directory.

**Please note:**
Even though the code and the docstrings have been written in English in order to stay compliant with the universally accepted standards, the report with mild exceptions has been written in Polish, since the entire course was conducted in Polish.    



## Presentation

> TBD...



## References

[1]  Mnih, V., Kavukcuoglu, K., Silver, D. et al. **Human-level control through deep reinforcement learning.** Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236



## License

Distributed under the MIT License. See `LICENSE` for more information.



[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/ppawlo97/si-summer-2020/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://pl.linkedin.com/in/piotr-paw%C5%82owski-64390917a
[python-shield]: https://img.shields.io/badge/python-3.7.7-blue?style=flat-square&logo=python
