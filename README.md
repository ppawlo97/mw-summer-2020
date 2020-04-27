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
* [License](#license)



## About the Project

> TBD...



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



## License

Distributed under the MIT License. See `LICENSE` for more information.



[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/ppawlo97/si-summer-2020/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://pl.linkedin.com/in/piotr-paw%C5%82owski-64390917a
[python-shield]: https://img.shields.io/badge/python-3.7.7-blue?style=flat-square&logo=python
