# capstone-oyez: Automated judicial case briefing

### Authors:
Yoon Tae Park <yp2201@nyu.edu> <br>
Yoobin Cheong <yc5206@nyu.edu> <br>
Yeong Koh <yeong.koh@nyu.edu> <br>
Kannan Venkataramanan <kv942@nyu.edu> <br>
Sandeep Bhupatiraju <sbhupatiraju@worldbank.org> <br>
Daniel Li Chen <dchen9@worldbank.org> <br>
<br>


Description:
- Capstone project for New York University's Capstone Project and Presentation (DS-GA 1006, Fall 2022) mentored by World Bank
- Due: Dec 14th, 2022

<!-- Criteria [here](///) -->

#### Repository Structure
```
Project
├── LICENSE
├── README.md         
├── data
│   ├── raw: web scarped case opinion texts from US courts (https://www.oyez.org/)
│   ├── interim: intermediate data that has been preprocessed
│   └── final: final data sets that are used for further analysis
│
├── notebooks: store your Jupyter Notebooks here.
│   ├── yp2201: treat this workspace as your own.
│   │           feel free to add subfolders or files as you see fit.
│   │           work in a way that enables YOU to succeed
│   ├── yc5206 
│   ├── yk2678 
│   └── shared: place anything you want to share with the group, or the final version of your notebook here.
│
├── src: store your source code (e.g. py, sh) here. This follows the same structure as /notebooks
│
├── reporting: Generated analysis as HTML, PDF, LaTeX, etc
│   ├── interim: any intermediate report that has been created
│   ├── figure: graphics and figures needed for report
│   └── final: final report
│
├── references: data dictionaries, manuals, and all other explanatory materials.         
├── requirements.txt: the requirements file for reproducing the analysis environment(TBD)
└── setup.py: make this project pip installable with `pip install -e`

```
This file structure is derived from [Cookiecutter Project Template](https://drivendata.github.io/cookiecutter-data-science/).
