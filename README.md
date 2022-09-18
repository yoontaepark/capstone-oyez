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
project
| -- data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.

|    | -- web-scraping: web scarped case opinion texts from US courts (https://www.oyez.org/)
|    | -- interrater-reliability: location of .csvs to evaluate interrater reliability
|    | -- to_label: tmp location of the unlabelled data selected for the project
|    | -- labelled: place labelled data selected for the project here.
| 
| -- notebooks: store your Jupyter Notebooks here.
|    | -- yp2201: treat this workspace as your own.
                  feel free to add subfolders or files as you see fit.
                  work in a way that enables YOU to succeed.
|    | -- yc5206
|    | -- yk2678
|    | -- shared: place anything you want to share with the group, or the final version of your notebook here.
| 
| -- src: store your source code (e.g. py, sh) here. This follows the same structure as /notebooks
| -- reporting:
|    | -- assets: place images or table-data here for our final report.
|    | -- latex: place the latex/bib file here(optional submodule location)
```
This file structure is derived from [Cookiecutter Project Template](https://drivendata.github.io/cookiecutter-data-science/).
