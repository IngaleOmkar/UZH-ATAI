# Advanced Topics in AI (ATAI) - UZH

This is the repository for our bot, `sharp-cloud`, which helps users answer questions related to movies and TV shows for the ATAI course at UZH. 

If you wish to see the progress of the project for each milestone, checkout different branches that we have created after each evaluation event. 

## Get Started

To get started, run the `movie_bot.py` file. This script serves as the main entry point for the project.

```bash
python movie_bot.py
```

You can also set up the environment using the `requirements.txt` file. An example using `conda` is:

```bash
conda create -n "your_env_name" python=3.10
conda actiavte your_bot_name
cd path/to/your/cloned/repository/UZH-ATAI
pip install -r requirements.txt
```

We have also included a script to help you test this out locally. This can be found in `local_bot.py`. This will allow you to run the bot locally without having to log into speakeasy everytime for testing. 

```bash
python local_bot.py
```

Do remember to change the userid and password to your assigned credentials.

## Troubleshooting

We have encountered some errors during our testing. For example, NER using BERT only works well with teansformers library version 4.45.2. There may be other examples/errors that may pop up from time to time. If you do find solutions to these problems, please create a pull request with the updated fix and we'll merge it whenever possible. 

## Data

The data required for this project is available at the following link:
[Google Drive - ATAI Data](https://drive.google.com/drive/folders/1p7acQXCy4h8m9tdeFT6SpEZK4mxdlfiC?usp=sharing)

The data for this bot is curated from multiple sources including kaggle, IMDB and others. This link also includes data obtained from processing the given data. 

## Features

This project includes functionalities to answer questions related to:
- Factual information: Uses triplets dictionary to answer questions. 
- Embedding-based queries: Users TransE logic to find answers using provided embeddings. 
- Recommendations: Uses embeddings to find user preferences and returns entities closest to those preferences.  
- Multimedia content: Uses IMDB datasets to get ids. NOTE: Only strings are returned as part of the response. Speakeasy renders the image for you. 
- Crowdsourcing: Uses preprocessed dataset and an updated graph. All related data can be found in the drive link.

Feel free to explore the code and contribute to the project!

## License

This project is licensed under the MIT License.

