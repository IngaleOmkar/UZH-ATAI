# Advanced Topics in AI (ATAI) - UZH

Welcome to the Advanced Topics in AI (ATAI) course at the University of Zurich (UZH). This repository contains the project files for the course.

## Entrypoint

To get started, run the `movie_bot.py` file. This script serves as the main entry point for the project.

```bash
python movie_bot.py
```

Do remember to change the userid and password to your assigned credentials.

## Data

The data required for this project is available at the following link:
[Google Drive - ATAI Data](https://drive.google.com/drive/folders/1p7acQXCy4h8m9tdeFT6SpEZK4mxdlfiC?usp=sharing)

The data for this bot is curated from multiple sources including kaggle, IMDB and others. This repository also includes data that was obtained after preprocessing the data given as part of this project. 

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

## Contact

For any questions or issues, please contact the course instructors.
