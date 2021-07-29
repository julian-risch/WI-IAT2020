# WI-IAT2020

This repository contains code that is associated with our paper accepted for publication at the IEEE/WIC/ACM International Joint Conference On Web Intelligence And Intelligent Agent Technology (WI-IAT '20). The paper is titled *HyCoNN: Hybrid Cooperative Neural Networks for Personalized News Discussion Recommendation*. It is based on a Master's thesis by Victor Künstler, co-supervised by Julian Risch and Ralf Krestel. A video presentation is available on [YouTube](https://youtu.be/F3w81B1DQw4).

# Citation
If you use our work, please cite our upcoming paper [**HyCoNN: Hybrid Cooperative Neural Networks for Personalized News Discussion Recommendation**](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/people/risch/risch2020hyconn.pdf) accepted for publication at [WI-IAT'20](http://wi2020.vcrab.com.au/) as follows:

    @inproceedings{risch2020hyconn,
    title = {HyCoNN: Hybrid Cooperative Neural Networks for Personalized News Discussion Recommendation},
    author = {Risch, Julian and K{\"u}nstler, Victor and Krestel, Ralf},
    booktitle = {Proceedings of the International Joint Conferences on Web Intelligence and Intelligent Agent Technologies (WI-IAT)},
    pages = {41--48},
    year = {2020},
    publisher = {IEEE Computer Society},
    doi = {10.1109/WIIAT50758.2020.00011},
    url = {https://doi.ieeecomputersociety.org/10.1109/WIIAT50758.2020.00011},
    }

There are this `README.md`, a `requirements.txt` and seven subdirectories:
* `baselines_evaluation` contains code for all experiments, including the rank fusion ensemble
* `category_crawler` is a small scraper to retrieve keywords (categories) of news articles
* `comment_explorer` is a comment exploration tool only used for debugging purposes and not described in the paper
* `data_exploration` contains code for exploratory data analysis, including association rule mining
* `models` contains the implementation of the neural network models and the community graph
* `preprocessing` contains the preprocessing methods, such as the selection of negative samples or the tf-idf model
* `torchtrainer-master` is used for callbacks during the training process: [torchtrainer](https://pypi.org/project/torchtrainer/)

# Dataset
We provide the comment ids of the training, validation and test datasets. There are positive and negative samples describing the state of discussions (at a particular point in time) where a particular user did or did not comment. 

Example row from one of the csv files:
```author_id,article_id,max_timestamp,comment_ids```
```44041,72032,2017-06-12 19:08:29+00:00,"[100252639, 100239810, 100224652, 100180729]"``` 

```author_id``` identifies the user, ```article_id``` identifies the news article discussion, ```max_timestamp``` is the point in time described by the row,
```comment_ids``` is the list of comment ids posted in the discussion until the point in time given by ```max_timestamp```.

The csv file contains either only positive or only negative samples. The example is from the file with negative samples. Therefore, the row describes a situation where the user with id ```author_id``` did not comment on the discussion on article ```article_id``` with the comments ```comment_ids```.  The discussion where the user did comment is in the file with the positive samples.

For easier processing, the files are split into partitions, e.g., ```partition-0_val.csv```, ```partition-2_val.csv```,  ```partition-3_val.csv```, ...

The zipped files can be downloaded here (1.5GB):
* [The Guardian training file](https://owncloud.hpi.de/s/Sm6CgQtltP0OjaC)
* [The Guardian validation file](https://owncloud.hpi.de/s/IqxKo0o4HY3YSfZ)
* [The Guardian test file](https://owncloud.hpi.de/s/Fr6Jfw7PnohflhM)
* [Daily Mail training file](https://owncloud.hpi.de/s/PsPKNxy08IAJjiW)
* [Daily Mail validation file](https://owncloud.hpi.de/s/ZcR69D5IqBziT54)
* [Daily Mail test file](https://owncloud.hpi.de/s/oaay9KnKBNpPBvb)
