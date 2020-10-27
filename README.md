# WI-IAT2020

This repository contains code that is associated with our paper accepted for publication at the IEEE/WIC/ACM International Joint Conference On Web Intelligence And Intelligent Agent Technology (WI-IAT '20). The paper is titled *HyCoNN: Hybrid Cooperative Neural Networks for Personalized News Discussion Recommendation*. It is based on a Master's thesis by Victor Künstler, co-supervised by Julian Risch and Ralf Krestel.

# Citation
If you use our work, please cite our upcoming paper [**HyCoNN: Hybrid Cooperative Neural Networks for Personalized News Discussion Recommendation**](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/people/risch/risch2020hyconn.pdf) accepted for publication at [WI-IAT'20](http://wi2020.vcrab.com.au/) as follows:

    @inproceedings{risch2020hyconn,
    title = {HyCoNN: Hybrid Cooperative Neural Networks for Personalized News Discussion Recommendation},
    author = {Risch, Julian and K{\"u}nstler, Victor and Krestel, Ralf},
    booktitle = {Proceedings of the International Joint Conferences on Web Intelligence and Intelligent Agent Technologies (WI-IAT)},
    year = {2020}
    }

There are this `README.md`, a `requirements.txt` and seven subdirectories:
* `baselines_evaluation` contains code for all experiments, including the rank fusion ensemble
* `category_crawler` is a small scraper to retrieve keywords (categories) of news articles
* `comment_explorer` is a comment exploration tool only used for debugging purposes and not described in the paper
* `data_exploration` contains code for exploratory data analysis, including association rule mining
* `models` contains the implementation of the neural network models and the community graph
* `preprocessing` contains the preprocessing methods, such as the selection of negative samples or the tf-idf model
* `torchtrainer-master` is used for callbacks during the training process: [torchtrainer](https://pypi.org/project/torchtrainer/)
