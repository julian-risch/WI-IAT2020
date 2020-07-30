from flask import Flask, request
from flask_restful import Api
from json import dumps
import fire
import logging

from explorer.text_collector import TextCollector
from explorer.user_comments import Users
from resources.article import Article
from resources.comments import Comments
from resources.user import Authors
from resources.user_comment_ids import UserCommentIds

app = Flask(__name__)
api = Api(app)

logging.basicConfig(level=logging.INFO)

def start(raw_comments_path, offset_path, selection_path, port=5002):
    text_collector = TextCollector(raw_comments_path, offset_path)
    users = Users(selection_path)

    api.add_resource(Authors, '/users', resource_class_kwargs={'user_comments': users})
    api.add_resource(UserCommentIds, '/user/<user_id>', resource_class_kwargs={'user_comments': users})
    api.add_resource(Article, '/article/<article_id>', resource_class_kwargs={'comment_meta_collector': users})
    api.add_resource(Comments, '/comment/<comment_id>',  resource_class_kwargs={'text_collector': text_collector, 'comment_meta_collector': users})

    app.run(port=port)


if __name__ == '__main__':
    fire.Fire(start)
