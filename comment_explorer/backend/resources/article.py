from flask import jsonify
from flask_restful import Resource


class Article(Resource):
    def __init__(self, comment_meta_collector):
        self.comment_meta = comment_meta_collector

    def get(self, article_id):
        comment_ids = self.comment_meta.get_article_comments(article_id)
        result = {'data':
                      {
                          'comment_ids': comment_ids
                      }
        }
        return jsonify(**result)
