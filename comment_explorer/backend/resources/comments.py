from flask import jsonify
from flask_restful import Resource


class Comments(Resource):
    def __init__(self, text_collector, comment_meta_collector):
        self.text_collector = text_collector
        self.comment_meta = comment_meta_collector

    def get(self, comment_id):
        all_user_ids = self.text_collector.get_comment_text(comment_id)
        parent_comment_id, timestamp, upvote, article_id, author_id, comment_id = self.comment_meta.get_comment_info(comment_id)
        print(self.comment_meta.get_comment_info(comment_id))
        result = {'data':
                      {
                          'comment_text': all_user_ids,
                          'parent_comment_id': parent_comment_id,
                          'timestamp': timestamp,
                          'upvote': upvote,
                          'article_id': article_id,
                          'author_id': author_id,
                          'comment_id': comment_id
                      }
        }
        return jsonify(**result)
