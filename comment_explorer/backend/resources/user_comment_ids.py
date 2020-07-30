from flask import jsonify
from flask_restful import Resource


class UserCommentIds(Resource):
    def __init__(self, user_comments):
        self.user_comments = user_comments

    def get(self, user_id):
        user_comment_ids = self.user_comments.get_user_comments(user_id)
        return jsonify(**{ 'data': {
            'comment_ids': user_comment_ids
        }})
