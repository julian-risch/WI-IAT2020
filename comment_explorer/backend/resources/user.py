from flask import jsonify
from flask_restful import Resource


class Authors(Resource):
    def __init__(self, user_comments):
        self.user_comments = user_comments

    def get(self):
        all_user_ids = self.user_comments.get_all_users()
        result = { 'data' : { 'users': all_user_ids}}
        return jsonify(**result)
