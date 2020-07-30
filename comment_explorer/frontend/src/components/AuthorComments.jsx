import React from 'react';
import { Card, Descriptions } from 'antd';

class AuthorComments extends React.Component {
    constructor(props) {
        super(props);
    }

    render() {
        if (this.props.userID) {
            return (
                <div>
                    <h1>Author: {this.props.userID}</h1>
                    {
                        this.props.comment_texts.map((comment, index) => 
                            <Card key={index} style={{ width: '100%' }} onClick={() => this.props.selectArticle(comment.article_id)}>
                                <span style={{ fontStyle: 'italic', 'fontSize': '12px'}}>{comment.timestamp}</span>
                                <p>{comment.comment_text}</p>
                                <Descriptions size='small' column={3}>
                                    <Descriptions.Item label="Parent" style={{ paddingBottom: 0}}>{comment.parent_comment_id}</Descriptions.Item>
                                    <Descriptions.Item label="Upvotes" style={{ paddingBottom: 0}}>{comment.upvote}</Descriptions.Item>
                                    <Descriptions.Item label="Article" style={{ paddingBottom: 0}}>{comment.article_id}</Descriptions.Item>
                                </Descriptions>
                            </Card>
                        )}
                </div>
            );
        } else {
            return <div>Select a user</div>
        }
    }
}

export default AuthorComments;
