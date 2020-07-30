import React from 'react';
import { Card, Descriptions } from 'antd';

class ArticleComments extends React.Component {
    constructor(props) {
        super(props);
    }

    render() {
        if (this.props.articleID) {
            return (
                <div>
                    <h1>Article: {this.props.articleID}</h1>
                    {
                        this.props.comment_texts.map((comment, index) => 
                            <Card key={index} style={{ width: '100%', backgroundColor: this.props.userID === comment.author_id ? 'yellow': 'white'}}>
                                <span style={{ fontStyle: 'italic', fontSize: '12px'}}>{comment.timestamp}</span>
                                <span style={{ fontStyle: 'italic', fontSize: '12px', paddingLeft: '12px'}}>{comment.comment_id}</span>
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
            return <div>Select an article</div>
        }
    }
}

export default ArticleComments;
