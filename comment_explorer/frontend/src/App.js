import React from 'react';
import { Layout, Spin, Row, Col } from 'antd';
import './App.css';
import 'antd/dist/antd.css';
import AuthorSelection from './components/AuthorSelection';
import AuthorComments from './components/AuthorComments';
import Axios from 'axios';
import ArticleComments from './components/ArticleComments';


const { Content, Sider } = Layout;

class App extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      userID: null,
      comment_ids: null,
      comment_texts: null,
      loading: false,
      loadingArticles: false,
      article_comments: null,
      article_id: null,
    }
  }

  render() {
    return (
      <Layout>
        <Sider style={{ overflow: 'auto', height: '100vh', position: 'fixed', left: 0 }}>
          <AuthorSelection selectUser={this.selectUser} />
        </Sider>
        <Layout style={{ marginLeft: 200 }}>
          <Content style={{ margin: '24px 16px 0', overflow: 'initial' }}>
            <Row gutter={24}>
              <Col span={12}>
              {this.state.loading ? <Spin /> : <AuthorComments userID={this.state.userID} comment_texts={this.state.comment_texts} selectArticle={this.selectArticle} />}
              </Col>
              <Col span={12}>
              {this.state.loadingArticles ? <Spin /> : <ArticleComments userID={this.state.userID} articleID={this.state.article_id} comment_texts={this.state.article_comments} />}
              </Col>
            </Row>
            
          </Content>
        </Layout>
      </Layout>
    );
  }

  selectUser = async (userID) => {
    if(userID) {
      this.setState({loading: true});
      console.log(userID);
      const response = await Axios.get(`/user/${userID}`);
      const data = response.data.data.comment_ids;
      this.setState({ comment_ids: data });
      const comment_texts = await Promise.all(data.map(async commentID => {
        const response = await Axios.get(`/comment/${commentID}`);
        return response.data.data;
      }));
      this.setState({ comment_texts });
      this.setState({ userID });
      this.setState({loading: false});
    }
    
  }
  selectArticle = async (articleID) => {
    if(articleID) {
      this.setState({loadingArticles: true});
      const response = await Axios.get(`/article/${articleID}`);
      const data = response.data.data.comment_ids;
      this.setState({ comment_ids: data });
      const comment_texts = await Promise.all(data.map(async commentID => {
        const response = await Axios.get(`/comment/${commentID}`);
        return response.data.data;
      }));
      this.setState({ article_comments: comment_texts, article_id: articleID });
      this.setState({loadingArticles: false});
    }
  }
}

export default App;
