import React, { Component } from 'react';
import axios from 'axios';
import 'react-select/dist/react-select.css'
import 'react-virtualized-select/styles.css'
import VirtualizedSelect from 'react-virtualized-select'

class AuthorSelection extends Component {
    constructor() {
        super();
        this.state = { data: [], selectedOption: null };
    }

    async componentDidMount() {
        const response = await axios.get('/users');
        this.setState({ data: response.data.data.users })
    }

    handleChange = selectedOption => {
        this.setState({ selectedOption });
        this.props.selectUser(selectedOption.value)
    };

    render() {
        return (
            <React.Fragment>
                <p style={{ color: 'white'}}>Users:</p>
                <VirtualizedSelect
                    value={this.state.selectedOption}
                    onChange={this.handleChange}
                    options={this.state.data.map(user => { return { label: user, value: user } })}
                />
            </React.Fragment>

        )
    }
}

export default AuthorSelection;
