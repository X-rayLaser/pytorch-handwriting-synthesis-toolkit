import React from 'react';
import Modal from 'react-bootstrap/Modal';
import Form from 'react-bootstrap/Form';


export default class PrimingModal extends React.Component {
    constructor(props) {
        super(props);
        this.state = {};
    }
    render() {
        return (
            <Modal show={show} onHide={handleClose}>
                <Modal.Header closeButton>
                    <Modal.Title>Modal heading</Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    <p>Please, enter any text below:</p>
                    <Form.Control type="text" className="mb-2" placeholder="Enter text to generate a handwriting for" 
                        value={this.state.text} onChange={this.handleChange}
                    />
                    <p>Please, write a corresponding handwriting for a text you entered by drawing on the canvas:</p>
                    <canvas width="2000" height="500" style={{width: '100%'}}></canvas>
                </Modal.Body>
                <Modal.Footer>
                    <Button variant="secondary" onClick={handleClose}>
                        Close
                    </Button>
                    <Button variant="primary" onClick={handleClose}>
                        Save Changes
                    </Button>
                </Modal.Footer>
            </Modal>
        );
    }
}