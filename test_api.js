const axios = require('axios'); // We need to install axios

const testPredictEndpoint = async () => {
    const url = 'http://localhost:5000/predict';
    const payload = {
        item_id: 'ITEM_001',
        date: '2025-11-27'
    };

    try {
        const response = await axios.post(url, payload, {
            headers: {
                'Content-Type': 'application/json'
            }
        });
        console.log('Prediction successful:');
        console.log(JSON.stringify(response.data, null, 2));
    } catch (error) {
        console.error('Error during prediction:', error.response ? error.response.data : error.message);
    }
};

testPredictEndpoint();
