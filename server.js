const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const port = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// API Endpoint for Demand Prediction
app.post('/predict', (req, res) => {
    const { item_id, date } = req.body;

    if (!item_id || !date) {
        return res.status(400).json({ error: 'item_id and date are required' });
    }

    const pythonScriptPath = path.join(__dirname, '../ml-model/predict.py');

    // Spawn a Python process
    const pythonProcess = spawn('python', [pythonScriptPath, item_id, date]);

    let pythonOutput = '';
    let pythonError = '';

    pythonProcess.stdout.on('data', (data) => {
        pythonOutput += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        pythonError += data.toString();
    });

    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            console.error(`Python script exited with code ${code}`);
            console.error(`Python Error: ${pythonError}`);
            return res.status(500).json({ error: 'Prediction failed', details: pythonError });
        }

        try {
            const result = JSON.parse(pythonOutput);
            if (result.error) {
                return res.status(500).json({ error: result.error });
            }
            res.json(result);
        } catch (parseError) {
            console.error('Failed to parse Python script output:', pythonOutput);
            console.error('Parse Error:', parseError);
            res.status(500).json({ error: 'Failed to parse prediction result', details: pythonOutput });
        }
    });

    pythonProcess.on('error', (err) => {
        console.error('Failed to start Python subprocess:', err);
        res.status(500).json({ error: 'Failed to execute prediction script', details: err.message });
    });
});

// Basic route for testing
app.get('/', (req, res) => {
    res.send('Inventory Management Backend is running!');
});

// Start the server
app.listen(port, () => {
    console.log(`Backend server listening at http://localhost:${port}`);
});
