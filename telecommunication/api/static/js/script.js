document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        bundle_type: document.getElementById('bundle_type').value,
        operator: document.getElementById('operator').value,
        validity: parseFloat(document.getElementById('validity').value),
        regular_price: parseFloat(document.getElementById('regular_price').value),
        selling_price: parseFloat(document.getElementById('selling_price').value),
        internet: parseFloat(document.getElementById('internet').value),
        minutes: parseFloat(document.getElementById('minutes').value)
    };

    const resultContainer = document.getElementById('resultContainer');
    const predictionResult = document.getElementById('predictionResult');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Prediction failed');
        }

        const result = await response.json();
        
        if (result.status === 'success') {
            predictionResult.innerHTML = `
                <p class="success">Predicted Commission: ${result.predicted_commission.toFixed(2)}</p>
            `;
        } else {
            predictionResult.innerHTML = `
                <p class="error">Error: ${result.error || 'Unknown error occurred'}</p>
            `;
        }
    } catch (error) {
        console.error('Error:', error);
        predictionResult.innerHTML = `
            <p class="error">Error: ${error.message}</p>
        `;
    } finally {
        resultContainer.style.display = 'block';
    }
});

document.getElementById('clearForm').addEventListener('click', () => {
    document.getElementById('predictionForm').reset();
    document.getElementById('resultContainer').style.display = 'none';
});
