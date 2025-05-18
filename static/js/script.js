// Updated js code
console.log('script.js: Initializing...');

let currentLanguage = 'en';
let translations = {};

async function loadTranslations() {
    console.log('Loading translations...');
    try {
        const response = await fetch('/static/translations.json');
        translations = await response.json();
        console.log('Translations loaded:', Object.keys(translations));
        updateUIText();
    } catch (error) {
        console.error('Error loading translations:', error);
        showToast('Failed to load translations - using default English');
        translations = { 
            en: {
                appTitle: 'AgriBot - Smart Farming Assistant',
                soilOption: 'Soil & Pest Detection',
                cropOption: 'Crop & Irrigation Management',
                aidOption: 'Government Aids',
                soilTitle: 'Soil & Pest Detection',
                soilImageLabel: 'Upload Soil Image',
                soilButton: 'Analyze Soil',
                cropTitle: 'Crop & Irrigation Management',
                nitrogenLabel: 'Nitrogen (N) Level',
                phosphorusLabel: 'Phosphorus (P) Level',
                potassiumLabel: 'Potassium (K) Level',
                tempLabel: 'Temperature (°C)',
                humidityLabel: 'Humidity (%)',
                phLabel: 'Soil pH',
                rainfallLabel: 'Rainfall (mm)',
                soilTypeLabel: 'Soil Type',
                cropButton: 'Recommend Crop & Irrigation',
                aidTitle: 'Government Aids',
                stateLabel: 'State',
                landLabel: 'Land Size (acres)',
                aidButton: 'Get Government Schemes',
                backButton: 'Back',
                resultFields: {
                    'Soil Type': 'Soil Type',
                    'Pest Detection': 'Pest Detection',
                    'Recommended Crops': 'Recommended Crops',
                    'Irrigation Status': 'Irrigation Status',
                    'Estimated Yield': 'Estimated Yield',
                    'State': 'State',
                    'Land Size': 'Land Size',
                    'Available Schemes': 'Available Schemes',
                    'Eligibility': 'Eligibility',
                    'Contact': 'Contact'
                }
            } 
        };
        updateUIText();
    }
}

function updateUIText() {
    console.log('Updating UI for language:', currentLanguage);
    const lang = translations[currentLanguage] || translations['en'] || {};
    
    document.getElementById('appTitle').textContent = lang.appTitle || 'AgriBot - Smart Farming Assistant';
    document.getElementById('soilOption').textContent = lang.soilOption || 'Soil & Pest Detection';
    document.getElementById('cropOption').textContent = lang.cropOption || 'Crop & Irrigation Management';
    document.getElementById('aidOption').textContent = lang.aidOption || 'Government Aids';
    
    document.getElementById('soilTitle').textContent = lang.soilTitle || 'Soil & Pest Detection';
    document.getElementById('soilImageLabel').textContent = lang.soilImageLabel || 'Upload Soil Image';
    document.getElementById('soilButton').textContent = lang.soilButton || 'Analyze Soil';
    
    document.getElementById('cropTitle').textContent = lang.cropTitle || 'Crop & Irrigation Management';
    document.getElementById('nitrogenLabel').textContent = lang.nitrogenLabel || 'Nitrogen (N) Level';
    document.getElementById('phosphorusLabel').textContent = lang.phosphorusLabel || 'Phosphorus (P) Level';
    document.getElementById('potassiumLabel').textContent = lang.potassiumLabel || 'Potassium (K) Level';
    document.getElementById('tempLabel').textContent = lang.tempLabel || 'Temperature (°C)';
    document.getElementById('humidityLabel').textContent = lang.humidityLabel || 'Humidity (%)';
    document.getElementById('phLabel').textContent = lang.phLabel || 'Soil pH';
    document.getElementById('rainfallLabel').textContent = lang.rainfallLabel || 'Rainfall (mm)';
    document.getElementById('soilTypeLabel').textContent = lang.soilTypeLabel || 'Soil Type';
    document.getElementById('cropButton').textContent = lang.cropButton || 'Recommend Crop & Irrigation';
    
    document.getElementById('aidTitle').textContent = lang.aidTitle || 'Government Aids';
    document.getElementById('stateLabel').textContent = lang.stateLabel || 'State';
    document.getElementById('landLabel').textContent = lang.landLabel || 'Land Size (acres)';
    document.getElementById('aidButton').textContent = lang.aidButton || 'Get Government Schemes';
    
    document.getElementById('backButton1').textContent = lang.backButton || 'Back';
    document.getElementById('backButton2').textContent = lang.backButton || 'Back';
    document.getElementById('backButton3').textContent = lang.backButton || 'Back';
}

function showForm(formId) {
    console.log('Showing form:', formId);
    document.getElementById('options').classList.add('hidden');
    document.getElementById(formId).classList.remove('hidden');
    document.getElementById('farmerBot').classList.add('hidden');
}

function goBack() {
    console.log('Going back to options');
    document.getElementById('soil_prediction').classList.add('hidden');
    document.getElementById('crop_mgmt').classList.add('hidden');
    document.getElementById('govt_aid').classList.add('hidden');
    document.getElementById('options').classList.remove('hidden');
    document.getElementById('farmerBot').classList.remove('hidden');
    document.getElementById('soilResult').innerHTML = '';
    document.getElementById('cropResult').innerHTML = '';
    document.getElementById('aidResult').innerHTML = '';
    
    // Reset all forms
    const soilForm = document.getElementById('soilForm');
    const cropForm = document.getElementById('cropForm');
    const aidForm = document.getElementById('aidForm');
    if (soilForm) soilForm.reset();
    if (cropForm) cropForm.reset();
    if (aidForm) aidForm.reset();
}

function showToast(message) {
    console.log('Showing toast:', message);
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function changeLanguage() {
    currentLanguage = document.getElementById('language').value;
    console.log('Language changed to:', currentLanguage);
    updateUIText();
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM fully loaded');
    loadTranslations();

    // Soil Form Submission Handler
    const soilForm = document.getElementById('soilForm');
    if (soilForm) {
        soilForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const loading = document.getElementById('soilLoading');
            const resultDiv = document.getElementById('soilResult');
            const soilImage = document.getElementById('soilImage').files[0];

            if (!soilImage) {
                showToast('Please select an image');
                return;
            }

            const formData = new FormData();
            formData.append('image', soilImage);
            formData.append('language', currentLanguage);

            loading.style.display = 'block';
            resultDiv.innerHTML = '';

            try {
                const response = await fetch('/predict_soil', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                console.log('Soil Prediction Response:', data);
                loading.style.display = 'none';

            if (response.ok && data.success) {
                resultDiv.innerHTML = `
                    <div class="result">
                        <div class="result-item"><span class="result-title">Soil Type:</span> ${data.data.soil_type}</div>
                        <div class="result-item"><span class="result-title">Pest Detection:</span> ${data.data.pest_detection}</div>
                    </div>
                `;
            } else {
                console.log('Server error:', data);
                resultDiv.innerHTML = <div class="error">Error: ${data.error || 'Failed to analyze soil'}</div>;
            }
        } catch (error) {
            console.error('Fetch error:', error);
            loading.style.display = 'none';
            resultDiv.innerHTML = <div class="error">Network error: ${error.message}</div>;
        }
    });

    // Crop Form Submission Handler
    const cropForm = document.getElementById('cropForm');
    if (cropForm) {
        cropForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            console.log('Crop form submitted');
            const loading = document.getElementById('cropLoading');
            const resultDiv = document.getElementById('cropResult');

            const inputs = {
                nitrogen: document.getElementById('nitrogen'),
                phosphorus: document.getElementById('phosphorus'),
                potassium: document.getElementById('potassium'),
                temperature: document.getElementById('temperature'),
                humidity: document.getElementById('humidity'),
                ph: document.getElementById('ph'),
                rainfall: document.getElementById('rainfall'),
                soil_type: document.getElementById('soil_type')
            };

            const data = {
                nitrogen: parseFloat(inputs.nitrogen.value) || 0,
                phosphorus: parseFloat(inputs.phosphorus.value) || 0,
                potassium: parseFloat(inputs.potassium.value) || 0,
                temperature: parseFloat(inputs.temperature.value) || 0,
                humidity: parseFloat(inputs.humidity.value) || 0,
                ph: parseFloat(inputs.ph.value) || 0,
                rainfall: parseFloat(inputs.rainfall.value) || 0,
                soil_type: inputs.soil_type.value.trim() || 'Alluvial',
                lang: currentLanguage
            };

        // Input validation
        if (!data.nitrogen || !data.phosphorus || !data.potassium || !data.temperature ||
            !data.humidity || !data.ph || !data.rainfall || !data.soil_type) {
            console.log('Missing crop form data');
            showToast('Please fill all fields');
            return;
        }
        if (data.nitrogen < 0 || data.phosphorus < 0 || data.potassium < 0 ||
            data.temperature < -50 || data.temperature > 50 ||
            data.humidity < 0 || data.humidity > 200 ||
            data.ph < 0 || data.ph > 14 ||
            data.rainfall < 0) {
            console.log('Invalid crop form data');
            showToast('Please enter valid values (e.g., positive numbers, pH 0-14)');
            return;
        }

            console.log('Sending crop data:', JSON.stringify(data));
            loading.style.display = 'block';
            resultDiv.innerHTML = '';

            try {
                const response = await fetch('/recommend_crop', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                console.log('Crop response:', result);
                loading.style.display = 'none';

            if (response.ok) {
                // Format crops into a single line (up to 4)
                const cropsText = (result.crops || []).slice(0, 4).map(crop => 
                    ${crop.crop} (${crop.probability}%)
                ).join(', ');
                
                resultDiv.innerHTML = `
                    <div class="result">
                        <div class="result-item"><span class="result-title">Crop:</span> ${cropsText || 'No crops recommended'}</div>
                        <div class="result-item"><span class="result-title">Irrigation:</span> ${result.irrigation || 'Not specified'}</div>
                        <div class="result-item"><span class="result-title">Estimated Yield:</span> ${result.estimated_yield || 'Not available'}</div>
                    </div>
                `;
            } else {
                console.log('Server error:', result);
                resultDiv.innerHTML = <div class="error">Error: ${result.error || 'Failed to recommend crop'}</div>;
            }
        } catch (error) {
            console.error('Fetch error:', error);
            loading.style.display = 'none';
            resultDiv.innerHTML = <div class="error">Network error: ${error.message}</div>;
        }
    });

    // Government Aid Form Submission Handler
    const aidForm = document.getElementById('aidForm');
    if (aidForm) {
        aidForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const loading = document.getElementById('aidLoading');
            const resultDiv = document.getElementById('aidResult');
            const data = {
                state: document.getElementById('state').value.trim().toLowerCase(),
                land_size: parseFloat(document.getElementById('land_size').value) || 0,
                lang: currentLanguage
            };

            if (!data.state || isNaN(data.land_size) || data.land_size < 0) {
                showToast('Please fill all fields with a valid state and non-negative land size');
                return;
            }

            console.log('Sending aid data:', data);
            loading.style.display = 'block';
            resultDiv.innerHTML = '';

            try {
                const response = await fetch('/government_aids', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                console.log('Received aid response:', result);
                loading.style.display = 'none';

            if (response.ok && result.success) {
                resultDiv.innerHTML = `
                    <div class="result">
                        <div class="result-item"><span class="result-title">State:</span> ${result.data.state}</div>
                        <div class="result-item"><span class="result-title">Land Size:</span> ${result.data.land_size} acres</div>
                        <div class="result-item"><span class="result-title">Schemes:</span> ${result.data.available_schemes.join(', ')}</div>
                        <div class="result-item"><span class="result-title">Eligibility:</span> ${result.data.eligibility}</div>
                        <div class="result-item"><span class="result-title">Contact:</span> ${result.data.contact}</div>
                    </div>
                `;
            } else {
                console.log('Server error:', result);
                resultDiv.innerHTML = <div class="error">Error: ${result.error || 'Failed to fetch schemes'}</div>;
            }
        } catch (error) {
            console.error('Fetch error:', error);
            loading.style.display = 'none';
            resultDiv.innerHTML = <div class="error">Network error: ${error.message}</div>;
        }
    });
});
