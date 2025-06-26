console.log('script.js: Initializing...');

let currentLanguage = 'en';
let translations = {};

async function loadTranslations() {
    console.log('Loading translations...');
    try {
        const response = await fetch('/translations.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        translations = await response.json();
        console.log('Translations loaded:', Object.keys(translations));
        updateUIText();
    } catch (error) {
        console.error('Error loading translations:', error);
        showToast('Failed to load translations - using default English');
        translations = {
            en: {
                appTitle: 'Uzhavan - Smart Farming Assistant',
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
                    'Contact': 'Contact',
                    'Note': 'Note',
                    'acres': 'acres'
                },
                errorMessages: {
                    noImage: 'Please select an image',
                    soilAnalysis: 'Failed to analyze soil',
                    network: 'Network error',
                    invalidNumber: 'must be a valid number',
                    rangeError: 'must be between',
                    and: 'and',
                    noCrops: 'No crops recommended',
                    notSpecified: 'Not specified',
                    notAvailable: 'Not available',
                    invalidAidInput: 'Please fill all fields with a valid state and non-negative land size',
                    noSchemes: 'No schemes found',
                    schemesFailed: 'Failed to load government schemes',
                    error: 'Error',
                    unknown: 'Unknown'
                },
                successMessages: {
                    soilAnalysis: 'Soil analysis completed!',
                    cropRecommendation: 'Crop recommendation generated!',
                    schemesLoaded: 'Government schemes loaded!'
                }
            }
        };
        // Dynamically translate UI labels for other languages
        for (const lang of ['ta', 'ml', 'te', 'kn', 'hi']) {
            translations[lang] = await fetchTranslations(lang);
        }
        updateUIText();
    }
}

// Fetch UI translations dynamically from backend
async function fetchTranslations(lang) {
    if (lang === 'en') return translations.en;
    try {
        const response = await fetch('/api/translate_ui', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language: lang, ui_text: translations.en })
        });
        const data = await response.json();
        return data.translated_ui || translations.en;
    } catch (error) {
        console.error(`Error fetching translations for ${lang}:`, error);
        return translations.en;
    }
}

// Pass through backend-translated values
function translatePredictionValue(category, value) {
    if (!value) return translations[currentLanguage]?.errorMessages?.notAvailable || 'Not available';
    // Backend sends translated text, so return as-is
    return value;
}

// Pass through backend-translated dynamic text
function translateDynamicText(text) {
    if (!text) return translations[currentLanguage]?.errorMessages?.notAvailable || 'Not available';
    // Backend sends translated text, so return as-is
    return text;
}

// Update translatePestNote to use backend translations
function translatePestNote(soilType, pests) {
    const lang = translations[currentLanguage] || translations['en'] || {};
    const template = lang.pest_note_template || 'This {soil_type} may have these pests: {pests}';
    const translatedSoilType = translatePredictionValue('soil_type', soilType);
    const translatedPests = pests.map(pest => translatePredictionValue('pest_detection', pest)).join(', ');
    return template.replace('{soil_type}', translatedSoilType).replace('{pests}', translatedPests);
}

function updateUIText() {
    console.log('Updating UI for language:', currentLanguage);
    const lang = translations[currentLanguage] || translations['en'] || {};
    
    document.getElementById('appTitle').textContent = lang.appTitle || 'Uzhavan - Smart Farming Assistant';
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

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM fully loaded');
    loadTranslations();

    const soilForm = document.getElementById('soilForm');
    if (soilForm) {
        soilForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const loading = document.getElementById('soilLoading');
            const resultDiv = document.getElementById('soilResult');
            const soilImage = document.getElementById('soilImage').files[0];

            if (!soilImage) {
                showToast(translations[currentLanguage]?.errorMessages?.noImage || 'Please select an image');
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
                    const translatedSoilType = translatePredictionValue('soil_type', data.data.soil_type);
                    const translatedPests = data.data.pest_detection.map(pest => translatePredictionValue('pest_detection', pest));
                    const translatedNote = translatePestNote(data.data.soil_type, data.data.pest_detection);
                    resultDiv.innerHTML = `
                        <div class="result">
                            <div class="result-item">${translations[currentLanguage]?.resultFields?.['Soil Type'] || 'Soil Type'}: ${translatedSoilType}</div>
                            <div class="result-item">${translations[currentLanguage]?.resultFields?.['Pest Detection'] || 'Pest Detection'}: ${translatedPests.join(', ')}</div>
                            <div class="result-item">${translations[currentLanguage]?.resultFields?.['Note'] || 'Note'}: ${translatedNote}</div>
                        </div>
                    `;
                    showToast(translations[currentLanguage]?.successMessages?.soilAnalysis || 'Soil analysis completed!');
                } else {
                    resultDiv.innerHTML = `<div class="error">${translations[currentLanguage]?.errorMessages?.error || 'Error'}: ${data.error || translations[currentLanguage]?.errorMessages?.soilAnalysis || 'Failed to analyze soil'}</div>`;
                    showToast(translations[currentLanguage]?.errorMessages?.soilAnalysis || 'Failed to analyze soil');
                }
            } catch (error) {
                console.error('Soil Prediction Error:', error);
                loading.style.display = 'none';
                resultDiv.innerHTML = `<div class="error">${translations[currentLanguage]?.errorMessages?.network || 'Network error'}: ${error.message}</div>`;
                showToast(translations[currentLanguage]?.errorMessages?.network || 'Network error during soil analysis');
            }
        });
    }

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

            const validationRules = {
                nitrogen: { min: 0, max: 300, label: translations[currentLanguage]?.nitrogenLabel || 'Nitrogen' },
                phosphorus: { min: 0, max: 300, label: translations[currentLanguage]?.phosphorusLabel || 'Phosphorus' },
                potassium: { min: 0, max: 300, label: translations[currentLanguage]?.potassiumLabel || 'Potassium' },
                temperature: { min: -50, max: 60, label: translations[currentLanguage]?.tempLabel || 'Temperature' },
                humidity: { min: 0, max: 100, label: translations[currentLanguage]?.humidityLabel || 'Humidity' },
                ph: { min: 0, max: 14, label: translations[currentLanguage]?.phLabel || 'pH' },
                rainfall: { min: 0, max: 5000, label: translations[currentLanguage]?.rainfallLabel || 'Rainfall' }
            };

            const validationErrors = [];
            for (const [field, value] of Object.entries(data)) {
                const rule = validationRules[field];
                if (rule) {
                    if (isNaN(value)) {
                        validationErrors.push(`${rule.label} ${translations[currentLanguage]?.errorMessages?.invalidNumber || 'must be a valid number'}`);
                    } else if (value < rule.min || value > rule.max) {
                        validationErrors.push(`${rule.label} ${translations[currentLanguage]?.errorMessages?.rangeError || 'must be between'} ${rule.min} ${translations[currentLanguage]?.errorMessages?.and || 'and'} ${rule.max}`);
                    }
                }
            }

            if (validationErrors.length > 0) {
                showToast(validationErrors.join('; '));
                console.error('Validation errors:', validationErrors);
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
                    const cropsText = (result.crops || [])
                        .slice(0, 4)
                        .map(crop => `${translatePredictionValue('crops', crop.crop)} (${(crop.probability * 100).toFixed(1)}%)`)
                        .join(', ');
                    const translatedIrrigation = translatePredictionValue('irrigation', result.irrigation);
                    const translatedEstimatedYield = translateDynamicText(result.estimated_yield);
                    const translatedNote = translateDynamicText(result.note);

                    resultDiv.innerHTML = `
                        <div class="result">
                            <div class="result-item"><span class="result-title">${translations[currentLanguage]?.resultFields?.['Recommended Crops'] || 'Recommended Crops'}:</span> ${cropsText || translations[currentLanguage]?.errorMessages?.noCrops || 'No crops recommended'}</div>
                            <div class="result-item"><span class="result-title">${translations[currentLanguage]?.resultFields?.['Irrigation Status'] || 'Irrigation Status'}:</span> ${translatedIrrigation || translations[currentLanguage]?.errorMessages?.notSpecified || 'Not specified'}</div>
                            <div class="result-item"><span class="result-title">${translations[currentLanguage]?.resultFields?.['Estimated Yield'] || 'Estimated Yield'}:</span> ${translatedEstimatedYield}</div>
                            ${translatedNote ? `<div class="result-item"><span class="result-title">${translations[currentLanguage]?.resultFields?.['Note'] || 'Note'}:</span> ${translatedNote}</div>` : ''}
                        </div>
                    `;
                    showToast(translations[currentLanguage]?.successMessages?.cropRecommendation || 'Crop recommendation generated!');
                } else {
                    throw new Error(result.error || `Server error (Status: ${response.status})`);
                }
            } catch (error) {
                console.error('Crop form error:', error);
                loading.style.display = 'none';
                resultDiv.innerHTML = `<div class="error">${translations[currentLanguage]?.errorMessages?.error || 'Error'}: ${error.message}</div>`;
                showToast(translations[currentLanguage]?.errorMessages?.cropRecommendation || 'Failed to generate crop recommendation');
            }
        });
    }

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
                showToast(translations[currentLanguage]?.errorMessages?.invalidAidInput || 'Please fill all fields with a valid state and non-negative land size');
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
                    const schemesHTML = (result.data.available_schemes || [])
                        .map(s => `<li>${translatePredictionValue('schemes', s)}</li>`)
                        .join('');
                    const translatedEligibility = translateDynamicText(result.data.eligibility);
                    const translatedContact = translateDynamicText(result.data.contact);

                    resultDiv.innerHTML = `
                        <div class="result">
                            <div class="result-item"><span class="result-title">${translations[currentLanguage]?.resultFields?.['State'] || 'State'}:</span> ${result.data.state}</div>
                            <div class="result-item"><span class="result-title">${translations[currentLanguage]?.resultFields?.['Land Size'] || 'Land Size'}:</span> ${result.data.land_size} ${translations[currentLanguage]?.resultFields?.acres || 'acres'}</div>
                            <div class="result-item"><span class="result-title">${translations[currentLanguage]?.resultFields?.['Available Schemes'] || 'Available Schemes'}:</span> <ul class="result-list">${schemesHTML}</ul></div>
                            <div class="result-item"><span class="result-title">${translations[currentLanguage]?.resultFields?.['Eligibility'] || 'Eligibility'}:</span> ${translatedEligibility}</div>
                            <div class="result-item"><span class="result-title">${translations[currentLanguage]?.resultFields?.['Contact'] || 'Contact'}:</span> ${translatedContact}</div>
                        </div>
                    `;
                    showToast(translations[currentLanguage]?.successMessages?.schemesLoaded || 'Government schemes loaded!');
                } else {
                    resultDiv.innerHTML = `<div class="error">${translations[currentLanguage]?.errorMessages?.error || 'Error'}: ${result.error || translations[currentLanguage]?.errorMessages?.noSchemes || 'No schemes found'}</div>`;
                    showToast(translations[currentLanguage]?.errorMessages?.schemesFailed || 'Failed to load government schemes');
                }
            } catch (error) {
                console.error('Aid form error:', error);
                loading.style.display = 'none';
                resultDiv.innerHTML = `<div class="error">${translations[currentLanguage]?.errorMessages?.network || 'Network error'}: ${error.message}</div>`;
                showToast(translations[currentLanguage]?.errorMessages?.network || 'Network error during scheme loading');
            }
        });
    }

    const languageSelect = document.getElementById('language');
    if (languageSelect) {
        languageSelect.addEventListener('change', changeLanguage);
    }
});