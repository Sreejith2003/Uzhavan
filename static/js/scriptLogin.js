document.addEventListener('DOMContentLoaded', () => {
    // Tab switching functionality
    const tabButtons = document.querySelectorAll(".tab-btn");
    const authForms = document.querySelectorAll(".auth-form");

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            tabButtons.forEach(b => b.classList.remove("active"));
            authForms.forEach(form => form.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(`${btn.dataset.tab}Form`).classList.add("active");
        });
    });

    // LOGIN form submission
    document.getElementById("loginForm").addEventListener("submit", async function(e) {
        e.preventDefault();
        const phone_number = document.getElementById("loginPhone").value.trim();
        const email = document.getElementById("loginEmail").value.trim();
        const password = document.getElementById("loginPassword").value.trim();

        // Validate phone number (10 digits, starting with 6-9)
        const phoneRegex = /^[6-9]\d{9}$/;
        if (!phoneRegex.test(phone_number)) {
            alert("Please enter a valid 10-digit Indian phone number starting with 6-9 (e.g., 9876543210)");
            return;
        }

        // Validate email if provided
        if (email && !/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/.test(email)) {
            alert("Please enter a valid email address or leave it blank");
            return;
        }

        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, phone_number, password })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            if (result.success) {
                alert("Login successful! Welcome, " + result.user.full_name);
                window.location.href = "/index";
            } else {
                alert(result.error || "Login failed. Please try again.");
            }
        } catch (err) {
            alert("Network error during login: " + err.message);
        }
    });

    // REGISTER form submission
    document.getElementById("registerForm").addEventListener("submit", async function(e) {
        e.preventDefault();
        const full_name = document.getElementById("registerName").value.trim();
        const phone_number = document.getElementById("registerPhone").value.trim();
        const email = document.getElementById("registerEmail").value.trim();
        const password = document.getElementById("registerPassword").value.trim();
        const confirmPassword = document.getElementById("confirmPassword").value.trim();

        // Validate phone number (10 digits, starting with 6-9)
        const phoneRegex = /^[6-9]\d{9}$/;
        if (!phoneRegex.test(phone_number)) {
            alert("Please enter a valid 10-digit Indian phone number starting with 6-9 (e.g., 9876543210)");
            return;
        }

        // Validate email if provided
        if (email && !/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/.test(email)) {
            alert("Please enter a valid email address or leave it blank");
            return;
        }

        if (password !== confirmPassword) {
            alert("Passwords do not match");
            return;
        }

        try {
            const response = await fetch('/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ full_name, email, phone_number, password })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            if (result.success) {
                alert("Registration successful! Please log in.");
                document.querySelector('.tab-btn[data-tab="login"]').click();
            } else {
                alert(result.error || "Registration failed. Please try again.");
            }
        } catch (err) {
            alert("Network error during registration: " + err.message);
        }
    });

    // Slogan rotation logic
    const slogans = [
        { text: "உழுதுண்டு வாழ்வாரே வாழ்வார் மற்றெல்லாம் தொழுதுண்டு பின்செல் பவர்", lang: "தமிழ்" },
        { text: "ഉഴുതുണ്ട് ജീവിക്കുന്നവരാണ് ജീവിക്കുന്നവർ മറ്റെല്ലാവരും അനുഗമിച്ച് ജീവിക്കുന്നവരാണ്", lang: "മലയാളം" },
        { text: "ఉధృతి తిని జీవించేవారే నిజంగా జీవించేవారు మిగతావారంతా ఇతరులను ఆధారపడి జీవించేవారు", lang: "తెలుగు" },
        { text: "ಉಳುಮೆ ಮಾಡಿ ತಿನ್ನುವವರೇ ನಿಜವಾಗಿ ಬದುಕುವವರು ಉಳಿದವರೆಲ್ಲಾ ಇತರರನ್ನು ಅವಲಂಬಿಸಿ ಬದುಕುವವರು", lang: "ಕನ್ನಡ" },
        { text: "जो हल चलाकर खाते हैं, वे ही सच्चे जीवन जीते हैं बाकी सब दूसरों के आगे हाथ फैलाकर जीते हैं", lang: "हिंदी" },
        { text: "Only those who plough and eat shall truly live All others are but followers, eating from their hands", lang: "English" }
    ];

    const sloganText = document.getElementById('dynamic-slogan');
    const sloganLanguage = document.getElementById('slogan-language');

    // Add CSS for line breaks and text alignment
    const style = document.createElement('style');
    style.textContent = `
        #dynamic-slogan {
            text-align: center;
            line-height: 1.4;
            min-height: 3em;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        .slogan-line {
            display: block;
            width: 100%;
            text-align: center;
            margin: 0;
        }
        
        .slogan-line.tamil {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        #slogan-language {
            font-style: italic;
            margin-top: 0.5em;
            opacity: 0;
            transition: opacity 0.5s ease;
        }
        
        #slogan-language.visible {
            opacity: 1;
        }
        
        .fade-out {
            opacity: 0;
            transition: opacity 0.7s ease;
        }
    `;
    document.head.appendChild(style);

    let currentSloganIndex = 0;

    function typeSlogan(slogan, callback) {
        sloganText.innerHTML = ''; // Clear previous content
        sloganLanguage.textContent = '';
        sloganText.classList.remove('fade-out');
        sloganLanguage.classList.remove('visible');

        // Define lines based on language
        let lines;
        const isTamil = slogan.lang === "Tamil";
        if (isTamil) {
            // Explicit split for Tamil to ensure two lines
            lines = [
                "உழுதுண்டு வாழ்வாரே வாழ்வார்",
                "மற்றெல்லாம் தொழுதுண்டு பின்செல் பவர்"
            ];
        } else {
            // For other languages, split at a logical point
            const splitPoint = slogan.text.indexOf("മറ്റെല്ലാവരും") || 
                              slogan.text.indexOf("మిగతావారంతా") || 
                              slogan.text.indexOf("ಉಳಿದವರೆಲ್ಲಾ") || 
                              slogan.text.indexOf("बाकी सब") || 
                              slogan.text.indexOf("All others") || 
                              (slogan.text.length / 2);
            lines = [
                slogan.text.substring(0, splitPoint).trim(),
                slogan.text.substring(splitPoint).trim()
            ];
        }

        const lineElements = [];

        // Create line spans with explicit content
        lines.forEach((line, index) => {
            const lineSpan = document.createElement('span');
            lineSpan.className = 'slogan-line';
            if (isTamil) {
                lineSpan.classList.add('tamil'); // Add tamil class for specific styling
            }
            lineSpan.textContent = line; // Set text directly
            sloganText.appendChild(lineSpan);
            lineElements.push(lineSpan);

            // Add line break after the first line
            if (index === 0) {
                sloganText.appendChild(document.createElement('br'));
            }
        });

        // Display language
        sloganLanguage.textContent = `— ${slogan.lang}`;
        sloganLanguage.classList.add('visible');
        callback();
    }

    function updateSlogan() {
        const currentSlogan = slogans[currentSloganIndex];
        sloganText.classList.add('fade-out');
        sloganLanguage.classList.remove('visible');

        setTimeout(() => {
            typeSlogan(currentSlogan, () => {
                currentSloganIndex = (currentSloganIndex + 1) % slogans.length;
                setTimeout(updateSlogan, 5000); // Show each slogan for 5 seconds
            });
        }, 700); // Match fade-out transition
    }

    // Initial call
    updateSlogan();
});