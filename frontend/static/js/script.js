// --- DOM Elements ---
const lookupBtn = document.getElementById('lookupBtn');
const employeeIdInput = document.getElementById('employeeIdInput');
const fetchStatus = document.getElementById('fetchStatus');
const resultEl = document.getElementById('result');
const themeToggle = document.getElementById('themeToggle');

// --- Dark Mode ---
if (localStorage.getItem('theme') === 'dark') {
    document.body.classList.add('dark');
    themeToggle.innerHTML = '<i data-lucide="sun"></i>';
}

themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark');
    const isDark = document.body.classList.contains('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    themeToggle.innerHTML = `<i data-lucide="${isDark ? 'sun' : 'moon'}"></i>`;
    lucide.createIcons();
});

// --- Profile Dropdown ---
const userAvatar = document.getElementById('userAvatar');
const profileDropdown = document.getElementById('profileDropdown');

userAvatar.addEventListener('click', (e) => {
    e.stopPropagation();
    profileDropdown.classList.toggle('open');
    if (profileDropdown.classList.contains('open')) lucide.createIcons();
});

document.addEventListener('click', () => {
    profileDropdown.classList.remove('open');
});

// Modal Elements
const customModal = document.getElementById('customModal');
const modalIcon = document.getElementById('modalIcon');
const modalTitle = document.getElementById('modalTitle');
const modalMessage = document.getElementById('modalMessage');

// --- Modal Functions ---
function showModal(title, message, type = 'error') {
    modalTitle.innerText = title;
    modalMessage.innerText = message;
    modalIcon.innerHTML = `<i data-lucide="${type === 'error' ? 'x-circle' : 'info'}"></i>`;
    customModal.style.display = 'flex';
    lucide.createIcons();
}

function closeModal() {
    customModal.style.display = 'none';
}

window.closeModal = closeModal;

function clearPrediction() {
    employeeIdInput.value = '';
    fetchStatus.innerHTML = '';
    resultEl.style.display = 'none';
    resultEl.innerHTML = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
    employeeIdInput.focus();
}
window.clearPrediction = clearPrediction;

// --- Core Prediction Logic (Background) ---
async function performPrediction(featureData, employeeId) {
    // Show loading state in the result section
    resultEl.style.display = 'block';
    resultEl.innerHTML = `
        <div class="card" style="text-align: center; padding: 3rem;">
            <div style="margin-bottom: 1rem; color: var(--primary-color);">
                <i data-lucide="loader-2" class="spin" style="width: 48px; height: 48px;"></i>
            </div>
            <p style="font-weight: 500;">Analyzing Employee #${employeeId}...</p>
            <p style="font-size: 0.875rem; color: var(--text-muted);">Processing retention signals and calculating risk score.</p>
        </div>
    `;
    lucide.createIcons();
    resultEl.scrollIntoView({ behavior: 'smooth', block: 'center' });

    // Prepare payload (mapping Feast keys to model expected keys)
    // Note: The model expects specific types and names
    const data = { 
        ...featureData,
        employee_id: Number(employeeId)
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!response.ok || result.error) {
            throw new Error(result.error || "Prediction engine failure");
        }

        const isLeaving = Number(result.prediction) === 1;
        
        resultEl.innerHTML = `
            <div class="risk-card ${isLeaving ? 'high' : 'low'}">
                <div style="margin-bottom: 1.5rem;">
                    <i data-lucide="${isLeaving ? 'alert-triangle' : 'check-circle'}" style="width: 64px; height: 64px;"></i>
                </div>
                <div class="risk-desc">${isLeaving ? 'ATTENTION REQUIRED' : 'STABLE RETENTION'}</div>
                <div class="risk-value">${isLeaving ? 'High Attrition Risk' : 'Low Attrition Risk'}</div>
                <p style="margin-top: 1rem; opacity: 0.9; max-width: 500px; margin-left: auto; margin-right: auto;">
                    ${isLeaving 
                        ? `Employee #${employeeId} shows patterns highly correlated with attrition. We recommend immediate engagement and review of their work environment.` 
                        : `Employee #${employeeId} is exhibiting strong retention signals. Their current profile aligns with long-term stability within the organization.`}
                </p>
                <div style="margin-top: 2rem;">
                    <button class="btn btn-outline" onclick="clearPrediction()">
                        <i data-lucide="x"></i> Clear
                    </button>
                </div>
            </div>
        `;
        lucide.createIcons();

    } catch (error) {
        resultEl.innerHTML = `
            <div class="card" style="border-left: 4px solid var(--danger); padding: 2rem;">
                <h3 style="color: var(--danger); display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <i data-lucide="x-circle"></i> Analysis Failed
                </h3>
                <p style="color: var(--text-muted);">${error.message}</p>
            </div>
        `;
        lucide.createIcons();
    }
}

// --- Combined Lookup & Predict Logic ---
lookupBtn.addEventListener('click', async () => {
    const employeeId = employeeIdInput.value.trim();
    
    if (!employeeId) {
        fetchStatus.innerHTML = `<span style="color: var(--warning); font-size: 0.875rem;">⚠️ Please enter an Employee ID.</span>`;
        return;
    }

    // UI Reset
    fetchStatus.innerHTML = `<span style="color: var(--text-muted); font-size: 0.875rem;">⏳ Syncing profile & running AI analysis...</span>`;
    lookupBtn.disabled = true;
    resultEl.style.display = 'none';

    try {
        // Step 1: Sync Data from Feast (Background)
        const res = await fetch(`/features/${employeeId}`);
        const response = await res.json();

        if (res.status === 404) {
            fetchStatus.innerHTML = `<span style="color: var(--danger); font-size: 0.875rem;">❌ Employee ID ${employeeId} not found.</span>`;
            showModal("Employee Not Found", `We couldn't find an active record for ID: ${employeeId}. Please verify the ID and try again.`);
            return;
        }

        if (!res.ok) throw new Error(response.error || "Failed to fetch data from Feast");

        // Step 2: Automatically trigger Prediction with the fetched features
        if (response['features']) {
            fetchStatus.innerHTML = `<span style="color: var(--success); font-size: 0.875rem;">✅ Data Synced. Generating results...</span>`;
            await performPrediction(response['features'], employeeId);
        }

    } catch (err) {
        fetchStatus.innerHTML = `<span style="color: var(--danger); font-size: 0.875rem;">❌ Error: ${err.message}</span>`;
    } finally {
        lookupBtn.disabled = false;
    }
});

// Utility: Loader Animation
const style = document.createElement('style');
style.textContent = `
    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    .spin { animation: spin 1s linear infinite; }
`;
document.head.appendChild(style);