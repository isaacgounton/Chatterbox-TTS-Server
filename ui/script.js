// ui/script.js
// Client-side JavaScript for the TTS Server web interface.
// Handles UI interactions, API communication, audio playback, and settings management.

// Helper function to make authenticated requests
function getAuthHeaders() {
    const token = localStorage.getItem('authToken');
    const headers = {};
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
}

// Helper function to handle authentication errors
function handleAuthError(response) {
    if (response.status === 401) {
        localStorage.removeItem('authToken');
        window.location.reload();
        return true;
    }
    return false;
}

document.addEventListener('DOMContentLoaded', async function () {
    // --- Global Flags & State ---
    let isGenerating = false;
    let wavesurfer = null;
    let currentAudioBlobUrl = null;
    let saveStateTimeout = null;

    let currentConfig = {};
    let currentUiState = {};
    let appPresets = [];
    let initialReferenceFiles = [];
    let initialPredefinedVoices = [];

    let hideChunkWarning = false;
    let hideGenerationWarning = false;
    let currentVoiceMode = 'predefined';

    const DEBOUNCE_DELAY_MS = 750;

    // --- DOM Element Selectors ---
    const appTitleLink = document.getElementById('app-title-link');
    const themeToggleButton = document.getElementById('theme-toggle-btn');
    const themeSwitchThumb = themeToggleButton ? themeToggleButton.querySelector('.theme-switch-thumb') : null;
    const logoutButton = document.getElementById('logout-btn');
    const notificationArea = document.getElementById('notification-area');
    const ttsForm = document.getElementById('tts-form');
    const ttsFormHeader = document.getElementById('tts-form-header');
    const textArea = document.getElementById('text');
    const charCount = document.getElementById('char-count');
    const generateBtn = document.getElementById('generate-btn');
    const splitTextToggle = document.getElementById('split-text-toggle');
    const chunkSizeControls = document.getElementById('chunk-size-controls');
    const chunkSizeSlider = document.getElementById('chunk-size-slider');
    const chunkSizeValue = document.getElementById('chunk-size-value');
    const chunkExplanation = document.getElementById('chunk-explanation');
    const voiceModeRadios = document.querySelectorAll('input[name="voice_mode"]');
    const predefinedVoiceOptionsDiv = document.getElementById('predefined-voice-options');
    const predefinedVoiceSelect = document.getElementById('predefined-voice-select');
    const predefinedVoiceImportButton = document.getElementById('predefined-voice-import-button');
    const predefinedVoiceRefreshButton = document.getElementById('predefined-voice-refresh-button');
    const predefinedVoiceFileInput = document.getElementById('predefined-voice-file-input');
    const cloneOptionsDiv = document.getElementById('clone-options');
    const cloneReferenceSelect = document.getElementById('clone-reference-select');
    const cloneImportButton = document.getElementById('clone-import-button');
    const cloneRefreshButton = document.getElementById('clone-refresh-button');
    const cloneFileInput = document.getElementById('clone-file-input');
    const presetsContainer = document.getElementById('presets-container');
    const presetsPlaceholder = document.getElementById('presets-placeholder');
    const temperatureSlider = document.getElementById('temperature');
    const temperatureValueDisplay = document.getElementById('temperature-value');
    const exaggerationSlider = document.getElementById('exaggeration');
    const exaggerationValueDisplay = document.getElementById('exaggeration-value');
    const cfgWeightSlider = document.getElementById('cfg-weight');
    const cfgWeightValueDisplay = document.getElementById('cfg-weight-value');
    const speedFactorSlider = document.getElementById('speed-factor');
    const speedFactorValueDisplay = document.getElementById('speed-factor-value');
    const speedFactorWarningSpan = document.getElementById('speed-factor-warning');
    const seedInput = document.getElementById('seed');
    const languageSelectContainer = document.getElementById('language-select-container');
    const languageSelect = document.getElementById('language');
    const saveGenDefaultsBtn = document.getElementById('save-gen-defaults-btn');
    const genDefaultsStatus = document.getElementById('gen-defaults-status');
    const serverConfigForm = document.getElementById('server-config-form');
    const saveConfigBtn = document.getElementById('save-config-btn');
    const restartServerBtn = document.getElementById('restart-server-btn');
    const configStatus = document.getElementById('config-status');
    const resetSettingsBtn = document.getElementById('reset-settings-btn');
    const audioPlayerContainer = document.getElementById('audio-player-container');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMessage = document.getElementById('loading-message');
    const loadingStatusText = document.getElementById('loading-status');
    const loadingCancelBtn = document.getElementById('loading-cancel-btn');
    const chunkWarningModal = document.getElementById('chunk-warning-modal');
    const chunkWarningOkBtn = document.getElementById('chunk-warning-ok');
    const chunkWarningCancelBtn = document.getElementById('chunk-warning-cancel');
    const hideChunkWarningCheckbox = document.getElementById('hide-chunk-warning-checkbox');
    const generationWarningModal = document.getElementById('generation-warning-modal');
    const generationWarningAcknowledgeBtn = document.getElementById('generation-warning-acknowledge');
    const hideGenerationWarningCheckbox = document.getElementById('hide-generation-warning-checkbox');

    // --- Utility Functions ---
    function showNotification(message, type = 'info', duration = 5000) {
        if (!notificationArea) return null;
        const icons = {
            success: '<svg class="notification-icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" /></svg>',
            error: '<svg class="notification-icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" /></svg>',
            warning: '<svg class="notification-icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clip-rule="evenodd" /></svg>',
            info: '<svg class="notification-icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clip-rule="evenodd" /></svg>'
        };
        const typeClassMap = { success: 'notification-success', error: 'notification-error', warning: 'notification-warning', info: 'notification-info' };
        const notificationDiv = document.createElement('div');
        notificationDiv.className = typeClassMap[type] || 'notification-info';
        notificationDiv.setAttribute('role', 'alert');
        notificationDiv.innerHTML = `${icons[type] || icons['info']} <span class="block sm:inline">${message}</span>`;
        const closeButton = document.createElement('button');
        closeButton.type = 'button';
        closeButton.className = 'ml-auto -mx-1.5 -my-1.5 bg-transparent rounded-lg p-1.5 inline-flex h-8 w-8 items-center justify-center text-current hover:bg-slate-200 dark:hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-slate-400';
        closeButton.innerHTML = '<span class="sr-only">Close</span><svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path></svg>';
        closeButton.onclick = () => {
            notificationDiv.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            notificationDiv.style.opacity = '0';
            notificationDiv.style.transform = 'translateY(-20px)';
            setTimeout(() => notificationDiv.remove(), 300);
        };
        notificationDiv.appendChild(closeButton);
        notificationArea.appendChild(notificationDiv);
        if (duration > 0) setTimeout(() => closeButton.click(), duration);
        return notificationDiv;
    }

    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
        return `${minutes}:${secs}`;
    }

    // --- Theme Management ---
    function applyTheme(theme) {
        const isDark = theme === 'dark';
        document.documentElement.classList.toggle('dark', isDark);
        if (themeSwitchThumb) {
            themeSwitchThumb.classList.toggle('translate-x-6', isDark);
            themeSwitchThumb.classList.toggle('bg-indigo-500', isDark);
            themeSwitchThumb.classList.toggle('bg-white', !isDark);
        }
        if (wavesurfer) {
            wavesurfer.setOptions({
                waveColor: isDark ? '#6366f1' : '#a5b4fc',
                progressColor: isDark ? '#4f46e5' : '#6366f1',
                cursorColor: isDark ? '#cbd5e1' : '#475569',
            });
        }
        localStorage.setItem('uiTheme', theme);
    }

    if (themeToggleButton) {
        themeToggleButton.addEventListener('click', () => {
            const newTheme = document.documentElement.classList.contains('dark') ? 'light' : 'dark';
            applyTheme(newTheme);
            debouncedSaveState();
        });
    }

    // Logout functionality
    if (logoutButton) {
        logoutButton.addEventListener('click', () => {
            localStorage.removeItem('authToken');
            window.location.reload();
        });
    }

    // --- UI State Persistence ---
    async function saveCurrentUiState() {
        const stateToSave = {
            last_text: textArea ? textArea.value : '',
            last_voice_mode: currentVoiceMode,
            last_predefined_voice: predefinedVoiceSelect ? predefinedVoiceSelect.value : null,
            last_reference_file: cloneReferenceSelect ? cloneReferenceSelect.value : null,
            last_seed: seedInput ? parseInt(seedInput.value, 10) || 0 : 0,
            last_chunk_size: chunkSizeSlider ? parseInt(chunkSizeSlider.value, 10) : 120,
            last_split_text_enabled: splitTextToggle ? splitTextToggle.checked : true,
            hide_chunk_warning: hideChunkWarning,
            hide_generation_warning: hideGenerationWarning,
            theme: localStorage.getItem('uiTheme') || 'dark'
        };
        try {
            const token = localStorage.getItem('authToken');
            const headers = { 'Content-Type': 'application/json' };
            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
            }
            
            const response = await fetch('/save_settings', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({ ui_state: stateToSave })
            });
            if (!response.ok) {
                const errorResult = await response.json();
                throw new Error(errorResult.detail || `Failed to save UI state (status ${response.status})`);
            }
        } catch (error) {
            console.error("Error saving UI state via API:", error);
            showNotification(`Error saving settings: ${error.message}. Some changes may not persist.`, 'error', 0);
        }
    }

    function debouncedSaveState() {
        clearTimeout(saveStateTimeout);
        saveStateTimeout = setTimeout(saveCurrentUiState, DEBOUNCE_DELAY_MS);
    }

    // --- Speed Factor Warning ---
    function updateSpeedFactorWarning() {
        if (speedFactorSlider && speedFactorWarningSpan) {
            const value = parseFloat(speedFactorSlider.value);
            if (value !== 1.0) {
                speedFactorWarningSpan.textContent = "* Experimental, may cause echo.";
                speedFactorWarningSpan.classList.remove('hidden');
            } else {
                speedFactorWarningSpan.classList.add('hidden');
            }
        }
    }


    // --- Initial Application Setup ---
    function initializeApplication() {
        const preferredTheme = localStorage.getItem('uiTheme') || currentUiState.theme || 'dark';
        applyTheme(preferredTheme);
        const pageTitle = currentConfig?.ui?.title || "TTS Server";
        document.title = pageTitle;
        if (appTitleLink) appTitleLink.textContent = pageTitle;
        if (ttsFormHeader) ttsFormHeader.textContent = `Generate Speech`;
        loadInitialUiState();
        populatePredefinedVoices();
        populateReferenceFiles();
        populatePresets();
        displayServerConfiguration();
        if (languageSelectContainer && currentConfig?.ui?.show_language_select === false) {
            languageSelectContainer.classList.add('hidden');
        }
        updateSpeedFactorWarning(); // Initial check for speed factor warning
        attachStateSavingListeners();
        const initialGenResult = currentConfig.initial_gen_result;
        if (initialGenResult && initialGenResult.outputUrl) {
            initializeWaveSurfer(initialGenResult.outputUrl, initialGenResult);
        }
    }

    async function fetchInitialData() {
        try {
            const token = localStorage.getItem('authToken');
            const headers = {};
            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
            }
            
            const response = await fetch('/api/ui/initial-data', { headers });
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to fetch initial UI data: ${response.status} ${response.statusText}. Server response: ${errorText}`);
            }
            const data = await response.json();
            currentConfig = data.config || {};
            currentUiState = currentConfig.ui_state || {};
            appPresets = data.presets || [];
            initialReferenceFiles = data.reference_files || [];
            initialPredefinedVoices = data.predefined_voices || [];
            hideChunkWarning = currentUiState.hide_chunk_warning || false;
            hideGenerationWarning = currentUiState.hide_generation_warning || false;
            currentVoiceMode = currentUiState.last_voice_mode || 'predefined';
            initializeApplication();
        } catch (error) {
            console.error("Error fetching initial data:", error);
            showNotification(`Could not load essential application data: ${error.message}. Please try refreshing.`, 'error', 0);
            if (Object.keys(currentConfig).length === 0) {
                currentConfig = { ui: { title: "TTS Server (Error Mode)" }, generation_defaults: {}, ui_state: {} };
                currentUiState = currentConfig.ui_state;
            }
            initializeApplication();
        }
    }

    function loadInitialUiState() {
        if (textArea && currentUiState.last_text) {
            textArea.value = currentUiState.last_text;
            if (charCount) charCount.textContent = textArea.value.length;
        }
        const modeRadioToSelect = document.querySelector(`input[name="voice_mode"][value="${currentVoiceMode}"]`);
        if (modeRadioToSelect) modeRadioToSelect.checked = true;
        else {
            document.querySelector('input[name="voice_mode"][value="predefined"]').checked = true;
            currentVoiceMode = 'predefined';
        }
        toggleVoiceOptionsDisplay();
        if (seedInput && currentUiState.last_seed !== undefined) seedInput.value = currentUiState.last_seed;
        else if (seedInput && currentConfig?.generation_defaults?.seed !== undefined) seedInput.value = currentConfig.generation_defaults.seed;
        if (splitTextToggle) splitTextToggle.checked = currentUiState.last_split_text_enabled !== undefined ? currentUiState.last_split_text_enabled : true;
        if (chunkSizeSlider && currentUiState.last_chunk_size !== undefined) chunkSizeSlider.value = currentUiState.last_chunk_size;
        if (chunkSizeValue) chunkSizeValue.textContent = chunkSizeSlider ? chunkSizeSlider.value : '120';
        toggleChunkControlsVisibility();
        const genDefaults = currentConfig.generation_defaults || {};
        if (temperatureSlider) temperatureSlider.value = genDefaults.temperature !== undefined ? genDefaults.temperature : 0.8;
        if (temperatureValueDisplay) temperatureValueDisplay.textContent = temperatureSlider.value;
        if (exaggerationSlider) exaggerationSlider.value = genDefaults.exaggeration !== undefined ? genDefaults.exaggeration : 0.5;
        if (exaggerationValueDisplay) exaggerationValueDisplay.textContent = exaggerationSlider.value;
        if (cfgWeightSlider) cfgWeightSlider.value = genDefaults.cfg_weight !== undefined ? genDefaults.cfg_weight : 0.5;
        if (cfgWeightValueDisplay) cfgWeightValueDisplay.textContent = cfgWeightSlider.value;
        if (speedFactorSlider) speedFactorSlider.value = genDefaults.speed_factor !== undefined ? genDefaults.speed_factor : 1.0;
        if (speedFactorValueDisplay) speedFactorValueDisplay.textContent = speedFactorSlider.value;
        if (languageSelect) languageSelect.value = genDefaults.language || 'en';
        if (hideChunkWarningCheckbox) hideChunkWarningCheckbox.checked = hideChunkWarning;
        if (hideGenerationWarningCheckbox) hideGenerationWarningCheckbox.checked = hideGenerationWarning;
        if (textArea && !textArea.value && appPresets && appPresets.length > 0) {
            const defaultPreset = appPresets.find(p => p.name === "Standard Narration") || appPresets[0];
            if (defaultPreset) applyPreset(defaultPreset, false);
        }
    }

    function attachStateSavingListeners() {
        if (textArea) textArea.addEventListener('input', () => { if (charCount) charCount.textContent = textArea.value.length; debouncedSaveState(); });
        if (predefinedVoiceSelect) predefinedVoiceSelect.addEventListener('change', debouncedSaveState);
        if (cloneReferenceSelect) cloneReferenceSelect.addEventListener('change', debouncedSaveState);
        if (seedInput) seedInput.addEventListener('change', debouncedSaveState);
        if (splitTextToggle) splitTextToggle.addEventListener('change', () => { toggleChunkControlsVisibility(); debouncedSaveState(); });
        if (chunkSizeSlider) {
            chunkSizeSlider.addEventListener('input', () => { if (chunkSizeValue) chunkSizeValue.textContent = chunkSizeSlider.value; });
            chunkSizeSlider.addEventListener('change', debouncedSaveState);
        }
        const genParamSliders = [temperatureSlider, exaggerationSlider, cfgWeightSlider, speedFactorSlider];
        genParamSliders.forEach(slider => {
            if (slider) {
                const valueDisplayId = slider.id + '-value';
                const valueDisplay = document.getElementById(valueDisplayId);
                slider.addEventListener('input', () => {
                    if (valueDisplay) valueDisplay.textContent = slider.value;
                    if (slider.id === 'speed-factor') updateSpeedFactorWarning(); // Update warning on input
                });
                slider.addEventListener('change', debouncedSaveState);
            }
        });
        if (languageSelect) languageSelect.addEventListener('change', debouncedSaveState);
    }

    // --- Dynamic UI Population ---
    function populatePredefinedVoices(voicesData = initialPredefinedVoices) {
        if (!predefinedVoiceSelect) return;
        const currentSelectedValue = predefinedVoiceSelect.value;
        predefinedVoiceSelect.innerHTML = '<option value="none">-- Select Voice --</option>';
        voicesData.forEach(voice => {
            const option = document.createElement('option');
            option.value = voice.filename;
            option.textContent = voice.display_name || voice.filename;
            predefinedVoiceSelect.appendChild(option);
        });
        const lastSelected = currentUiState.last_predefined_voice;
        const defaultFromConfig = currentConfig?.tts_engine?.default_voice_id;
        if (currentSelectedValue !== 'none' && voicesData.some(v => v.filename === currentSelectedValue)) {
            predefinedVoiceSelect.value = currentSelectedValue;
        } else if (lastSelected && voicesData.some(v => v.filename === lastSelected)) {
            predefinedVoiceSelect.value = lastSelected;
        } else if (defaultFromConfig && voicesData.some(v => v.filename === defaultFromConfig)) {
            predefinedVoiceSelect.value = defaultFromConfig;
        } else {
            predefinedVoiceSelect.value = 'none';
        }
    }

    function populateReferenceFiles(filesData = initialReferenceFiles) {
        if (!cloneReferenceSelect) return;
        const currentSelectedValue = cloneReferenceSelect.value;
        cloneReferenceSelect.innerHTML = '<option value="none">-- Select Reference File --</option>';
        filesData.forEach(filename => {
            const option = document.createElement('option');
            option.value = filename;
            option.textContent = filename;
            cloneReferenceSelect.appendChild(option);
        });
        const lastSelected = currentUiState.last_reference_file;
        if (currentSelectedValue !== 'none' && filesData.includes(currentSelectedValue)) {
            cloneReferenceSelect.value = currentSelectedValue;
        } else if (lastSelected && filesData.includes(lastSelected)) {
            cloneReferenceSelect.value = lastSelected;
        } else {
            cloneReferenceSelect.value = 'none';
        }
    }

    function populatePresets() {
        if (!presetsContainer || !appPresets) return;
        if (appPresets.length === 0) {
            if (presetsPlaceholder) presetsPlaceholder.textContent = 'No presets available.';
            return;
        }
        if (presetsPlaceholder) presetsPlaceholder.remove();
        presetsContainer.innerHTML = '';
        appPresets.forEach((preset, index) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.id = `preset-btn-${index}`;
            button.className = 'preset-button';
            button.title = `Load '${preset.name}' text and settings`;
            button.textContent = preset.name;
            button.addEventListener('click', () => applyPreset(preset));
            presetsContainer.appendChild(button);
        });
    }

    function applyPreset(presetData, showNotif = true) {
        if (!presetData) return;
        if (textArea && presetData.text !== undefined) {
            textArea.value = presetData.text;
            if (charCount) charCount.textContent = textArea.value.length;
        }
        const genParams = presetData.params || presetData;
        if (temperatureSlider && genParams.temperature !== undefined) temperatureSlider.value = genParams.temperature;
        if (exaggerationSlider && genParams.exaggeration !== undefined) exaggerationSlider.value = genParams.exaggeration;
        if (cfgWeightSlider && genParams.cfg_weight !== undefined) cfgWeightSlider.value = genParams.cfg_weight;
        if (speedFactorSlider && genParams.speed_factor !== undefined) speedFactorSlider.value = genParams.speed_factor;
        if (seedInput && genParams.seed !== undefined) seedInput.value = genParams.seed;
        if (languageSelect && genParams.language !== undefined) languageSelect.value = genParams.language;
        if (temperatureValueDisplay && temperatureSlider) temperatureValueDisplay.textContent = temperatureSlider.value;
        if (exaggerationValueDisplay && exaggerationSlider) exaggerationValueDisplay.textContent = exaggerationSlider.value;
        if (cfgWeightValueDisplay && cfgWeightSlider) cfgWeightValueDisplay.textContent = cfgWeightSlider.value;
        if (speedFactorValueDisplay && speedFactorSlider) speedFactorValueDisplay.textContent = speedFactorSlider.value;
        updateSpeedFactorWarning(); // Update warning after applying preset
        if (genParams.voice_id && predefinedVoiceSelect) {
            const voiceExists = Array.from(predefinedVoiceSelect.options).some(opt => opt.value === genParams.voice_id);
            if (voiceExists) {
                predefinedVoiceSelect.value = genParams.voice_id;
                document.querySelector('input[name="voice_mode"][value="predefined"]').click();
            }
        } else if (genParams.reference_audio_filename && cloneReferenceSelect) {
            const refExists = Array.from(cloneReferenceSelect.options).some(opt => opt.value === genParams.reference_audio_filename);
            if (refExists) {
                cloneReferenceSelect.value = genParams.reference_audio_filename;
                document.querySelector('input[name="voice_mode"][value="clone"]').click();
            }
        }
        if (showNotif) showNotification(`Preset "${presetData.name}" loaded.`, 'info', 3000);
        debouncedSaveState();
    }

    // --- Voice Mode and Options Visibility ---
    function toggleVoiceOptionsDisplay() {
        const selectedMode = document.querySelector('input[name="voice_mode"]:checked')?.value;
        currentVoiceMode = selectedMode;
        if (predefinedVoiceOptionsDiv) predefinedVoiceOptionsDiv.classList.toggle('hidden', selectedMode !== 'predefined');
        if (cloneOptionsDiv) cloneOptionsDiv.classList.toggle('hidden', selectedMode !== 'clone');
        if (predefinedVoiceSelect) predefinedVoiceSelect.required = (selectedMode === 'predefined');
        if (cloneReferenceSelect) cloneReferenceSelect.required = (selectedMode === 'clone');
        debouncedSaveState();
    }
    voiceModeRadios.forEach(radio => radio.addEventListener('change', toggleVoiceOptionsDisplay));

    function toggleChunkControlsVisibility() {
        const isChecked = splitTextToggle ? splitTextToggle.checked : false;
        if (chunkSizeControls) chunkSizeControls.classList.toggle('hidden', !isChecked);
        if (chunkExplanation) chunkExplanation.classList.toggle('hidden', !isChecked);
    }
    if (splitTextToggle) toggleChunkControlsVisibility();

    // --- Audio Player (WaveSurfer) ---
    function initializeWaveSurfer(audioUrl, resultDetails = {}) {
        if (wavesurfer) {
            wavesurfer.unAll(); // Remove all event listeners before destroying
            wavesurfer.destroy();
            wavesurfer = null;
        }
        if (currentAudioBlobUrl) {
            URL.revokeObjectURL(currentAudioBlobUrl);
            currentAudioBlobUrl = null;
        }
        currentAudioBlobUrl = audioUrl;

        // Ensure the container is clean or re-created
        audioPlayerContainer.innerHTML = `
            <div class="audio-player-card">
                <div class="p-6 sm:p-8">
                    <h2 class="card-header">Generated Audio</h2>
                    <div class="mb-5"><div id="waveform" class="waveform-container"></div></div>
                    <div class="audio-player-controls">
                        <div class="audio-player-buttons">
                            <button id="play-btn" class="btn-primary" disabled>
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5 mr-1.5"><path fill-rule="evenodd" d="M2 10a8 8 0 1 1 16 0 8 8 0 0 1-16 0Zm6.39-2.908a.75.75 0 0 1 .766.027l3.5 2.25a.75.75 0 0 1 0 1.262l-3.5 2.25A.75.75 0 0 1 8 12.25v-4.5a.75.75 0 0 1 .39-.658Z" clip-rule="evenodd" /></svg> Play
                            </button>
                            <a id="download-link" href="#" download="tts_output.wav" class="btn-secondary opacity-50 pointer-events-none">
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5 mr-1.5"><path d="M10.75 2.75a.75.75 0 0 0-1.5 0v8.614L6.295 8.235a.75.75 0 1 0-1.09 1.03l4.25 4.5a.75.75 0 0 0 1.09 0l4.25 4.5a.75.75 0 0 0-1.09-1.03l-2.955 3.129V2.75Z" /><path d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z" /></svg> Download
                            </a>
                        </div>
                        <div class="audio-player-info text-xs sm:text-sm">
                            Mode: <span id="player-voice-mode" class="font-medium text-indigo-600 dark:text-indigo-400">--</span>
                            <span id="player-voice-file-details"></span>
                            <span class="mx-1">•</span> Gen Time: <span id="player-gen-time" class="font-medium tabular-nums">--s</span>
                            <span class="mx-1">•</span> Duration: <span id="audio-duration" class="font-medium tabular-nums">--:--</span>
                        </div>
                    </div>
                </div>
            </div>`;

        // Re-select elements after recreating them
        const waveformDiv = audioPlayerContainer.querySelector('#waveform');
        const playBtn = audioPlayerContainer.querySelector('#play-btn'); // Crucial: re-select after innerHTML change
        const downloadLink = audioPlayerContainer.querySelector('#download-link');
        const playerModeSpan = audioPlayerContainer.querySelector('#player-voice-mode');
        const playerFileSpan = audioPlayerContainer.querySelector('#player-voice-file-details');
        const playerGenTimeSpan = audioPlayerContainer.querySelector('#player-gen-time');
        const audioDurationSpan = audioPlayerContainer.querySelector('#audio-duration');

        const audioFilename = resultDetails.filename || (typeof audioUrl === 'string' ? audioUrl.split('/').pop() : 'tts_output.wav');
        if (downloadLink) {
            downloadLink.href = audioUrl;
            downloadLink.download = audioFilename;
            downloadLink.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5 mr-1.5"><path d="M10.75 2.75a.75.75 0 0 0-1.5 0v8.614L6.295 8.235a.75.75 0 1 0-1.09 1.03l4.25 4.5a.75.75 0 0 0 1.09 0l4.25 4.5a.75.75 0 0 0-1.09-1.03l-2.955 3.129V2.75Z" /><path d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z" /></svg> Download ${audioFilename.split('.').pop().toUpperCase()}`;
        }
        if (playerModeSpan) playerModeSpan.textContent = resultDetails.submittedVoiceMode || currentVoiceMode || '--';
        if (playerFileSpan) {
            let fileDetail = '';
            if ((resultDetails.submittedVoiceMode || currentVoiceMode) === 'clone' && resultDetails.submittedCloneFile) {
                fileDetail = `(<span class="font-medium text-slate-700 dark:text-slate-300">${resultDetails.submittedCloneFile}</span>)`;
            } else if ((resultDetails.submittedVoiceMode || currentVoiceMode) === 'predefined' && resultDetails.submittedPredefinedVoice) {
                fileDetail = `(<span class="font-medium text-slate-700 dark:text-slate-300">${resultDetails.submittedPredefinedVoice}</span>)`;
            }
            playerFileSpan.innerHTML = fileDetail;
        }
        if (playerGenTimeSpan) playerGenTimeSpan.textContent = resultDetails.genTime ? `${resultDetails.genTime}s` : '--s';

        const playIconSVG = playBtn.innerHTML; // Capture after it's created
        const pauseIconSVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5 mr-1.5"><path fill-rule="evenodd" d="M2 10a8 8 0 1 1 16 0 8 8 0 0 1-16 0Zm5-2.25A.75.75 0 0 1 7.75 7h4.5a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-.75.75h-4.5a.75.75 0 0 1-.75-.75v-4.5Z" clip-rule="evenodd" /></svg> Pause`;
        const isDark = document.documentElement.classList.contains('dark');

        wavesurfer = WaveSurfer.create({
            container: waveformDiv, waveColor: isDark ? '#6366f1' : '#a5b4fc', progressColor: isDark ? '#4f46e5' : '#6366f1',
            cursorColor: isDark ? '#cbd5e1' : '#475569', barWidth: 3, barRadius: 3, cursorWidth: 1, height: 80, barGap: 2,
            responsive: true, url: audioUrl, mediaControls: false, normalize: true,
        });

        wavesurfer.on('ready', () => {
            const duration = wavesurfer.getDuration();
            if (audioDurationSpan) audioDurationSpan.textContent = formatTime(duration);
            if (playBtn) { playBtn.disabled = false; playBtn.innerHTML = playIconSVG; }
            if (downloadLink) { downloadLink.classList.remove('opacity-50', 'pointer-events-none'); downloadLink.setAttribute('aria-disabled', 'false'); }
        });
        wavesurfer.on('play', () => { if (playBtn) playBtn.innerHTML = pauseIconSVG; });
        wavesurfer.on('pause', () => { if (playBtn) playBtn.innerHTML = playIconSVG; });
        wavesurfer.on('finish', () => { if (playBtn) playBtn.innerHTML = playIconSVG; wavesurfer.seekTo(0); });
        wavesurfer.on('error', (err) => {
            console.error("WaveSurfer error:", err);
            showNotification(`Error loading audio waveform: ${err.message || err}`, 'error');
            if (waveformDiv) waveformDiv.innerHTML = `<p class="p-4 text-sm text-red-600 dark:text-red-400">Could not load waveform.</p>`;
            if (playBtn) playBtn.disabled = true;
        });

        // This is crucial: ensure playBtn refers to the newly created button in the DOM
        if (playBtn) {
            playBtn.onclick = () => {
                if (wavesurfer) {
                    wavesurfer.playPause();
                }
            };
        }
        setTimeout(() => audioPlayerContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 150);
    }


    // --- TTS Generation Logic ---
    function getTTSFormData() {
        const jsonData = {
            text: textArea.value,
            temperature: parseFloat(temperatureSlider.value),
            exaggeration: parseFloat(exaggerationSlider.value),
            cfg_weight: parseFloat(cfgWeightSlider.value),
            speed_factor: parseFloat(speedFactorSlider.value),
            seed: parseInt(seedInput.value, 10),
            language: languageSelect.value,
            voice_mode: currentVoiceMode,
            split_text: splitTextToggle.checked,
            chunk_size: parseInt(chunkSizeSlider.value, 10),
            output_format: currentConfig?.audio_output?.format || 'wav'
        };
        if (currentVoiceMode === 'predefined' && predefinedVoiceSelect.value !== 'none') {
            jsonData.predefined_voice_id = predefinedVoiceSelect.value;
        } else if (currentVoiceMode === 'clone' && cloneReferenceSelect.value !== 'none') {
            jsonData.reference_audio_filename = cloneReferenceSelect.value;
        }
        return jsonData;
    }

    async function submitTTSRequest() {
        isGenerating = true;
        showLoadingOverlay();
        const startTime = performance.now();
        const jsonData = getTTSFormData();
        try {
            const token = localStorage.getItem('authToken');
            const headers = { 'Content-Type': 'application/json' };
            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
            }
            
            const response = await fetch('/tts', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(jsonData)
            });
            if (!response.ok) {
                const errorResult = await response.json().catch(() => ({ detail: `HTTP error ${response.status}` }));
                throw new Error(errorResult.detail || 'TTS generation failed.');
            }
            const audioBlob = await response.blob();
            const endTime = performance.now();
            const genTime = ((endTime - startTime) / 1000).toFixed(2);
            const filenameFromServer = response.headers.get('Content-Disposition')?.split('filename=')[1]?.replace(/"/g, '') || 'generated_audio.wav';
            const resultDetails = {
                outputUrl: URL.createObjectURL(audioBlob), filename: filenameFromServer, genTime: genTime,
                submittedVoiceMode: jsonData.voice_mode, submittedPredefinedVoice: jsonData.predefined_voice_id,
                submittedCloneFile: jsonData.reference_audio_filename
            };
            initializeWaveSurfer(resultDetails.outputUrl, resultDetails);
            showNotification('Audio generated successfully!', 'success');
        } catch (error) {
            console.error('TTS Generation Error:', error);
            showNotification(error.message || 'An unknown error occurred during TTS generation.', 'error');
        } finally {
            isGenerating = false;
            hideLoadingOverlay();
        }
    }

    function proceedWithSubmissionChecks() {
        const textContent = textArea.value.trim();
        const isSplittingEnabled = splitTextToggle.checked;
        const currentChunkSz = parseInt(chunkSizeSlider.value, 10);
        const needsChunkWarn = isSplittingEnabled && textContent.length >= currentChunkSz * 1.5 &&
            currentVoiceMode !== 'predefined' && currentVoiceMode !== 'clone' &&
            (!seedInput || parseInt(seedInput.value, 10) === 0 || seedInput.value === '') && !hideChunkWarning;
        if (needsChunkWarn) { showChunkWarningModal(); return; }
        submitTTSRequest();
    }

    if (ttsForm) {
        ttsForm.addEventListener('submit', function (event) {
            event.preventDefault();
            if (isGenerating) { showNotification("Generation is already in progress.", "warning"); return; }
            const textContent = textArea.value.trim();
            if (!textContent) { showNotification("Please enter some text to generate speech.", 'error'); return; }
            if (currentVoiceMode === 'predefined' && (!predefinedVoiceSelect || predefinedVoiceSelect.value === 'none')) {
                showNotification("Please select a predefined voice.", 'error'); return;
            }
            if (currentVoiceMode === 'clone' && (!cloneReferenceSelect || cloneReferenceSelect.value === 'none')) {
                showNotification("Please select a reference audio file for Voice Cloning.", 'error'); return;
            }
            if (!hideGenerationWarning) { showGenerationWarningModal(); return; }
            proceedWithSubmissionChecks();
        });
    }

    // --- Modal Handling ---
    function showChunkWarningModal() { if (chunkWarningModal) { chunkWarningModal.classList.remove('hidden', 'opacity-0'); chunkWarningModal.dataset.state = 'open'; } }
    function hideChunkWarningModal() { if (chunkWarningModal) { chunkWarningModal.classList.add('opacity-0'); setTimeout(() => { chunkWarningModal.classList.add('hidden'); chunkWarningModal.dataset.state = 'closed'; }, 300); } }
    function showGenerationWarningModal() { if (generationWarningModal) { generationWarningModal.classList.remove('hidden', 'opacity-0'); generationWarningModal.dataset.state = 'open'; } }
    function hideGenerationWarningModal() { if (generationWarningModal) { generationWarningModal.classList.add('opacity-0'); setTimeout(() => { generationWarningModal.classList.add('hidden'); generationWarningModal.dataset.state = 'closed'; }, 300); } }
    if (chunkWarningOkBtn) chunkWarningOkBtn.addEventListener('click', () => {
        if (hideChunkWarningCheckbox && hideChunkWarningCheckbox.checked) hideChunkWarning = true;
        hideChunkWarningModal(); debouncedSaveState(); submitTTSRequest();
    });
    if (chunkWarningCancelBtn) chunkWarningCancelBtn.addEventListener('click', hideChunkWarningModal);
    if (generationWarningAcknowledgeBtn) generationWarningAcknowledgeBtn.addEventListener('click', () => {
        if (hideGenerationWarningCheckbox && hideGenerationWarningCheckbox.checked) hideGenerationWarning = true;
        hideGenerationWarningModal(); debouncedSaveState(); proceedWithSubmissionChecks();
    });
    if (loadingCancelBtn) loadingCancelBtn.addEventListener('click', () => {
        if (isGenerating) { isGenerating = false; hideLoadingOverlay(); showNotification("Generation UI cancelled by user.", "info"); }
    });
    function showLoadingOverlay() {
        if (loadingOverlay && generateBtn && loadingCancelBtn) {
            loadingMessage.textContent = 'Generating audio...';
            loadingStatusText.textContent = 'Please wait. This may take some time.';
            loadingOverlay.classList.remove('hidden', 'opacity-0'); loadingOverlay.dataset.state = 'open';
            generateBtn.disabled = true; loadingCancelBtn.disabled = false;
        }
    }
    function hideLoadingOverlay() {
        if (loadingOverlay && generateBtn) {
            loadingOverlay.classList.add('opacity-0');
            setTimeout(() => { loadingOverlay.classList.add('hidden'); loadingOverlay.dataset.state = 'closed'; }, 300);
            generateBtn.disabled = false;
        }
    }

    // --- Configuration Management ---
    function displayServerConfiguration() {
        if (!serverConfigForm || !currentConfig || Object.keys(currentConfig).length === 0) return;
        const fieldsToDisplay = {
            "server.host": currentConfig.server?.host, "server.port": currentConfig.server?.port,
            "tts_engine.device": currentConfig.tts_engine?.device, "tts_engine.default_voice_id": currentConfig.tts_engine?.default_voice_id,
            "paths.model_cache": currentConfig.paths?.model_cache, "tts_engine.predefined_voices_path": currentConfig.tts_engine?.predefined_voices_path,
            "tts_engine.reference_audio_path": currentConfig.tts_engine?.reference_audio_path, "paths.output": currentConfig.paths?.output,
            "audio_output.format": currentConfig.audio_output?.format, "audio_output.sample_rate": currentConfig.audio_output?.sample_rate
        };
        for (const name in fieldsToDisplay) {
            const input = serverConfigForm.querySelector(`input[name="${name}"]`);
            if (input) {
                input.value = fieldsToDisplay[name] !== undefined ? fieldsToDisplay[name] : '';
                if (name.includes('.host') || name.includes('.port') || name.includes('.device') || name.includes('paths.')) input.readOnly = true;
                else input.readOnly = false;
            }
        }
    }
    async function updateConfigStatus(button, statusElem, message, type = 'info', duration = 5000, enableButtonAfter = true) {
        const statusClasses = { success: 'text-green-600 dark:text-green-400', error: 'text-red-600 dark:text-red-400', warning: 'text-yellow-600 dark:text-yellow-400', info: 'text-indigo-600 dark:text-indigo-400', processing: 'text-yellow-600 dark:text-yellow-400 animate-pulse' };
        const isProcessing = message.toLowerCase().includes('saving') || message.toLowerCase().includes('restarting') || message.toLowerCase().includes('resetting');
        const messageType = isProcessing ? 'processing' : type;
        if (statusElem) {
            statusElem.textContent = message;
            statusElem.className = `text-xs ml-2 ${statusClasses[messageType] || statusClasses['info']}`;
            statusElem.classList.remove('hidden');
        }
        if (button) button.disabled = isProcessing || (type === 'error' && !enableButtonAfter) || (type === 'success' && !enableButtonAfter);
        if (duration > 0) setTimeout(() => { if (statusElem) statusElem.classList.add('hidden'); if (button && enableButtonAfter) button.disabled = false; }, duration);
        else if (button && enableButtonAfter && !isProcessing) button.disabled = false;
    }

    if (saveConfigBtn && configStatus) {
        saveConfigBtn.addEventListener('click', async () => {
            const configDataToSave = {};
            const inputs = serverConfigForm.querySelectorAll('input[name]:not([readonly]), select[name]:not([readonly])');
            inputs.forEach(input => {
                const keys = input.name.split('.'); let currentLevel = configDataToSave;
                keys.forEach((key, index) => {
                    if (index === keys.length - 1) {
                        let value = input.value;
                        if (input.type === 'number') value = parseFloat(value) || 0;
                        else if (input.type === 'checkbox') value = input.checked;
                        currentLevel[key] = value;
                    } else { currentLevel[key] = currentLevel[key] || {}; currentLevel = currentLevel[key]; }
                });
            });
            if (Object.keys(configDataToSave).length === 0) { showNotification("No editable configuration values to save.", "info"); return; }
            updateConfigStatus(saveConfigBtn, configStatus, 'Saving configuration...', 'info', 0, false);
            try {
                const headers = { 'Content-Type': 'application/json', ...getAuthHeaders() };
                const response = await fetch('/save_settings', { method: 'POST', headers, body: JSON.stringify(configDataToSave) });
                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || 'Failed to save configuration');
                updateConfigStatus(saveConfigBtn, configStatus, result.message || 'Configuration saved.', 'success', 5000);
                if (result.restart_needed && restartServerBtn) restartServerBtn.classList.remove('hidden');
                await fetchInitialData();
                showNotification("Configuration saved. Some changes may require a server restart if prompted.", "success");
            } catch (error) {
                console.error('Error saving server config:', error);
                updateConfigStatus(saveConfigBtn, configStatus, `Error: ${error.message}`, 'error', 0);
            }
        });
    }

    if (saveGenDefaultsBtn && genDefaultsStatus) {
        saveGenDefaultsBtn.addEventListener('click', async () => {
            const genParams = {
                temperature: parseFloat(temperatureSlider.value), exaggeration: parseFloat(exaggerationSlider.value),
                cfg_weight: parseFloat(cfgWeightSlider.value), speed_factor: parseFloat(speedFactorSlider.value),
                seed: parseInt(seedInput.value, 10) || 0, language: languageSelect.value
            };
            updateConfigStatus(saveGenDefaultsBtn, genDefaultsStatus, 'Saving generation defaults...', 'info', 0, false);
            try {
                const headers = { 'Content-Type': 'application/json', ...getAuthHeaders() };
                const response = await fetch('/save_settings', { method: 'POST', headers, body: JSON.stringify({ generation_defaults: genParams }) });
                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || 'Failed to save generation defaults');
                updateConfigStatus(saveGenDefaultsBtn, genDefaultsStatus, result.message || 'Generation defaults saved.', 'success', 5000);
                if (currentConfig.generation_defaults) Object.assign(currentConfig.generation_defaults, genParams);
            } catch (error) {
                console.error('Error saving generation defaults:', error);
                updateConfigStatus(saveGenDefaultsBtn, genDefaultsStatus, `Error: ${error.message}`, 'error', 0);
            }
        });
    }

    if (resetSettingsBtn) {
        resetSettingsBtn.addEventListener('click', async () => {
            if (!confirm("Are you sure you want to reset ALL settings to their initial defaults? This will affect config.yaml and UI preferences. This action cannot be undone.")) return;
            updateConfigStatus(resetSettingsBtn, configStatus, 'Resetting settings...', 'info', 0, false);
            try {
                const headers = getAuthHeaders();
                const response = await fetch('/reset_settings', { method: 'POST', headers });
                if (!response.ok) {
                    const errorResult = await response.json().catch(() => ({ detail: 'Failed to reset settings on server.' }));
                    throw new Error(errorResult.detail);
                }
                const result = await response.json();
                updateConfigStatus(resetSettingsBtn, configStatus, result.message + " Reloading page...", 'success', 0, false);
                setTimeout(() => window.location.reload(true), 2000);
            } catch (error) {
                console.error('Error resetting settings:', error);
                updateConfigStatus(resetSettingsBtn, configStatus, `Reset Error: ${error.message}`, 'error', 0);
                showNotification(`Error resetting settings: ${error.message}`, 'error');
            }
        });
    }

    if (restartServerBtn) {
        restartServerBtn.addEventListener('click', async () => {
            if (!confirm("Are you sure you want to restart the server?")) return;
            updateConfigStatus(restartServerBtn, configStatus, 'Attempting server restart...', 'processing', 0, false);
            try {
                const headers = getAuthHeaders();
                const response = await fetch('/restart_server', { method: 'POST', headers });
                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || 'Server responded with error on restart command');
                showNotification("Server restart initiated. Please wait a moment for the server to come back online, then refresh the page.", "info", 10000);
            } catch (error) {
                showNotification(`Server restart command failed: ${error.message}`, "error");
                updateConfigStatus(restartServerBtn, configStatus, `Restart failed.`, 'error', 5000, true);
            }
        });
    }

    // --- File Upload & Refresh ---
    async function handleFileUpload(fileInput, endpoint, successCallback, buttonToAnimate) {
        const files = fileInput.files;
        if (!files || files.length === 0) return;
        const originalButtonHTML = buttonToAnimate ? buttonToAnimate.innerHTML : '';
        if (buttonToAnimate) {
            buttonToAnimate.innerHTML = `<svg class="animate-spin h-5 w-5 mr-1.5 inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>Uploading...`;
            buttonToAnimate.disabled = true;
        }
        const uploadNotification = showNotification(`Uploading ${files.length} file(s)...`, 'info', 0);
        const formData = new FormData();
        for (const file of files) formData.append('files', file);
        try {
            const headers = getAuthHeaders();
            const response = await fetch(endpoint, { method: 'POST', headers, body: formData });
            const result = await response.json();
            if (uploadNotification) uploadNotification.remove();
            if (!response.ok) throw new Error(result.message || result.detail || `Upload failed with status ${response.status}`);
            if (result.errors && result.errors.length > 0) {
                result.errors.forEach(err => showNotification(`Upload Warning: ${err.filename || 'File'} - ${err.error}`, 'warning', 10000));
            }
            const successfulUploads = result.uploaded_files || [];
            if (successfulUploads.length > 0) {
                showNotification(`Successfully uploaded: ${successfulUploads.join(', ')}`, 'success');
            } else if (!result.errors || result.errors.length === 0) {
                showNotification("Files processed. No new valid files were added or an issue occurred.", 'info');
            }
            successCallback(result);
            debouncedSaveState();
        } catch (error) {
            console.error(`Error uploading to ${endpoint}:`, error);
            if (uploadNotification) uploadNotification.remove();
            showNotification(`Upload Error: ${error.message}`, 'error');
        } finally {
            if (buttonToAnimate) {
                buttonToAnimate.disabled = false;
                buttonToAnimate.innerHTML = originalButtonHTML;
            }
            fileInput.value = '';
        }
    }

    if (cloneImportButton && cloneFileInput) {
        cloneImportButton.addEventListener('click', () => cloneFileInput.click());
        cloneFileInput.addEventListener('change', () => handleFileUpload(cloneFileInput, '/upload_reference', (result) => {
            initialReferenceFiles = result.all_reference_files || [];
            populateReferenceFiles();
            const firstUploaded = result.uploaded_files?.[0];
            if (firstUploaded && cloneReferenceSelect && Array.from(cloneReferenceSelect.options).some(opt => opt.value === firstUploaded)) {
                cloneReferenceSelect.value = firstUploaded;
            }
        }, cloneImportButton));
    }

    if (predefinedVoiceImportButton && predefinedVoiceFileInput) {
        predefinedVoiceImportButton.addEventListener('click', () => predefinedVoiceFileInput.click());
        predefinedVoiceFileInput.addEventListener('change', () => handleFileUpload(predefinedVoiceFileInput, '/upload_predefined_voice', (result) => {
            initialPredefinedVoices = result.all_predefined_voices || [];
            populatePredefinedVoices();
            const firstUploadedFilename = result.uploaded_files?.[0];
            if (firstUploadedFilename && predefinedVoiceSelect && initialPredefinedVoices.some(v => v.filename === firstUploadedFilename)) {
                predefinedVoiceSelect.value = firstUploadedFilename;
            }
        }, predefinedVoiceImportButton));
    }

    if (cloneRefreshButton && cloneReferenceSelect) {
        cloneRefreshButton.addEventListener('click', async () => {
            const originalButtonIcon = cloneRefreshButton.innerHTML;
            cloneRefreshButton.innerHTML = `<svg class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>`;
            cloneRefreshButton.disabled = true;
            try {
                const headers = getAuthHeaders();
                const response = await fetch('/get_reference_files', { headers });
                if (!response.ok) throw new Error('Failed to fetch reference files list');
                const files = await response.json();
                initialReferenceFiles = files;
                populateReferenceFiles();
                showNotification("Reference file list refreshed.", 'info', 2000);
                debouncedSaveState();
            } catch (error) {
                console.error("Error refreshing reference files:", error);
                showNotification(`Error refreshing list: ${error.message}`, 'error');
            } finally {
                cloneRefreshButton.disabled = false;
                cloneRefreshButton.innerHTML = originalButtonIcon;
            }
        });
    }

    if (predefinedVoiceRefreshButton && predefinedVoiceSelect) {
        predefinedVoiceRefreshButton.addEventListener('click', async () => {
            const originalButtonIcon = predefinedVoiceRefreshButton.innerHTML;
            predefinedVoiceRefreshButton.innerHTML = `<svg class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>`;
            predefinedVoiceRefreshButton.disabled = true;
            try {
                const headers = getAuthHeaders();
                const response = await fetch('/get_predefined_voices', { headers });
                if (!response.ok) throw new Error('Failed to fetch predefined voices list');
                const voices = await response.json();
                initialPredefinedVoices = voices;
                populatePredefinedVoices();
                showNotification("Predefined voices list refreshed.", 'info', 2000);
                debouncedSaveState();
            } catch (error) {
                console.error("Error refreshing predefined voices:", error);
                showNotification(`Error refreshing list: ${error.message}`, 'error');
            } finally {
                predefinedVoiceRefreshButton.disabled = false;
                predefinedVoiceRefreshButton.innerHTML = originalButtonIcon;
            }
        });
    }

    await fetchInitialData();
});