document.addEventListener('DOMContentLoaded', () => {
    // Emotion to Emoji Map
    const EMOJI_MAP = {
        'Angry': '😠',
        'Disgust': '🤢',
        'Fear': '😨',
        'Happy': '😊',
        'Neutral': '😐',
        'Sad': '😢',
        'Surprise': '😲'
    };

    // Mode Switcher Tabs
    const tabUpload = document.getElementById('tab-upload');
    const tabWebcam = document.getElementById('tab-webcam');
    const uploadModeSection = document.getElementById('upload-mode-section');
    const webcamModeSection = document.getElementById('webcam-mode-section');

    // Upload Elements
    const dropzone = document.getElementById('dropzone-box');
    const imageInput = document.getElementById('image-input');
    const btnPredict = document.getElementById('btn-predict');
    const btnReset = document.getElementById('btn-reset');
    const uploadPreviewWrapper = document.getElementById('upload-preview-wrapper');
    const uploadPreviewImg = document.getElementById('upload-preview-img');

    // Webcam Elements
    const webcamVideo = document.getElementById('webcam-video');
    const webcamCanvas = document.getElementById('webcam-canvas');
    const camPlaceholder = document.getElementById('cam-placeholder');
    const btnStartCam = document.getElementById('btn-start-cam');
    const btnCaptureCam = document.getElementById('btn-capture-cam');
    const btnStopCam = document.getElementById('btn-stop-cam');

    // Result Display Elements
    const placeholderState = document.getElementById('placeholder-state');
    const loadingState = document.getElementById('loading-state');
    const resultsState = document.getElementById('results-state');

    const loadingPreviewWrapper = document.getElementById('loading-preview-wrapper');
    const loadingPreviewImg = document.getElementById('loading-preview-img');
    const loadingSpinnerOnly = document.getElementById('loading-spinner-only');

    const outputImage = document.getElementById('output-image');
    const emotionEmoji = document.getElementById('emotion-emoji');
    const predictedLabel = document.getElementById('predicted-label');
    const predictedConfidence = document.getElementById('predicted-confidence');
    const barsList = document.getElementById('bars-list');
    const btnPredictAgain = document.getElementById('btn-predict-again');

    let selectedFile = null;
    let cameraStream = null;
    let currentPreviewDataUrl = null;

    // ── Mode Switcher ────────────────────────────────────────────────────────
    tabUpload.addEventListener('click', () => {
        tabUpload.classList.add('active');
        tabWebcam.classList.remove('active');
        uploadModeSection.classList.remove('hidden');
        webcamModeSection.classList.add('hidden');
        stopCamera();
        resetAllResults();
    });

    tabWebcam.addEventListener('click', () => {
        tabWebcam.classList.add('active');
        tabUpload.classList.remove('active');
        webcamModeSection.classList.remove('hidden');
        uploadModeSection.classList.add('hidden');
        resetAllResults();
    });

    // ── File Upload Logic ─────────────────────────────────────────────────────
    dropzone.addEventListener('click', () => imageInput.click());

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
    });

    ['dragenter', 'dragover'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            dropzone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            dropzone.classList.remove('drag-over');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files.length > 0) handleFileSelect(dt.files[0]);
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPG, PNG, WEBP).');
            return;
        }
        selectedFile = file;
        btnPredict.disabled = false;

        // Show image preview locally immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            currentPreviewDataUrl = e.target.result;
            uploadPreviewImg.src = currentPreviewDataUrl;
            uploadPreviewWrapper.classList.remove('hidden');
        };
        reader.readAsDataURL(file);

        dropzone.querySelector('.drop-text').innerHTML = `Selected: <strong>${file.name}</strong>`;
        dropzone.querySelector('.file-info').innerText = `${(file.size / 1024).toFixed(1)} KB`;
    }

    // ── Reset ─────────────────────────────────────────────────────────────────
    function resetAllResults() {
        selectedFile = null;
        currentPreviewDataUrl = null;
        imageInput.value = '';
        btnPredict.disabled = true;

        uploadPreviewWrapper.classList.add('hidden');
        uploadPreviewImg.src = '';

        dropzone.querySelector('.drop-text').innerHTML = `<strong>Click to Browse</strong> or Drag &amp; Drop`;
        dropzone.querySelector('.file-info').innerText = `Supports JPEG, PNG, WEBP · Max 16MB`;

        outputImage.src = '';
        predictedLabel.innerText = 'Neutral';
        predictedConfidence.innerText = '0%';
        barsList.innerHTML = '';

        placeholderState.classList.remove('hidden');
        loadingState.classList.add('hidden');
        resultsState.classList.add('hidden');
    }

    btnReset.addEventListener('click', resetAllResults);
    btnPredictAgain.addEventListener('click', resetAllResults);

    // ── Predict from Uploaded File ───────────────────────────────────────────
    btnPredict.addEventListener('click', async () => {
        if (!selectedFile) return;

        showLoadingState(currentPreviewDataUrl);

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                alert(`Prediction Error: ${data.error || 'Server error'}`);
                hideLoadingState();
                return;
            }

            const imgUrl = data.annotated_b64 || (data.annotated_image_url + '?t=' + Date.now());
            renderResults(imgUrl, data.top_emotion, data.confidence, data.probabilities);

        } catch (err) {
            console.error(err);
            alert('Failed to connect to emotion prediction server.');
            hideLoadingState();
        }
    });

    // ── Live Camera Logic ─────────────────────────────────────────────────────
    btnStartCam.addEventListener('click', async () => {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
            });
            webcamVideo.srcObject = cameraStream;
            camPlaceholder.classList.add('hidden');
            btnCaptureCam.disabled = false;
            btnStartCam.disabled = true;
        } catch (err) {
            console.error(err);
            alert('Could not access camera. Please grant browser camera permissions.');
        }
    });

    btnStopCam.addEventListener('click', stopCamera);

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
            webcamVideo.srcObject = null;
        }
        camPlaceholder.classList.remove('hidden');
        btnCaptureCam.disabled = true;
        btnStartCam.disabled = false;
    }

    btnCaptureCam.addEventListener('click', async () => {
        if (!cameraStream) return;

        // Capture frame onto canvas
        const ctx = webcamCanvas.getContext('2d');
        webcamCanvas.width = webcamVideo.videoWidth || 640;
        webcamCanvas.height = webcamVideo.videoHeight || 480;
        ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);

        const frameBase64 = webcamCanvas.toDataURL('image/jpeg', 0.9);
        showLoadingState(frameBase64);

        try {
            const response = await fetch('/predict_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_data: frameBase64 })
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                alert(`Camera Prediction Error: ${data.error || 'Server error'}`);
                hideLoadingState();
                return;
            }

            renderResults(data.annotated_b64, data.top_emotion, data.confidence, data.probabilities);

        } catch (err) {
            console.error(err);
            alert('Failed to process live camera frame.');
            hideLoadingState();
        }
    });

    // ── Helper UI Functions ──────────────────────────────────────────────────
    function showLoadingState(previewSrc) {
        placeholderState.classList.add('hidden');
        resultsState.classList.add('hidden');
        loadingState.classList.remove('hidden');

        if (previewSrc) {
            loadingPreviewImg.src = previewSrc;
            loadingPreviewWrapper.classList.remove('hidden');
            loadingSpinnerOnly.classList.add('hidden');
        } else {
            loadingPreviewWrapper.classList.add('hidden');
            loadingSpinnerOnly.classList.remove('hidden');
        }
    }

    function hideLoadingState() {
        loadingState.classList.add('hidden');
        placeholderState.classList.remove('hidden');
    }

    function renderResults(imgSource, emotion, confidence, probabilities) {
        outputImage.src = imgSource;
        emotionEmoji.innerText = EMOJI_MAP[emotion] || '🎭';
        predictedLabel.innerText = emotion;
        predictedConfidence.innerText = `${confidence.toFixed(1)}% Confidence`;

        barsList.innerHTML = '';

        // Sort probabilities descending
        const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

        entries.forEach(([em, val]) => {
            const row = document.createElement('div');
            row.className = 'prob-row';
            const icon = EMOJI_MAP[em] || '•';
            row.innerHTML = `
                <div class="prob-meta">
                    <span>${icon} ${em}</span>
                    <span><strong>${val.toFixed(1)}%</strong></span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: ${Math.max(val, 2).toFixed(1)}%;"></div>
                </div>
            `;
            barsList.appendChild(row);
        });

        loadingState.classList.add('hidden');
        resultsState.classList.remove('hidden');
    }
});
