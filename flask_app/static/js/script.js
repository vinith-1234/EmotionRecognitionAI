document.addEventListener('DOMContentLoaded', () => {
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
    
    // Webcam Elements
    const webcamVideo = document.getElementById('webcam-video');
    const webcamCanvas = document.getElementById('webcam-canvas');
    const btnStartCam = document.getElementById('btn-start-cam');
    const btnCaptureCam = document.getElementById('btn-capture-cam');
    const btnStopCam = document.getElementById('btn-stop-cam');
    
    // Result Display Elements
    const placeholderState = document.getElementById('placeholder-state');
    const loadingState = document.getElementById('loading-state');
    const resultsState = document.getElementById('results-state');
    
    const outputImage = document.getElementById('output-image');
    const predictedLabel = document.getElementById('predicted-label');
    const predictedConfidence = document.getElementById('predicted-confidence');
    const barsList = document.getElementById('bars-list');
    
    let selectedFile = null;
    let cameraStream = null;

    // Mode switching
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

    // File Upload logic
    dropzone.addEventListener('click', () => imageInput.click());

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('drag-over');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files.length > 0) handleFileSelect(dt.files[0]);
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPEG, PNG, WEBP).');
            return;
        }
        selectedFile = file;
        btnPredict.disabled = false;
        
        dropzone.querySelector('.drop-text').innerHTML = `Selected File: <strong>${file.name}</strong>`;
        dropzone.querySelector('.file-info').innerText = `${(file.size / 1024).toFixed(1)} KB`;
    }

    // Comprehensive Reset Function
    function resetAllResults() {
        selectedFile = null;
        imageInput.value = '';
        btnPredict.disabled = true;
        
        dropzone.querySelector('.drop-text').innerHTML = `<strong>Click to Browse</strong> or Drag & Drop image here`;
        dropzone.querySelector('.file-info').innerText = `Supports JPEG, PNG, WEBP up to 16MB`;
        
        outputImage.src = '';
        predictedLabel.innerText = '';
        predictedConfidence.innerText = '';
        barsList.innerHTML = '';
        
        placeholderState.classList.remove('hidden');
        loadingState.classList.add('hidden');
        resultsState.classList.add('hidden');
    }

    btnReset.addEventListener('click', resetAllResults);

    btnPredict.addEventListener('click', async () => {
        if (!selectedFile) return;

        placeholderState.classList.add('hidden');
        resultsState.classList.add('hidden');
        loadingState.classList.remove('hidden');

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                alert(`Error: ${data.error}`);
                loadingState.classList.add('hidden');
                placeholderState.classList.remove('hidden');
                return;
            }

            renderResults(data.annotated_image_url + '?t=' + new Date().getTime(), data.top_emotion, data.confidence, data.probabilities);

        } catch (err) {
            console.error(err);
            alert('Failed to connect to emotion prediction server.');
            loadingState.classList.add('hidden');
            placeholderState.classList.remove('hidden');
        }
    });

    // Webcam Live Stream logic
    btnStartCam.addEventListener('click', async () => {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
            });
            webcamVideo.srcObject = cameraStream;
            btnCaptureCam.disabled = false;
            btnStartCam.disabled = true;
        } catch (err) {
            console.error(err);
            alert('Could not access integrated camera. Please grant camera permissions in your browser.');
        }
    });

    btnStopCam.addEventListener('click', stopCamera);

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
            webcamVideo.srcObject = null;
        }
        btnCaptureCam.disabled = true;
        btnStartCam.disabled = false;
    }

    btnCaptureCam.addEventListener('click', async () => {
        if (!cameraStream) return;

        placeholderState.classList.add('hidden');
        resultsState.classList.add('hidden');
        loadingState.classList.remove('hidden');

        const context = webcamCanvas.getContext('2d');
        webcamCanvas.width = webcamVideo.videoWidth || 640;
        webcamCanvas.height = webcamVideo.videoHeight || 480;
        context.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);

        const imageDataBase64 = webcamCanvas.toDataURL('image/jpeg', 0.9);

        try {
            const response = await fetch('/predict_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_data: imageDataBase64 })
            });

            const data = await response.json();

            if (data.error) {
                alert(`Camera Prediction Error: ${data.error}`);
                loadingState.classList.add('hidden');
                placeholderState.classList.remove('hidden');
                return;
            }

            renderResults(data.annotated_b64, data.top_emotion, data.confidence, data.probabilities);

        } catch (err) {
            console.error(err);
            alert('Failed to process live camera frame.');
            loadingState.classList.add('hidden');
            placeholderState.classList.remove('hidden');
        }
    });

    function renderResults(imgSource, emotion, confidence, probabilities) {
        outputImage.src = imgSource;
        predictedLabel.innerText = emotion;
        predictedConfidence.innerText = `${confidence.toFixed(1)}% Confidence`;
        
        barsList.innerHTML = '';
        const sortedEntries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
        
        sortedEntries.forEach(([em, val]) => {
            const row = document.createElement('div');
            row.className = 'prob-row';
            row.innerHTML = `
                <div class="prob-meta">
                    <span>${em}</span>
                    <span>${val.toFixed(1)}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: ${val.toFixed(1)}%;"></div>
                </div>
            `;
            barsList.appendChild(row);
        });

        loadingState.classList.add('hidden');
        resultsState.classList.remove('hidden');
    }
});
