document.addEventListener('DOMContentLoaded', () => {
    const modelSelect = document.getElementById('modelSelect');
    const scenarioContainer = document.getElementById('scenarioContainer');
    const uploadBox = document.getElementById('uploadBox');
    const audioUpload = document.getElementById('audioUpload');
    const recordBtn = document.getElementById('recordBtn');
    const recordStatus = document.getElementById('recordStatus');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    // Result UI Elements
    const statusText = document.querySelector('.status-text');
    const loader = document.getElementById('loader');
    const predictionResult = document.getElementById('predictionResult');
    const genderIcon = document.getElementById('genderIcon');
    const genderText = document.getElementById('genderText');
    const modelUsed = document.getElementById('modelUsed');
    const scenarioUsed = document.getElementById('scenarioUsed');
    const audioPlayerContainer = document.getElementById('audioPlayerContainer');
    const audioPlayback = document.getElementById('audioPlayback');

    let currentAudioBlob = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    // 1. Model Logic: Hide Scenario if CNN is selected
    modelSelect.addEventListener('change', (e) => {
        if (e.target.value === 'cnn') {
            scenarioContainer.style.opacity = '0.5';
            scenarioContainer.style.pointerEvents = 'none';
        } else {
            scenarioContainer.style.opacity = '1';
            scenarioContainer.style.pointerEvents = 'auto';
        }
    });

    // 2. Upload Box Logic
    uploadBox.addEventListener('click', () => audioUpload.click());
    
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('dragover');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleAudioSelection(e.dataTransfer.files[0]);
        }
    });

    audioUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleAudioSelection(e.target.files[0]);
        }
    });

    function handleAudioSelection(file) {
        if (!file.type.startsWith('audio/')) {
            alert('Please select a valid audio file.');
            return;
        }
        currentAudioBlob = file;
        
        // Update UI
        uploadBox.querySelector('p').innerHTML = `<strong>Selected:</strong> ${file.name}`;
        uploadBox.style.borderColor = 'var(--accent)';
        uploadBox.style.background = '#fafafa';
        
        setupPlayback();
        analyzeBtn.disabled = false;
    }

    // 3. Microphone Recording Logic
    recordBtn.addEventListener('click', async () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks);
                
                try {
                    // Convert WebM/OPUS to true WAV so backend can read without FFmpeg
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    const wavBlob = audioBufferToWav(audioBuffer);
                    
                    currentAudioBlob = wavBlob;
                    
                    // Update UI
                    recordBtn.classList.remove('recording');
                    recordBtn.innerHTML = '<i class="fa-solid fa-microphone"></i> Re-record';
                    recordStatus.textContent = 'Recording saved (Max 5s reached)';
                    uploadBox.querySelector('p').innerHTML = `Drag & drop audio file or <strong>browse</strong>`;
                    uploadBox.style.borderColor = 'var(--border)';
                    uploadBox.style.background = 'transparent';
                    
                    setupPlayback();
                    analyzeBtn.disabled = false;
                } catch (err) {
                    alert('Error processing recorded audio.');
                    console.error(err);
                }
                
                // Stop all tracks to turn off microphone indicator
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            isRecording = true;
            recordBtn.classList.add('recording');
            recordBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Recording';
            recordStatus.textContent = 'Recording... Will stop automatically in 5s';

            // Auto stop after 5 seconds
            setTimeout(() => {
                if (isRecording) stopRecording();
            }, 5000);

        } catch (err) {
            alert('Microphone access denied or not available.');
            console.error(err);
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            isRecording = false;
        }
    }

    function setupPlayback() {
        if (currentAudioBlob) {
            const url = URL.createObjectURL(currentAudioBlob);
            audioPlayback.src = url;
            audioPlayerContainer.classList.remove('hidden');
        }
    }

    // 4. Send to Backend
    analyzeBtn.addEventListener('click', async () => {
        if (!currentAudioBlob) return;

        const formData = new FormData();
        formData.append('audio', currentAudioBlob, 'recording.wav');
        formData.append('model', modelSelect.value);
        formData.append('scenario', document.getElementById('scenarioSelect').value);

        // UI Updates for loading
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'ANALYZING...';
        statusText.classList.add('hidden');
        predictionResult.classList.add('hidden');
        loader.classList.remove('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                alert(`Error: ${data.error}`);
                resetUI();
                return;
            }

            displayResult(data);

        } catch (error) {
            alert('Failed to connect to the server. Is the Flask app running?');
            console.error(error);
            resetUI();
        }
    });

    function displayResult(data) {
        loader.classList.add('hidden');
        predictionResult.classList.remove('hidden');
        
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'ANALYZE AUDIO';

        const isMale = data.prediction === 'Male';
        
        genderIcon.className = `gender-icon fa-solid ${isMale ? 'fa-mars male' : 'fa-venus female'}`;
        genderText.textContent = data.prediction;
        genderText.style.color = isMale ? '#3b82f6' : '#ec4899';
        
        modelUsed.textContent = data.model_used;
        scenarioUsed.textContent = data.model_used === 'CNN' ? 'Spectrogram' : `Scenario ${data.scenario.replace('exp', '')}`;
    }

    function resetUI() {
        loader.classList.add('hidden');
        statusText.classList.remove('hidden');
        statusText.textContent = 'Analysis Failed. Try Again.';
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'ANALYZE AUDIO';
    }

    // Helper function to convert AudioBuffer to WAV format
    function audioBufferToWav(buffer) {
        let numOfChan = buffer.numberOfChannels,
            length = buffer.length * numOfChan * 2 + 44,
            bufferArr = new ArrayBuffer(length),
            view = new DataView(bufferArr),
            channels = [], i, sample,
            offset = 0,
            pos = 0;

        function setUint16(data) {
            view.setUint16(offset, data, true);
            offset += 2;
        }

        function setUint32(data) {
            view.setUint32(offset, data, true);
            offset += 4;
        }

        setUint32(0x46464952);                         // "RIFF"
        setUint32(length - 8);                         // file length - 8
        setUint32(0x45564157);                         // "WAVE"
        setUint32(0x20746d66);                         // "fmt " chunk
        setUint32(16);                                 // length = 16
        setUint16(1);                                  // PCM (uncompressed)
        setUint16(numOfChan);
        setUint32(buffer.sampleRate);
        setUint32(buffer.sampleRate * 2 * numOfChan);  // avg. bytes/sec
        setUint16(numOfChan * 2);                      // block-align
        setUint16(16);                                 // 16-bit
        setUint32(0x61746164);                         // "data" - chunk
        setUint32(length - pos - 4);                   // chunk length

        for(i = 0; i < buffer.numberOfChannels; i++)
            channels.push(buffer.getChannelData(i));

        while(pos < buffer.length) {
            for(i = 0; i < numOfChan; i++) {
                sample = Math.max(-1, Math.min(1, channels[i][pos]));
                sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767)|0;
                view.setInt16(offset, sample, true);
                offset += 2;
            }
            pos++;
        }

        return new Blob([bufferArr], {type: "audio/wav"});
    }
});
