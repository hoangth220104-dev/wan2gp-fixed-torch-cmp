

// ===== State Management =====
const state = {
    currentTaskId: null,
    pollInterval: null,
    taskHistory: [],
    isSubmitting: false,
};

// ===== API Configuration =====
const API_BASE = window.location.origin;
const API = {
    generate: `${API_BASE}/api/v1/generate`,
    taskStatus: (id) => `${API_BASE}/api/v1/tasks/${id}`,
    downloadVideo: (id) => `${API_BASE}/api/v1/tasks/${id}/video`,
    health: `${API_BASE}/api/v1/health`,
};

// ===== DOM Elements =====
const elements = {
    form: document.getElementById('generationForm'),
    submitBtn: document.getElementById('submitBtn'),
    statusIndicator: document.getElementById('statusIndicator'),
    statusText: document.getElementById('statusText'),
    modelInfo: document.getElementById('modelInfo'),
    emptyState: document.getElementById('emptyState'),
    activeTask: document.getElementById('activeTask'),
    taskId: document.getElementById('taskId'),
    taskBadge: document.getElementById('taskBadge'),
    progressContainer: document.getElementById('progressContainer'),
    progressFill: document.getElementById('progressFill'),
    progressPercent: document.getElementById('progressPercent'),
    stepInfo: document.getElementById('stepInfo'),
    errorMessage: document.getElementById('errorMessage'),
    errorText: document.getElementById('errorText'),
    videoResult: document.getElementById('videoResult'),
    videoPlayer: document.getElementById('videoPlayer'),
    metaSeed: document.getElementById('metaSeed'),
    metaTime: document.getElementById('metaTime'),
    downloadBtn: document.getElementById('downloadBtn'),
    taskHistoryList: document.getElementById('taskHistoryList'),
    toastContainer: document.getElementById('toastContainer'),
};

// ===== Utility Functions =====
function formatDuration(seconds) {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}m ${secs.toFixed(0)}s`;
}

function truncate(text, maxLength) {
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}

function randomSeed() {
    const seed = Math.floor(Math.random() * 4294967295);
    document.getElementById('seed').value = seed;
}

function showToast(message, type = 'info') {
    const icons = { success: 'check_circle', error: 'error', warning: 'warning', info: 'info' };
    const colors = { success: 'var(--success)', error: 'var(--error)', warning: 'var(--warning)', info: 'var(--info)' };

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span class="material-symbols-outlined" style="font-size:18px;color:${colors[type]}">${icons[type]}</span>
        <div class="toast-body">
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close-btn" onclick="this.parentElement.remove()">
            <span class="material-symbols-outlined" style="font-size:14px">close</span>
        </button>
    `;

    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ===== File Upload Handling =====
function setupFileUploads() {
    const uploads = {
        'image_start': { box: 'imageStartBox', preview: 'imageStartPreview', img: 'imageStartImg' },
        'image_end': { box: 'imageEndBox', preview: 'imageEndPreview', img: 'imageEndImg' },
        'audio_guide': { box: 'audioBox', preview: 'audioPreview' },
        'input_video': { box: 'inputVideoBox', preview: 'videoPreview' },
    };

    Object.entries(uploads).forEach(([inputId, config]) => {
        const input = document.getElementById(inputId);
        const box = document.getElementById(config.box);
        const preview = document.getElementById(config.preview);

        input.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;

            box.classList.add('has-file');
            preview.hidden = false;

            if (config.img) {
                const img = document.getElementById(config.img);
                const reader = new FileReader();
                reader.onload = (e) => img.src = e.target.result;
                reader.readAsDataURL(file);
            } else if (inputId === 'audio_guide') {
                document.getElementById('audioFileName').textContent = file.name;
            } else if (inputId === 'input_video') {
                const videoPlayer = document.getElementById('videoPreviewPlayer');
                const url = URL.createObjectURL(file);
                videoPlayer.src = url;
            }
        });
    });
}

function removeUpload(inputId) {
    const input = document.getElementById(inputId);
    const configs = {
        'image_start': { box: 'imageStartBox', preview: 'imageStartPreview' },
        'image_end': { box: 'imageEndBox', preview: 'imageEndPreview' },
        'audio_guide': { box: 'audioBox', preview: 'audioPreview' },
        'input_video': { box: 'inputVideoBox', preview: 'videoPreview' },
    };

    const config = configs[inputId];
    input.value = '';
    document.getElementById(config.box).classList.remove('has-file');
    document.getElementById(config.preview).hidden = true;

    // Clear video player source if exists
    if (inputId === 'input_video') {
        const videoPlayer = document.getElementById('videoPreviewPlayer');
        if (videoPlayer.src) {
            URL.revokeObjectURL(videoPlayer.src);
            videoPlayer.src = '';
        }
    }
}

// Make removeUpload globally accessible
window.removeUpload = removeUpload;
window.randomSeed = randomSeed;

// ===== Health Check =====
async function checkHealth() {
    try {
        const response = await fetch(API.health);
        if (response.ok) {
            const data = await response.json();
            elements.statusIndicator.classList.add('connected');
            elements.statusText.textContent = 'Connected';
            if (elements.modelInfo) {
                elements.modelInfo.textContent = data.model_type || 'Ready';
            }
            return true;
        }
    } catch (error) {
        elements.statusIndicator.classList.remove('connected');
        elements.statusText.textContent = 'Disconnected';
        if (elements.modelInfo) {
            elements.modelInfo.textContent = 'Offline';
        }
        return false;
    }
}

// ===== Task Management =====
async function submitTask(formData) {
    try {
        state.isSubmitting = true;
        elements.submitBtn.disabled = true;
        elements.submitBtn.innerHTML = '<span class="spinner"></span> Submitting...';

        const response = await fetch(API.generate, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to submit task');
        }

        const data = await response.json();
        state.currentTaskId = data.task_id;

        showToast('Task submitted successfully!', 'success');

        // Start polling
        startPolling(data.task_id);

    } catch (error) {
        showToast(error.message, 'error');
        throw error;
    } finally {
        state.isSubmitting = false;
        elements.submitBtn.disabled = false;
        elements.submitBtn.innerHTML = `
            <span class="material-symbols-outlined">play_arrow</span>
            Generate Video
        `;
    }
}

function startPolling(taskId) {
    // Show active task UI
    elements.emptyState.hidden = true;
    elements.activeTask.hidden = false;
    elements.taskId.textContent = `Task: ${taskId.substring(0, 8)}...`;
    elements.taskBadge.textContent = 'QUEUED';
    elements.taskBadge.className = 'task-badge queued';

    elements.progressContainer.hidden = true;
    elements.errorMessage.hidden = true;
    elements.videoResult.hidden = true;

    // Poll every 1 second
    state.pollInterval = setInterval(async () => {
        await pollTaskStatus(taskId);
    }, 1000);
}

async function pollTaskStatus(taskId) {
    try {
        const response = await fetch(API.taskStatus(taskId));
        if (!response.ok) return;

        const data = await response.json();
        updateTaskUI(data);

        // Stop polling if task is done
        if (['completed', 'failed', 'cancelled'].includes(data.status)) {
            stopPolling();
            handleTaskComplete(data);
        }
    } catch (error) {
        console.error('Polling error:', error);
    }
}

function updateTaskUI(data) {
    // Update badge
    elements.taskBadge.textContent = data.status.toUpperCase();
    elements.taskBadge.className = `task-badge ${data.status}`;

    // Update progress
    if (data.status === 'processing' || data.status === 'queued') {
        elements.progressContainer.hidden = false;
        elements.progressFill.style.width = `${data.progress || 0}%`;
        elements.progressPercent.textContent = `${(data.progress || 0).toFixed(1)}%`;

        if (data.current_step && data.total_steps) {
            elements.stepInfo.textContent = `Step ${data.current_step} / ${data.total_steps}`;
        }
    }
}

function handleTaskComplete(data) {
    if (data.status === 'completed') {
        elements.progressContainer.hidden = false;
        elements.progressFill.style.width = '100%';
        elements.progressPercent.textContent = '100%';

        elements.videoResult.hidden = false;
        elements.videoPlayer.src = API.downloadVideo(data.task_id);

        elements.metaSeed.textContent = data.result.seed;
        elements.metaTime.textContent = formatDuration(data.result.generation_time);
        elements.downloadBtn.href = API.downloadVideo(data.task_id);

        // Add to history
        addToHistory(data);

        showToast('Video generation complete!', 'success');
    } else if (data.status === 'failed') {
        elements.errorMessage.hidden = false;
        elements.errorText.textContent = data.error || 'Unknown error occurred';
        showToast('Generation failed', 'error');
    }
}

function stopPolling() {
    if (state.pollInterval) {
        clearInterval(state.pollInterval);
        state.pollInterval = null;
    }
    state.currentTaskId = null;
}

// ===== Task History =====
function addToHistory(taskData) {
    const historyItem = {
        id: taskData.task_id,
        prompt: taskData.result.prompt || document.getElementById('prompt').value,
        seed: taskData.result.seed,
        time: taskData.result.generation_time,
        completedAt: new Date().toLocaleTimeString(),
    };

    state.taskHistory.unshift(historyItem);
    if (state.taskHistory.length > 10) state.taskHistory.pop();

    renderTaskHistory();
}

function renderTaskHistory() {
    if (state.taskHistory.length === 0) {
        elements.taskHistoryList.innerHTML = `
            <div class="empty-history">
                <small>No completed tasks yet</small>
            </div>
        `;
        return;
    }

    elements.taskHistoryList.innerHTML = state.taskHistory
        .map(item => `
            <div class="task-history-item" onclick="loadTask('${item.id}')">
                <div class="task-history-thumb"></div>
                <div class="task-history-info">
                    <div class="task-history-prompt">${truncate(item.prompt, 50)}</div>
                    <div class="task-history-meta">Seed: ${item.seed} · ${formatDuration(item.time)} · ${item.completedAt}</div>
                </div>
            </div>
        `)
        .join('');
}

function loadTask(taskId) {
    state.currentTaskId = taskId;
    startPolling(taskId);
    // Scroll to results
    document.querySelector('.page-body').scrollTo({ top: 0, behavior: 'smooth' });
}

window.loadTask = loadTask;

// ===== Form Submission =====
function setupFormSubmission() {
    elements.form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(elements.form);

        // Debug: log what's being sent
        console.log('[Form Submit] Raw form data before cleanup:');
        for (let [key, value] of formData.entries()) {
            if (value instanceof File) {
                console.log(`  ${key}: File(name="${value.name}", size=${value.size}, type="${value.type}")`);
            } else {
                console.log(`  ${key}: "${value}"`);
            }
        }

        // Remove empty values and empty files
        const keysToRemove = [];
        for (let [key, value] of formData.entries()) {
            if (value === '' || value === null) {
                keysToRemove.push(key);
            }
            if (value instanceof File) {
                // Remove files with no name and no size (empty file inputs)
                if (value.name === '' && value.size === 0) {
                    keysToRemove.push(key);
                    console.log(`  Removing empty file: ${key}`);
                }
            }
        }
        keysToRemove.forEach(key => formData.delete(key));

        console.log('[Form Submit] Form data after cleanup:');
        for (let [key, value] of formData.entries()) {
            if (value instanceof File) {
                console.log(`  ${key}: File(${value.name}, ${value.size} bytes)`);
            } else {
                console.log(`  ${key}: ${value}`);
            }
        }

        try {
            await submitTask(formData);
        } catch (error) {
            console.error('Submission error:', error);
        }
    });
}

// ===== Initialization =====
async function init() {
    // Check health
    await checkHealth();
    setInterval(checkHealth, 30000); // Check every 30 seconds

    // Setup file uploads
    setupFileUploads();

    // Setup form submission
    setupFormSubmission();

    // Render task history
    renderTaskHistory();

    console.log('Video Generator initialized');
}

// Start app when DOM is ready
document.addEventListener('DOMContentLoaded', init);
