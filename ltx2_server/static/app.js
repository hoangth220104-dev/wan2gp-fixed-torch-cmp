/**
 * LTX-2 Video Generator - Frontend Application
 */

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
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    elements.toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ===== File Upload Handling =====
function setupFileUploads() {
    const uploads = {
        'image_start': { box: 'imageStartBox', preview: 'imageStartPreview', img: 'imageStartImg' },
        'image_end': { box: 'imageEndBox', preview: 'imageEndPreview', img: 'imageEndImg' },
        'audio_guide': { box: 'audioBox', preview: 'audioPreview' },
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
    };

    const config = configs[inputId];
    input.value = '';
    document.getElementById(config.box).classList.remove('has-file');
    document.getElementById(config.preview).hidden = true;
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
            elements.statusText.textContent = `Connected • ${data.model_type || 'Unknown'}`;
            return true;
        }
    } catch (error) {
        elements.statusIndicator.classList.remove('connected');
        elements.statusText.textContent = 'Disconnected';
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
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
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
                <small>No completed tasks</small>
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
                    <div class="task-history-meta">Seed: ${item.seed} • ${formatDuration(item.time)} • ${item.completedAt}</div>
                </div>
            </div>
        `)
        .join('');
}

function loadTask(taskId) {
    state.currentTaskId = taskId;
    startPolling(taskId);
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.loadTask = loadTask;

// ===== Form Submission =====
function setupFormSubmission() {
    elements.form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(elements.form);

        // Debug: log what's being sent
        console.log('[Form Submit] Sending form data:');
        for (let [key, value] of formData.entries()) {
            if (value instanceof File) {
                console.log(`  ${key}: File(${value.name}, ${value.size} bytes)`);
            } else {
                console.log(`  ${key}: ${value}`);
            }
        }

        // Remove empty values
        for (let [key, value] of formData.entries()) {
            if (value === '' || value === null) {
                formData.delete(key);
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

    console.log('LTX-2 Video Generator initialized');
}

// Start app when DOM is ready
document.addEventListener('DOMContentLoaded', init);
