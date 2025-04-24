// DOM Elements
const dropArea = document.getElementById('drop-area');
const preview = document.getElementById('preview');
const promptPre = document.getElementById('prompt');
const negativePromptPre = document.getElementById('negative-prompt');
const parametersTable = document.getElementById('parameters-table');
const notification = document.getElementById('notification');
const resultsDiv = document.getElementById('results');
const editDropArea = document.getElementById('edit-drop-area');
const editPreview = document.getElementById('edit-preview');
const editPrompt = document.getElementById('edit-prompt');
const editNegativePrompt = document.getElementById('edit-negative-prompt');
const editParameters = document.getElementById('edit-parameters');
const saveButton = document.getElementById('save-changes');
const editNotification = document.getElementById('edit-notification');

const loraDropArea = document.getElementById('lora-drop-area');
const loraNotification = document.getElementById('lora-notification');
const loraResults = document.getElementById('lora-results');
const loraSummary = document.getElementById('lora-summary');

// State
let currentFile = null;
let currentMetadata = null;

// Constants
const negativePrefix = 'Negative prompt: ';
const paramsPrefix = 'Steps: ';

// Initialize
function init() {
    setupDragAndDrop();
    setupEditDragAndDrop();
    setupCopyButtons();
    setupTabSwitching();
    setupSaveHandler();
    setupFileInputs();
	setupLoraDragAndDrop();
    setupChannelNavigation();
}

function setupLoraDragAndDrop() {
    const highlight = () => loraDropArea.classList.add('highlight');
    const unhighlight = () => loraDropArea.classList.remove('highlight');
    const preventDefaults = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        loraDropArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        loraDropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        loraDropArea.addEventListener(eventName, unhighlight, false);
    });

    loraDropArea.addEventListener('drop', handleLoraDrop, false);
    
    // File input handling
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.safetensors';
    fileInput.style.display = 'none';
    document.body.appendChild(fileInput);
    
    loraDropArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleLoraFile(e.target.files[0]);
    });
}

async function handleLoraDrop(e) {
    const file = e.dataTransfer.files[0];
    if (file) handleLoraFile(file);
}

async function handleLoraFile(file) {
    if (!file.name.endsWith('.safetensors')) {
        showLoraNotification('Please drop a valid .safetensors file', 'error');
        return;
    }
    
    try {
        const arrayBuffer = await file.slice(0, 8).arrayBuffer();
        const metadataSize = new DataView(arrayBuffer).getUint32(0, true);
        const metadataArrayBuffer = await file.slice(8, 8 + metadataSize).arrayBuffer();
        const header = JSON.parse(new TextDecoder('utf-8').decode(new Uint8Array(metadataArrayBuffer)));
        
        // Extract and normalize metadata
        const metadata = header['__metadata__'] || {};
        const normalizedMetadata = normalizeLoraMetadata(metadata);
        
        // Display the normalized metadata
        displayLoraMetadata(normalizedMetadata);
        
        loraResults.classList.remove('hidden');
        showLoraNotification('File loaded successfully', 'success');
    } catch (error) {
        console.error('Error processing file:', error);
        showLoraNotification('Error processing file: ' + error.message, 'error');
    }
}

function normalizeLoraMetadata(metadata) {
    // Create a normalized structure with fallbacks
    return {
        // Model Info
        ss_output_name: metadata.ss_output_name || metadata.output_name,
        ss_sd_model_name: metadata.ss_sd_model_name || metadata.sd_model_name,
        ss_vae_name: metadata.ss_vae_name || metadata.vae_name,
        
        // Training Parameters
        ss_total_batch_size: metadata.ss_total_batch_size || metadata.batch_size,
        ss_resolution: metadata.ss_resolution || metadata.resolution,
        ss_clip_skip: metadata.ss_clip_skip || metadata.clip_skip || '1',
        ss_epoch: metadata.ss_epoch || metadata.epoch,
        ss_num_epochs: metadata.ss_num_epochs || metadata.num_epochs,
        ss_steps: metadata.ss_steps || metadata.steps,
        ss_max_train_steps: metadata.ss_max_train_steps || metadata.max_train_steps,
        
        // Optimizer
        ss_optimizer: metadata.ss_optimizer || metadata.optimizer,
        ss_lr_scheduler: metadata.ss_lr_scheduler || metadata.lr_scheduler,
        ss_learning_rate: metadata.ss_learning_rate || metadata.learning_rate,
        ss_text_encoder_lr: metadata.ss_text_encoder_lr || metadata.text_encoder_lr,
        ss_unet_lr: metadata.ss_unet_lr || metadata.unet_lr,
        ss_network_args: metadata.ss_network_args || metadata.network_args,
        
        // Training Info
        ss_training_started_at: metadata.ss_training_started_at || metadata.training_started_at,
        ss_training_finished_at: metadata.ss_training_finished_at || metadata.training_finished_at,
        
        // Dataset
        ss_dataset_dirs: metadata.ss_dataset_dirs || metadata.dataset_dirs,
        
        // Other
        ss_training_comment: metadata.ss_training_comment || metadata.training_comment,
        ss_tag_frequency: metadata.ss_tag_frequency || metadata.tag_frequency
    };
}
function displayLoraMetadata(metadata) {
    try {
        // Clear previous results
        loraResults.classList.remove('hidden');
        
        // ===================== MODEL SECTION =====================
        document.getElementById('model-name').textContent = metadata.ss_output_name || 'N/A';
        
        // Base Model with link
        const baseModelElem = document.getElementById('base-model');
        if (metadata.ss_sd_model_name) {
            if (metadata.ss_sd_model_name.includes('/')) {
                // HuggingFace model
                baseModelElem.innerHTML = `<a href="https://huggingface.co/${metadata.ss_sd_model_name}" target="_blank" style="color: var(--discord-accent);">${metadata.ss_sd_model_name}</a>`;
            } else {
                // CivitAI model
                baseModelElem.textContent = metadata.ss_sd_model_name;
            }
        } else {
            baseModelElem.textContent = 'N/A';
        }
        
        document.getElementById('model-vae').textContent = metadata.ss_vae_name || 'N/A';

        // ===================== GENERAL SECTION =====================
        // Batch Size - show actual value or N/A
        const batchSize = metadata.ss_total_batch_size || 
                         (metadata.ss_datasets?.[0]?.batch_size_per_device) || 
                         'N/A';
        document.getElementById('batch-size').textContent = batchSize;
        
        // Resolution - handle array format
        const resolution = metadata.ss_resolution || 
                          (metadata.ss_datasets?.[0]?.resolution);
        document.getElementById('resolution').textContent = 
            resolution ? (Array.isArray(resolution) ? resolution.join('x') : resolution) : 'N/A';
        
        document.getElementById('clip-skip').textContent = metadata.ss_clip_skip || '1';
        document.getElementById('epoch').textContent = 
            `${metadata.ss_epoch || 'N/A'} of ${metadata.ss_num_epochs || 'N/A'}`;
        document.getElementById('steps').textContent = 
            `${metadata.ss_steps || 'N/A'} of ${metadata.ss_max_train_steps || 'N/A'}`;

        // ===================== OPTIMIZER SECTION =====================
        document.getElementById('optimizer-type').textContent = 
            extractOptimizerName(metadata.ss_optimizer) || 'N/A';
        document.getElementById('scheduler').textContent = metadata.ss_lr_scheduler || 'N/A';
        
        // Learning rates
        const learningRates = [
            `LR: ${metadata.ss_learning_rate || 'N/A'}`,
            `TE: ${metadata.ss_text_encoder_lr || 'N/A'}`,
            `UNET: ${metadata.ss_unet_lr || 'N/A'}`
        ].join('\n');
        document.getElementById('learning-rates').textContent = learningRates;
        
        // Optional args
        const optionalArgs = metadata.ss_network_args || extractOptimizerArgs(metadata.ss_optimizer);
        document.getElementById('optional-args').textContent = 
            optionalArgs ? JSON.stringify(optionalArgs, null, 2) : 'N/A';

        // ===================== TRAINING INFO SECTION =====================
        const startDate = metadata.ss_training_started_at ? 
            new Date(metadata.ss_training_started_at * 1000).toLocaleDateString('en-US', { 
                month: 'short', 
                day: 'numeric', 
                year: 'numeric' 
            }) : 'N/A';
        document.getElementById('train-date').textContent = startDate;
        
        // Training time formatting
        let trainTime = 'N/A';
        if (metadata.ss_training_started_at && metadata.ss_training_finished_at) {
            const duration = metadata.ss_training_finished_at - metadata.ss_training_started_at;
            const hours = Math.floor(duration / 3600);
            const minutes = Math.floor((duration % 3600) / 60);
            const seconds = (duration % 60).toFixed(1);
            trainTime = `${hours}h ${minutes}m ${seconds}s`;
        }
        document.getElementById('train-time').textContent = trainTime;

        // ===================== DATASET SECTION =====================
        const datasetInfoElem = document.getElementById('dataset-info');
        datasetInfoElem.innerHTML = '';
        
        try {
            const datasetInfo = metadata.ss_dataset_dirs ? 
                (typeof metadata.ss_dataset_dirs === 'string' ? 
                    JSON.parse(metadata.ss_dataset_dirs) : 
                    metadata.ss_dataset_dirs) : 
                null;

            if (datasetInfo) {
                if (datasetInfo.img) {
                    // Create separate div for "img" category
                    const imgCategoryDiv = document.createElement('div');
                    imgCategoryDiv.className = 'param-row';
                    imgCategoryDiv.style.marginTop = '8px';
                    imgCategoryDiv.style.fontWeight = '600';
                    imgCategoryDiv.textContent = 'img:';
                    datasetInfoElem.appendChild(imgCategoryDiv);

                    // Add img properties
                    const addDatasetProperty = (name, value) => {
                        const row = document.createElement('div');
                        row.className = 'param-row';
                        row.style.marginLeft = '20px';
                        row.innerHTML = `
                            <div class="param-name">${name}:</div>
                            <div class="param-value">${value || 'N/A'}</div>
                        `;
                        datasetInfoElem.appendChild(row);
                    };

                    addDatasetProperty('n_repeats', datasetInfo.img.n_repeats);
                    addDatasetProperty('img_count', datasetInfo.img.img_count);
                } else {
                    datasetInfoElem.textContent = JSON.stringify(datasetInfo, null, 2);
                }
            } else {
                datasetInfoElem.textContent = 'N/A';
            }
        } catch (e) {
            datasetInfoElem.textContent = metadata.ss_dataset_dirs || 'N/A';
        }

        // ===================== SUGGESTED PROMPT =====================
        const suggestedPrompt = metadata.ss_training_comment || 'N/A';
        const suggestedPromptElem = document.getElementById('suggested-prompt');
        suggestedPromptElem.textContent = suggestedPrompt;
        
        // Single copy button at the top
        const promptCopyBtn = document.createElement('button');
        promptCopyBtn.className = 'copy-btn discord-btn';
        promptCopyBtn.textContent = 'Copy';
        promptCopyBtn.onclick = () => {
            copyToClipboard(suggestedPrompt);
            showLoraNotification('Prompt copied!', 'success');
        };
        suggestedPromptElem.parentNode.insertBefore(promptCopyBtn, suggestedPromptElem.nextSibling);

        // ===================== TAG FREQUENCY SECTION =====================
        const tagFrequencyElem = document.getElementById('tag-frequency');
        tagFrequencyElem.innerHTML = '';
        
        try {
            const tagData = metadata.ss_tag_frequency ? 
                (typeof metadata.ss_tag_frequency === 'string' ? 
                    JSON.parse(metadata.ss_tag_frequency) : 
                    metadata.ss_tag_frequency) : 
                {};

            if (Object.keys(tagData).length > 0) {
                // Add category header
                const categoryDiv = document.createElement('div');
                categoryDiv.className = 'param-row';
                categoryDiv.style.marginTop = '8px';
                categoryDiv.style.fontWeight = '600';
                categoryDiv.textContent = 'img:';
                tagFrequencyElem.appendChild(categoryDiv);

                // Single copy button at the top
                const copyBtn = document.createElement('button');
                copyBtn.className = 'copy-btn discord-btn';
                copyBtn.textContent = 'Copy All';
                copyBtn.style.marginLeft = '10px';
                copyBtn.onclick = () => {
                    copyToClipboard(JSON.stringify(tagData, null, 2));
                    showLoraNotification('All tags copied!', 'success');
                };
                categoryDiv.appendChild(copyBtn);

                // Display tags in simple format
                const tagsContainer = document.createElement('div');
                tagsContainer.style.maxHeight = '300px';
                tagsContainer.style.overflowY = 'auto';
                tagsContainer.style.marginTop = '8px';
                tagsContainer.style.fontFamily = 'monospace';
                
                Object.entries(tagData.img || {}).forEach(([tag, count]) => {
                    const tagDiv = document.createElement('div');
                    tagDiv.className = 'param-row';
                    tagDiv.style.marginLeft = '20px';
                    tagDiv.innerHTML = `
                        <span style="color: var(--discord-light)">${tag}</span>
                        <span style="color: var(--discord-green); margin-left: 10px">${count}</span>
                    `;
                    tagsContainer.appendChild(tagDiv);
                });

                tagFrequencyElem.appendChild(tagsContainer);
            } else {
                tagFrequencyElem.textContent = 'No tag frequency data available';
            }
        } catch (e) {
            tagFrequencyElem.textContent = metadata.ss_tag_frequency || 'N/A';
        }

        showLoraNotification('File loaded successfully', 'success');
    } catch (error) {
        console.error('Error displaying metadata:', error);
        showLoraNotification('Error displaying metadata', 'error');
    }
}


// Helper functions
function extractOptimizerName(optimizerString) {
    if (!optimizerString) return null;
    // Match the main optimizer name (before parenthesis)
    const match = optimizerString.match(/^([^(]+)/);
    return match ? match[0].replace('opt.', '').trim() : optimizerString;
}

function extractOptimizerArgs(optimizerString) {
    if (!optimizerString) return null;
    // Extract arguments inside parentheses
    const argsMatch = optimizerString.match(/\(([^)]+)\)/);
    if (!argsMatch) return null;
    
    // Parse arguments into key-value pairs
    const args = {};
    argsMatch[1].split(',').forEach(arg => {
        const [key, value] = arg.split('=').map(s => s.trim());
        if (key && value) args[key] = value;
    });
    return args;
}

function formatDatasetInfo(datasetDirs) {
    if (!datasetDirs) return 'N/A';
    if (typeof datasetDirs === 'string') return datasetDirs;
    
    // Format for common dataset structure
    if (datasetDirs.img) {
        return JSON.stringify({
            img: {
                n_repeats: datasetDirs.img.n_repeats || 'N/A',
                img_count: datasetDirs.img.img_count || 'N/A'
            }
        }, null, 2);
    }
    
    return JSON.stringify(datasetDirs, null, 2);
}

function formatTagFrequency(tagFrequency) {
    if (!tagFrequency) return 'N/A';
    if (typeof tagFrequency === 'string') return tagFrequency;
    
    // Convert to sorted array of entries
    const entries = Object.entries(tagFrequency);
    entries.sort((a, b) => b[1] - a[1]);
    
    // Format as string with counts
    return entries.map(([tag, count]) => `${tag}: ${count}`).join('\n');
}

function showLoraNotification(message, type) {
    loraNotification.textContent = message;
    loraNotification.className = type ? `discord-notification ${type}` : '';
    loraNotification.style.display = 'block';
    
    setTimeout(() => {
        loraNotification.style.display = 'none';
    }, 3000);
}

// Set up file inputs for both tabs
function setupFileInputs() {
    // For view tab
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.style.display = 'none';
    document.body.appendChild(fileInput);
    
    dropArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // For edit tab
    const editFileInput = document.createElement('input');
    editFileInput.type = 'file';
    editFileInput.accept = 'image/*';
    editFileInput.style.display = 'none';
    document.body.appendChild(editFileInput);
    
    editDropArea.addEventListener('click', () => editFileInput.click());
    editFileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleEditFile(e.target.files[0]);
        }
    });
}

// Set up channel navigation
function setupChannelNavigation() {
    const channels = document.querySelectorAll('.channel');
    channels.forEach(channel => {
        channel.addEventListener('click', function() {
            // Remove active class from all channels
            channels.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked channel
            this.classList.add('active');
            
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show the corresponding tab
            const tabId = this.getAttribute('data-tab') + '-tab';
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// Set up drag and drop for view tab
function setupDragAndDrop() {
    const highlight = () => dropArea.classList.add('highlight');
    const unhighlight = () => dropArea.classList.remove('highlight');
    const preventDefaults = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    dropArea.addEventListener('drop', handleDrop, false);
}

// Set up drag and drop for edit tab
function setupEditDragAndDrop() {
    const highlight = () => editDropArea.classList.add('highlight');
    const unhighlight = () => editDropArea.classList.remove('highlight');
    const preventDefaults = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        editDropArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        editDropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        editDropArea.addEventListener(eventName, unhighlight, false);
    });

    editDropArea.addEventListener('drop', handleEditDrop, false);
}

// Set up tab switching
function setupTabSwitching() {
    document.querySelectorAll('.channel').forEach(channel => {
        channel.addEventListener('click', function() {
            // Update active channel
            document.querySelectorAll('.channel').forEach(c => c.classList.remove('active'));
            this.classList.add('active');
            
            // Show corresponding tab
            const tabId = this.getAttribute('data-tab') + '-tab';
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// Set up copy buttons
function setupCopyButtons() {
    document.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetId = e.target.parentElement.querySelector('pre').id;
            const textToCopy = document.getElementById(targetId).textContent;
            copyToClipboard(textToCopy);
            showNotification('Copied to clipboard!', 'success');
        });
    });
}

// Set up save button
// Update your setupSaveHandler function to this:
function setupSaveHandler() {
    // Disable the save button
    saveButton.disabled = true;
    saveButton.innerHTML = '<i class="fas fa-lock"></i> Save & Download Image';
    saveButton.classList.add('disabled-btn');
    
    // Create "In Development" notice
    const devNotice = document.createElement('div');
    devNotice.className = 'dev-notice';
    devNotice.textContent = 'In Development';
    saveButton.parentElement.appendChild(devNotice);
    
    // Optional: Show tooltip on hover
    saveButton.title = 'This feature is currently in development';
}


// Handle file drop in view tab
function handleDrop(e) {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
}

// Handle file drop in edit tab
function handleEditDrop(e) {
    const file = e.dataTransfer.files[0];
    if (file) handleEditFile(file);
}

// Process file in view tab
async function handleFile(file) {
    if (!file.type.match('image.*')) {
        showNotification('Please select an image file (PNG, JPG, JPEG)', 'error');
        return;
    }

    try {
        await displayPreview(file);
        const metadata = await extractMetadata(file);
        displayMetadata(metadata);
    } catch (error) {
        console.error('Error processing file:', error);
        showNotification('Error processing file: ' + error.message, 'error');
    }
}

// Process file in edit tab
async function handleEditFile(file) {
    if (!file.type.match('image.*')) {
        showEditNotification('Please select an image file (PNG, JPG, JPEG)', 'error');
        return;
    }

    try {
        currentFile = file;
        await displayEditPreview(file);
        const metadata = await extractMetadata(file);
        currentMetadata = metadata;
        populateEditFields(metadata);
        showEditNotification('File loaded for editing', 'success');
    } catch (error) {
        console.error('Error processing file:', error);
        showEditNotification('Error processing file: ' + error.message, 'error');
    }
}

// Display preview in view tab
function displayPreview(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            resolve();
        };
        reader.readAsDataURL(file);
    });
}

// Display preview in edit tab
function displayEditPreview(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            editPreview.src = e.target.result;
            resolve();
        };
        reader.readAsDataURL(file);
    });
}

// Extract metadata from image
async function extractMetadata(file) {
    if (file.type === 'image/png') {
        return extractPNGMetadata(file);
    } else if (file.type.match('image/jpeg')) {
        return extractJPEGMetadata(file);
    } else {
        throw new Error('Unsupported image format');
    }
}

// Extract metadata from PNG
function extractPNGMetadata(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const arrayBuffer = e.target.result;
                const dataView = new DataView(arrayBuffer);
                
                // Check PNG signature
                if (dataView.getUint32(0) !== 0x89504E47 || dataView.getUint32(4) !== 0x0D0A1A0A) {
                    reject(new Error('Not a valid PNG file'));
                    return;
                }
                
                let offset = 8;
                const metadata = {
                    parameters: []
                };
                
                // Parse chunks
                while (offset < dataView.byteLength) {
                    const length = dataView.getUint32(offset);
                    const type = String.fromCharCode(
                        dataView.getUint8(offset + 4),
                        dataView.getUint8(offset + 5),
                        dataView.getUint8(offset + 6),
                        dataView.getUint8(offset + 7)
                    );
                    
                    if (type === 'tEXt' || type === 'iTXt') {
                        const dataStart = offset + 8;
                        const textData = new Uint8Array(arrayBuffer, dataStart, length);
                        const text = new TextDecoder().decode(textData);
                        const [key, value] = text.split('\0');
                        
                        if (key === 'parameters') {
                            metadata.parameters = parseParameters(value);
                        } else if (key === 'prompt' || key === 'workflow') {
                            metadata[key] = value;
                        }
                    }
                    
                    offset += 12 + length; // Move to next chunk
                    if (type === 'IEND') break; // End of file
                }
                
                resolve(metadata);
            } catch (error) {
                reject(error);
            }
        };
        reader.onerror = () => reject(new Error('Error reading file'));
        reader.readAsArrayBuffer(file);
    });
}

// Extract metadata from JPEG
function extractJPEGMetadata(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const arrayBuffer = e.target.result;
                const exifData = piexif.load(arrayBuffer);
                const metadata = {
                    parameters: []
                };

                // Check for UserComment in EXIF
                if (exifData.Exif && exifData.Exif[piexif.ExifIFD.UserComment]) {
                    const userComment = exifData.Exif[piexif.ExifIFD.UserComment];
                    const commentStr = userComment ? 
                        (typeof userComment === 'string' ? userComment : new TextDecoder('utf-8').decode(userComment)) : 
                        '';
                    
                    if (commentStr) {
                        metadata.parameters = parseParameters(commentStr);
                    }
                }

                // Check for XMP data
                if (exifData.XMP) {
                    try {
                        const xmpStr = exifData.XMP;
                        if (xmpStr.includes('<dc:description>')) {
                            const descStart = xmpStr.indexOf('<dc:description>') + 16;
                            const descEnd = xmpStr.indexOf('</dc:description>');
                            const description = xmpStr.substring(descStart, descEnd);
                            if (description) {
                                metadata.prompt = description;
                            }
                        }
                    } catch (xmpError) {
                        console.warn('Error parsing XMP data:', xmpError);
                    }
                }

                resolve(metadata);
            } catch (error) {
                reject(error);
            }
        };
        reader.onerror = () => reject(new Error('Error reading file'));
        reader.readAsArrayBuffer(file);
    });
}

// Parse parameters string into components
function parseParameters(parameters) {
    const result = [];
    let current = '';
    let inNegative = false;
    
    // Split by newlines first
    const lines = parameters.split('\n');
    
    for (const line of lines) {
        if (line.startsWith(negativePrefix)) {
            result.push(line);
            inNegative = true;
        } else if (line.startsWith(paramsPrefix)) {
            result.push(line);
        } else if (inNegative) {
            result[result.length - 1] += '\n' + line;
        } else if (result.length > 0 && !result[result.length - 1].startsWith(paramsPrefix)) {
            result[result.length - 1] += '\n' + line;
        } else {
            result.push(line);
        }
    }
    
    return result;
}

// Display metadata in UI
function displayMetadata(metadata) {
    resultsDiv.classList.remove('hidden');
    promptPre.textContent = '';
    negativePromptPre.textContent = '';
    parametersTable.innerHTML = '';
    
    if (metadata.parameters) {
        metadata.parameters.forEach(param => {
            if (param.startsWith(negativePrefix)) {
                negativePromptPre.textContent = param.substring(negativePrefix.length);
            } else if (param.startsWith(paramsPrefix)) {
                displayParametersTable(param);
            } else {
                promptPre.textContent += (promptPre.textContent ? '\n' : '') + param;
            }
        });
    }
    
    if (metadata.prompt) {
        promptPre.textContent = metadata.prompt;
    }
}

// Display parameters in a table format
function displayParametersTable(parameters) {
    // Extract key-value pairs
    const params = {};
    const parts = parameters.split(', ');
    
    parts.forEach(part => {
        const separatorIndex = part.indexOf(': ');
        if (separatorIndex > -1) {
            const key = part.substring(0, separatorIndex);
            const value = part.substring(separatorIndex + 2);
            params[key] = value;
        }
    });
    
    // Create table rows
    parametersTable.innerHTML = '';
    
    // Always show these common parameters first
    const commonParams = ['Steps', 'Sampler', 'CFG scale', 'Seed', 'Size', 'Model', 'Model hash'];
    
    commonParams.forEach(key => {
        if (params[key]) {
            addParameterRow(key, params[key]);
            delete params[key];
        }
    });
    
    // Show remaining parameters
    Object.keys(params).forEach(key => {
        addParameterRow(key, params[key]);
    });
}

// Add parameter row to table
function addParameterRow(name, value) {
    const row = document.createElement('div');
    row.className = 'param-row';
    
    const nameCell = document.createElement('div');
    nameCell.className = 'param-name';
    nameCell.textContent = name + ':';
    
    const valueCell = document.createElement('div');
    valueCell.className = 'param-value';
    valueCell.textContent = value;
    
    row.appendChild(nameCell);
    row.appendChild(valueCell);
    parametersTable.appendChild(row);
}

// Populate edit fields with metadata
function populateEditFields(metadata) {
    editPrompt.value = '';
    editNegativePrompt.value = '';
    editParameters.innerHTML = '';

    if (metadata.parameters) {
        metadata.parameters.forEach(param => {
            if (param.startsWith(negativePrefix)) {
                editNegativePrompt.value = param.substring(negativePrefix.length);
            } else if (param.startsWith(paramsPrefix)) {
                createParameterEditors(param);
            } else {
                editPrompt.value += (editPrompt.value ? '\n' : '') + param;
            }
        });
    }

    if (metadata.prompt) {
        editPrompt.value = metadata.prompt;
    }
}

// Create editable parameter fields
function createParameterEditors(parameters) {
    const params = parameters.split(', ');
    params.forEach(param => {
        const separatorIndex = param.indexOf(': ');
        if (separatorIndex > -1) {
            const key = param.substring(0, separatorIndex);
            const value = param.substring(separatorIndex + 2);
            
            const row = document.createElement('div');
            row.className = 'param-edit-row';
            
            const nameCell = document.createElement('div');
            nameCell.className = 'param-name';
            nameCell.textContent = key + ':';
            
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'param-edit-input';
            input.value = value;
            input.dataset.key = key;
            
            row.appendChild(nameCell);
            row.appendChild(input);
            editParameters.appendChild(row);
        }
    });
}

// Collect modified metadata from edit form
function collectModifiedMetadata() {
    const metadata = {
        prompt: editPrompt.value,
        negativePrompt: editNegativePrompt.value,
        parameters: []
    };

    // Reconstruct parameters string
    const params = [];
    document.querySelectorAll('.param-edit-input').forEach(input => {
        params.push(`${input.dataset.key}: ${input.value}`);
    });

    if (metadata.negativePrompt) {
        metadata.parameters.push(`${negativePrefix}${metadata.negativePrompt}`);
    }
    
    if (params.length > 0) {
        metadata.parameters.push(params.join(', '));
    }

    return metadata;
}

// Write metadata to image file
async function writeMetadata(file, metadata) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                let newFile;
                if (file.type === 'image/png') {
                    newFile = writePNGMetadata(file, e.target.result, metadata);
                } else if (file.type.match('image/jpeg')) {
                    newFile = writeJPEGMetadata(file, e.target.result, metadata);
                } else {
                    throw new Error('Unsupported image format');
                }
                resolve(newFile);
            } catch (error) {
                reject(error);
            }
        };
        reader.onerror = () => reject(new Error('Error reading file'));
        reader.readAsArrayBuffer(file);
    });
}

// Write metadata to PNG file
function writePNGMetadata(file, arrayBuffer, metadata) {
    try {
        const uint8Array = new Uint8Array(arrayBuffer);
        const encoder = new TextEncoder();
        
        // Створюємо буфер для нового файлу
        const chunks = [];
        let totalLength = 8; // Починаємо з розміру сигнатури PNG
        
        // Додаємо оригінальні чанки (крім IEND)
        let offset = 8; // Пропускаємо сигнатуру
        while (offset < uint8Array.length) {
            const length = new DataView(uint8Array.buffer, offset, 4).getUint32(0);
            const type = String.fromCharCode(...uint8Array.slice(offset + 4, offset + 8));
            
            if (type !== 'IEND') {
                const chunk = uint8Array.slice(offset, offset + 12 + length);
                chunks.push(chunk);
                totalLength += chunk.length;
                offset += 12 + length;
            } else {
                offset += 12; // Пропускаємо IEND
            }
        }

        // Додаємо нові метадані
        if (metadata.prompt) {
            const chunk = createPNGTextChunk('prompt', metadata.prompt);
            chunks.push(chunk);
            totalLength += chunk.length;
        }
        if (metadata.parameters.length > 0) {
            const chunk = createPNGTextChunk('parameters', metadata.parameters.join('\n'));
            chunks.push(chunk);
            totalLength += chunk.length;
        }

        // Додаємо IEND chunk
        const iendChunk = new Uint8Array([0,0,0,0,0x49,0x45,0x4E,0x44,0xAE,0x42,0x60,0x82]);
        totalLength += iendChunk.length;

        // Створюємо фінальний масив
        const result = new Uint8Array(totalLength);
        let position = 0;
        
        // Додаємо сигнатуру PNG
        result.set(new Uint8Array([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A]), position);
        position += 8;

        // Додаємо всі чанки
        chunks.forEach(chunk => {
            result.set(chunk, position);
            position += chunk.length;
        });

        // Додаємо IEND
        result.set(iendChunk, position);

        return new File([result], file.name, {type: file.type});
    } catch (error) {
        console.error('Error writing PNG metadata:', error);
        throw new Error('Failed to update PNG metadata');
    }
}

// Create PNG text chunk
function createPNGTextChunk(key, value) {
    const encoder = new TextEncoder();
    const keyValue = encoder.encode(key + '\0' + value);
    const length = new Uint8Array(new Uint32Array([keyValue.length]).buffer);
    const type = encoder.encode('tEXt');
    const crc = calculateCRC(new Uint8Array([...type, ...keyValue]));
    
    return new Uint8Array([
        ...length,
        ...type,
        ...keyValue,
        ...crc
    ]);
}

// Calculate CRC for PNG chunks
function calculateCRC(type, data) {
    let crc = 0xffffffff;
    const combined = new Uint8Array([...type, ...data]);
    
    for (let i = 0; i < combined.length; i++) {
        crc ^= combined[i];
        for (let j = 0; j < 8; j++) {
            crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
        }
    }
    
    crc ^= 0xffffffff;
    return new Uint8Array(new Uint32Array([crc]).buffer);
}

// Write metadata to JPEG file
function writeJPEGMetadata(file, arrayBuffer, metadata) {
    try {
        const exifObj = piexif.load(arrayBuffer);
        
        // Format the metadata string
        let metadataStr = metadata.prompt || '';
        if (metadata.negativePrompt) {
            metadataStr += `\n${negativePrefix}${metadata.negativePrompt}`;
        }
        if (metadata.parameters.length > 0) {
            metadataStr += `\n${metadata.parameters.join('\n')}`;
        }
        
        // Update EXIF UserComment
        exifObj.Exif[piexif.ExifIFD.UserComment] = metadataStr;
        
        // Update XMP data
        if (exifObj.XMP) {
            try {
                let xmpStr = exifObj.XMP;
                
                // Update description
                if (metadata.prompt) {
                    if (xmpStr.includes('<dc:description>')) {
                        xmpStr = xmpStr.replace(
                            /<dc:description>.*?<\/dc:description>/s,
                            `<dc:description>${metadata.prompt}</dc:description>`
                        );
                    } else {
                        xmpStr = xmpStr.replace(
                            /(<\/rdf:Description>)/s,
                            `<dc:description>${metadata.prompt}</dc:description>$1`
                        );
                    }
                }
                
                // Update parameters
                if (xmpStr.includes('<exif:UserComment>')) {
                    xmpStr = xmpStr.replace(
                        /<exif:UserComment>.*?<\/exif:UserComment>/s,
                        `<exif:UserComment>${metadataStr}</exif:UserComment>`
                    );
                } else {
                    xmpStr = xmpStr.replace(
                        /(<\/rdf:Description>)/s,
                        `<exif:UserComment>${metadataStr}</exif:UserComment>$1`
                    );
                }
                
                exifObj.XMP = xmpStr;
            } catch (xmpError) {
                console.warn('Error updating XMP data:', xmpError);
            }
        }
        
        // Serialize and create new file
        const exifBytes = piexif.dump(exifObj);
        const newJpeg = piexif.insert(exifBytes, arrayBuffer);
        return new File([newJpeg], file.name, {type: file.type});
    } catch (error) {
        console.error('Error writing JPEG metadata:', error);
        throw new Error('Failed to update JPEG metadata');
    }
}

// Download file
function downloadFile(file) {
    try {
        const url = URL.createObjectURL(file);
        const a = document.createElement('a');
        a.href = url;
        
        // Generate filename with timestamp
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const ext = file.type.split('/')[1] || (file.name.split('.').pop() || 'jpg');
        a.download = `metadata-edited-${timestamp}.${ext}`;
        
        document.body.appendChild(a);
        a.click();
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 100);
    } catch (error) {
        console.error('Error downloading file:', error);
        showEditNotification('Error downloading file', 'error');
    }
}

// Copy text to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).catch(err => {
        console.error('Failed to copy text: ', err);
    });
}

// Show notification in view tab
function showNotification(message, type) {
    notification.textContent = message;
    notification.className = type ? `notification-${type}` : '';
    notification.style.display = 'block';
    
    setTimeout(() => {
        notification.style.display = 'none';
    }, 3000);
}

// Show notification in edit tab
function showEditNotification(message, type) {
    editNotification.textContent = message;
    editNotification.className = type ? `edit-notification-${type}` : '';
    editNotification.style.display = 'block';
    
    setTimeout(() => {
        editNotification.style.display = 'none';
    }, 3000);
}

// Initialize the app
document.addEventListener('DOMContentLoaded', init);