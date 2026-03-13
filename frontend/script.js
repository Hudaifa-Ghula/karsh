const canvas = document.getElementById('morphCanvas');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('morphSlider');
const fileInput = document.getElementById('fileInput');
const presetList = document.getElementById('presetList');
const status = document.getElementById('status');
const playBtn = document.getElementById('playBtn');
const speedSlider = document.getElementById('speedSlider');

let sourceImg = null;
let targetImg = null;
let assignments = null;
let sidelen = 128;
let isPlaying = false;
let playDirection = 1;

// Initialize
async function init() {
    canvas.width = 512;
    canvas.height = 512;
    
    // Load presets
    try {
        const response = await fetch('/presets');
        const presets = await response.json();
        renderPresets(presets);
        
        // Load target image
        targetImg = new Image();
        targetImg.src = '/presets/shrek/target.png'; // Any preset has the same target
        await new Promise(r => targetImg.onload = r);
        
        // Load default preset (shrek)
        loadPreset('shrek');
    } catch (e) {
        showStatus('Error connecting to backend');
    }
}

function renderPresets(presets) {
    presetList.innerHTML = '';
    presets.forEach(p => {
        const item = document.createElement('div');
        item.className = 'preset-item';
        item.textContent = p.name;
        item.onclick = () => loadPreset(p.id);
        item.id = `preset-${p.id}`;
        presetList.appendChild(item);
    });
}

async function loadPreset(id) {
    showStatus(`Loading preset: ${id}...`);
    document.querySelectorAll('.preset-item').forEach(el => el.classList.remove('active'));
    document.getElementById(`preset-${id}`).classList.add('active');
    
    try {
        const [srcRes, assignRes] = await Promise.all([
            fetch(`/presets/${id}/source`),
            fetch(`/presets/${id}/assignments`)
        ]);
        
        const blob = await srcRes.blob();
        assignments = await assignRes.json();
        sidelen = Math.sqrt(assignments.length);
        
        // Use FileReader and Image exactly like upload logic
        const reader = new FileReader();
        reader.onload = async (re) => {
            const img = new Image();
            img.onload = async () => {
                // Resize for local display
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = sidelen;
                tempCanvas.height = sidelen;
                const tCtx = tempCanvas.getContext('2d');
                tCtx.drawImage(img, 0, 0, sidelen, sidelen);
                sourceImg = await createImageBitmap(tempCanvas);
                draw(0);
                showStatus('Ready');
            };
            img.src = re.target.result;
        };
        reader.readAsDataURL(blob);

    } catch (e) {
        showStatus('Error loading preset');
    }
}

fileInput.onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    showStatus('Uploading and processing...');
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/transform', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        // Load the source image locally for display
        const reader = new FileReader();
        reader.onload = async (re) => {
            const img = new Image();
            img.onload = async () => {
                // Resize for local display
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = 128;
                tempCanvas.height = 128;
                const tCtx = tempCanvas.getContext('2d');
                tCtx.drawImage(img, 0, 0, 128, 128);
                sourceImg = await createImageBitmap(tempCanvas);
                assignments = data.assignments;
                sidelen = 128;
                draw(0);
                showStatus('Ready');
            };
            img.src = re.target.result;
        };
        reader.readAsDataURL(file);
        
        document.querySelectorAll('.preset-item').forEach(el => el.classList.remove('active'));
    } catch (e) {
        showStatus('Processing failed');
    }
};

function draw(t) {
    if (!sourceImg || !assignments) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const pixelSize = canvas.width / sidelen;
    
    for (let i = 0; i < assignments.length; i++) {
        const targetIdx = i;
        const sourceIdx = assignments[i];
        
        const tx = (targetIdx % sidelen) * pixelSize;
        const ty = Math.floor(targetIdx / sidelen) * pixelSize;
        
        const sx = (sourceIdx % sidelen) * pixelSize;
        const sy = Math.floor(sourceIdx / sidelen) * pixelSize;
        
        // Linearly interpolate between source and target positions
        // This creates the "movement" effect
        const x = sx + (tx - sx) * t;
        const y = sy + (ty - sy) * t;
        
        // Grab pixel color from source image
        // We use the source image at its original grid resolution
        ctx.drawImage(sourceImg, 
            (sourceIdx % sidelen), Math.floor(sourceIdx / sidelen), 1, 1,
            x, y, pixelSize, pixelSize
        );
    }
}

slider.oninput = () => {
    isPlaying = false;
    playBtn.textContent = 'Play';
    draw(parseFloat(slider.value));
};

playBtn.onclick = () => {
    isPlaying = !isPlaying;
    playBtn.textContent = isPlaying ? 'Pause' : 'Play';
    
    if (isPlaying) {
        if (parseFloat(slider.value) >= 1) slider.value = 0;
        playDirection = 1;
        requestAnimationFrame(animatePlay);
    }
};

function animatePlay() {
    if (!isPlaying) return;
    
    let val = parseFloat(slider.value);
    const speed = parseFloat(speedSlider.value) / 100;
    
    val += speed * playDirection;
    
    if (val >= 1) {
        val = 1;
        isPlaying = false;
        playBtn.textContent = 'Play';
    }
    
    slider.value = val;
    draw(val);
    
    if (isPlaying) {
        requestAnimationFrame(animatePlay);
    }
}

function showStatus(msg) {
    status.textContent = msg;
    status.classList.add('show');
    if (msg === 'Ready') {
        setTimeout(() => status.classList.remove('show'), 2000);
    }
}

init();
