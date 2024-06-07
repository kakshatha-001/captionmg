document.getElementById('captionBtn').addEventListener('click', async function() {
    const fileInput = document.getElementById('fileInput');
    const outputDiv = document.getElementById('output');
    
    const file = fileInput.files[0];
    if (!file) {
        outputDiv.innerText = 'Please select an image.';
        return;
    }
    
    const formData = new FormData();
    formData.append('image', file);
    
    try {
        const response = await fetch('/process_image', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Failed to process image.');
        }
        
        const data = await response.json();
        outputDiv.innerText = `Caption: ${data.caption}`;
    } catch (error) {
        console.error('Error:', error);
        outputDiv.innerText = 'An error occurred.';
    }
});
