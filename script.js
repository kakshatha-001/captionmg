document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    const fileInput = document.getElementById('file-input');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    if (result.error) {
        alert(result.error);
    } else {
        document.getElementById('caption').textContent = result.caption;
        const audio = document.getElementById('audio');
        audio.src = result.audio_url;
        audio.load();
    }
});
