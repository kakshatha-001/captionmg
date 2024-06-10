document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    var formData = new FormData();
    formData.append('file', document.getElementById('fileInput').files[0]);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Display caption
        document.getElementById('captionContainer').innerText = 'Caption: ' + data.caption;

        // Play audio
        var audioPlayer = document.getElementById('audioPlayer');
        audioPlayer.src = 'static/caption.mp3';  // Audio file path
        audioPlayer.style.display = 'block';
        audioPlayer.play();
    })
    .catch(error => console.error('Error:', error));
});
