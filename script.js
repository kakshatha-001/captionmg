document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();

    var formData = new FormData();
    formData.append('file', document.getElementById('file').files[0]);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Display caption
        document.getElementById('output').innerHTML = '<p><strong>Caption:</strong> ' + data.caption + '</p>';

        // Play audio
        var audioBlob = new Blob([data.audio], { type: 'audio/mp3' });
        var audioUrl = URL.createObjectURL(audioBlob);
        var audio = new Audio(audioUrl);
        audio.play();
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('output').innerHTML = '<p>Error occurred. Please try again.</p>';
    });
});
