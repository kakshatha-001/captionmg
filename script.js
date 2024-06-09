document.addEventListener('DOMContentLoaded', function() {
    var form = document.getElementById('upload-form');
    var fileInput = document.getElementById('file-input');

    form.addEventListener('submit', function(event) {
        event.preventDefault();
        var formData = new FormData(form);
        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(data => {
            var output = document.getElementById('output');
            output.innerHTML = '<p>' + data + '</p>';
        })
        .catch(error => console.error('Error:', error));
    });
});
