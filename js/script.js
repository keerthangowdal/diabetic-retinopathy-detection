document.getElementById("upload-form").addEventListener("submit", function(event) {
    event.preventDefault(); // Prevent default form submission

    var formData = new FormData();
    var fileInput = document.getElementById("file");
    var file = fileInput.files[0];

    if (!file) {
        alert("Please select an image to upload.");
        return;
    }

    // Show loading animation and hide previous results
    document.getElementById("loading").style.display = "block";
    document.getElementById("result-container").style.display = "none";

    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("loading").style.display = "none";

        if (data.error) {
            alert(data.error);
        } else {
            document.getElementById("predicted-class").textContent = data.predicted_class;
            document.getElementById("predicted-label").textContent = data.predicted_label;
            document.getElementById("uploaded-image").src = data.filepath;
            document.getElementById("result-container").style.display = "block";
        }
    })
    .catch(error => {
        document.getElementById("loading").style.display = "none";
        console.error("Error:", error);
        alert("There was an error with the prediction. Please try again.");
    });
});

document.addEventListener("DOMContentLoaded", function() {
    const loginForm = document.getElementById("loginForm");
    const registerForm = document.getElementById("registerForm");

    if (loginForm) {
        loginForm.addEventListener("submit", function(event) {
            const email = loginForm.querySelector('input[name="email"]').value;
            const password = loginForm.querySelector('input[name="password"]').value;
            if (!email || !password) {
                alert("Please fill in all fields.");
                event.preventDefault();
            }
        });
    }

    if (registerForm) {
        registerForm.addEventListener("submit", function(event) {
            const username = registerForm.querySelector('input[name="username"]').value;
            const email = registerForm.querySelector('input[name="email"]').value;
            const password = registerForm.querySelector('input[name="password"]').value;
            if (!username || !email || !password) {
                alert("Please fill in all fields.");
                event.preventDefault();
            }
        });
    }
});

