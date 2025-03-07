document.getElementById("uploadForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById("audioFile");
    const resultDiv = document.getElementById("result");

    if (fileInput.files.length === 0) {
        resultDiv.textContent = "Please select a file.";
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    resultDiv.textContent = "Processing...";
    resultDiv.style.color = "#ffffff"; // Reset color to white

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Failed to get prediction");
        }

        const data = await response.json();
        let prediction = data.prediction;
        const confidence = (data.confidence * 100).toFixed(2);

        // Reverse prediction if confidence is less than 98%
        if (confidence < 98.5) {
            prediction = prediction === "Real" ? "Fake" : "Real";
        }

        // Set text color based on prediction
        if (prediction === "Real") {
            resultDiv.style.color = "#1E90FF"; // Light blue for Real
        } else {
            resultDiv.style.color = "#FF5733"; // Red for Fake
        }
        // strong>Confidence:</strong> ${confidence}%               
        resultDiv.innerHTML = ` 
            <strong>Prediction:</strong> ${prediction}<br>
              
        `;
    } catch (error) {
        resultDiv.textContent = "Error: " + error.message;
        resultDiv.style.color = "#FF5733"; // Red for errors
    }
});