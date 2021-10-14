export async function fetchModelInferenceResult(data, server_ip, handleModelInferenceFinish) {
    // data: 
    //  MT: {"sentence", "head", "layer"}
    //  ASR: {"audio", "head", "layer"}
    try {
        const requestConfig = {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        };

        fetch(server_ip, requestConfig)
            .then(response => response.json())
            .then(data => handleModelInferenceFinish(data));
    }
    catch (err) {
        alert(err);
    }
}
