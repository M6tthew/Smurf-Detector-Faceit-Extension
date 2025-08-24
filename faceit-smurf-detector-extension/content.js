(function () {
    console.log("Content script loaded!");

function showProcessingOverlay() {
    // Remove existing overlay if present
    const oldOverlay = document.getElementById("smurf-processing-overlay");
    if (oldOverlay) oldOverlay.remove();

    const overlay = document.createElement("div");
    overlay.id = "smurf-processing-overlay";
    overlay.textContent = "PROCESSING SMURF SCORES...";
    overlay.style.cssText = `
        width: 400px;
        background: #008000;
        color: #fff;
        font-size: 1.5rem;
        font-weight: bold;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 24px auto;
        padding: 24px 0;
        z-index: 999999;
        position: fixed;
        left: 50%;
        bottom: 0;
        transform: translateX(-50%);
    `;

    document.body.appendChild(overlay);
}


    function hideProcessingOverlay() {
        const overlay = document.getElementById("smurf-processing-overlay");
        if (overlay) overlay.remove();
    }

    async function sendMatchId(matchId) {
        console.log("Sending Match ID:", matchId);

        showProcessingOverlay();

        try {
            const res = await fetch("http://localhost:5001/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ match_id: matchId })
            });

            const predictions = await res.json();
            console.log("Server predictions:", predictions);

            hideProcessingOverlay();

            // Keep trying to inject until players show up
            const interval = setInterval(() => {
                const players = document.querySelectorAll("[class*='Nickname__Name']");
                console.log("Checking for players... found:", players.length);

                if (players.length > 0) {
                    injectScores(predictions);
                    clearInterval(interval);
                }
            }, 1000);
        } catch (err) {
            console.error("Error sending match ID:", err);
            hideProcessingOverlay();
        }
    }

    function getMatchIdFromUrl() {
        const url = window.location.href;
        const matchPattern = /room\/([a-zA-Z0-9\-]+)/;
        const match = url.match(matchPattern);
        if (match && match[1]) {
            sendMatchId(match[1]);
        }
    }

    function getColor(score) {
        if (score <= 25) return "#006400";      // dark green
        if (score <= 45) return "#008000";      // green
        if (score <= 60) return "#FFD700";      // yellow
        if (score <= 75) return "#FFA500";      // orange
        if (score <= 85) return "#ff0000ff";
        return "#5f0000ff";                       // red
    }

    function injectScores(predictions) {
        const players = document.querySelectorAll("[class*='Nickname__Name']");
        console.log("Injecting into players:", players.length);

        players.forEach(el => {
            const name = el.innerText.trim();

            console.log("Nickname in DOM:", name);
            console.log("Predictions keys:", Object.keys(predictions));

            if (predictions[name] !== undefined) {
                const score = predictions[name];
                console.log(`Injecting score for ${name}: ${score}`);

                if (el.parentElement.querySelector(".smurf-score")) return;

                const badge = document.createElement("div");
                badge.className = "smurf-score";
                badge.textContent = `${score}%`;
                badge.style.cssText = `
                    margin-left: 8px;
                    padding: 4px 8px;
                    border-radius: 6px;
                    font-size: 16px;
                    font-weight: bold;
                    color: white;
                    background-color: ${getColor(score)};
                    display: inline-block;
                    z-index: 999999;
                `;

                el.insertAdjacentElement("afterend", badge);
            }
        });
    }
    // Run on page load
    getMatchIdFromUrl();

    // Watch for SPA navigation (Faceit uses React Router)
    let lastUrl = location.href;
    new MutationObserver(() => {
        const url = location.href;
        if (url !== lastUrl) {
            lastUrl = url;
            getMatchIdFromUrl();
        }
    }).observe(document, { subtree: true, childList: true });
})();