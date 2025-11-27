// script.js
async function getBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(",")[1]);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

async function postJSON(endpoint, payload) {
    const out = document.getElementById("output");
    try {
        const res = await fetch(endpoint, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });
        const json = await res.json();
        out.textContent = JSON.stringify(json, null, 2);
        return json;
    } catch (e) {
        out.textContent = "Error: " + e;
        return null;
    }
}

// UI elements
const upload = document.getElementById("upload");
const preview = document.getElementById("preview");
const registerBtn = document.getElementById("registerBtn");
const compareBtn = document.getElementById("compareBtn");
const resultCard = document.getElementById("resultCard");
const selectedImg = document.getElementById("selectedImg");
const matchedImg = document.getElementById("matchedImg");
const bboxQuery = document.getElementById("bboxQuery");
const bboxMatch = document.getElementById("bboxMatch");
const matchStatus = document.getElementById("matchStatus");
const scoreBox = document.getElementById("scoreBox");
const popup = document.getElementById("popup");
const zoomOverlay = document.getElementById("zoomOverlay");
const zoomLeft = document.getElementById("zoomLeft");
const zoomRight = document.getElementById("zoomRight");
const zoomBtn = document.getElementById("zoomBtn");
const closeZoom = document.getElementById("closeZoom");
const historyList = document.getElementById("historyList");
const themeToggle = document.getElementById("themeToggle");
const clearHistoryBtn = document.getElementById("clearHistory");

upload.addEventListener("change", (e) => {
    preview.innerHTML = "";
    const files = [...e.target.files];
    files.forEach(f => {
        const img = document.createElement("img");
        img.src = URL.createObjectURL(f);
        img.title = f.name;
        preview.appendChild(img);
        img.addEventListener("click", () => {
            selectedFile = f;
            selectedImg.src = URL.createObjectURL(f);
        });
    });
    if (files.length === 1) {
        selectedFile = files[0];
        selectedImg.src = URL.createObjectURL(files[0]);
    }
});

let selectedFile = null;

registerBtn.addEventListener("click", async () => {
    const files = [...upload.files];
    if (!files.length) return alert("Select images to register");
    const images = [];
    for (const f of files) images.push(await getBase64(f));
    await postJSON("/register", { images });
    await loadHistory();
});

compareBtn.addEventListener("click", async () => {
    const files = [...upload.files];
    if (!files.length && !selectedFile) return alert("Select an image");
    const f = selectedFile || files[0];
    const b64 = await getBase64(f);

    selectedImg.src = URL.createObjectURL(f);
    resultCard.classList.remove("hidden");
    bboxQuery.style.display = "none";
    bboxMatch.style.display = "none";
    matchStatus.textContent = "Checking...";
    matchStatus.style.background = "transparent";
    scoreBox.textContent = "";

    const res = await postJSON("/compare", { image: b64 });
    if (!res) return;

    if (res.image) matchedImg.src = "/images/" + res.image;
    else matchedImg.src = "";

    if (res.msg === "FACE_MATCHED") {
        matchStatus.textContent = "MATCHED ✔";
        matchStatus.style.background = "rgba(0,255,0,0.12)";
        matchStatus.style.color = "#00ff77";
        scoreBox.textContent = "Similarity: " + Number(res.similarity).toFixed(4);
        showPopup("FACE MATCHED ✔");
    } else {
        matchStatus.textContent = "NOT MATCHED ✘";
        matchStatus.style.background = "rgba(255,0,0,0.12)";
        matchStatus.style.color = "#ff9090";
        scoreBox.textContent = res.similarity ? ("Best similarity: " + Number(res.similarity).toFixed(4)) : "No match";
    }

    function drawBoxes() {
        if (res.query_bbox) drawBBox(res.query_bbox, selectedImg, bboxQuery);
        else bboxQuery.style.display = "none";
        if (res.matched_bbox && matchedImg.src) drawBBox(res.matched_bbox, matchedImg, bboxMatch);
        else bboxMatch.style.display = "none";
    }

    const leftLoaded = selectedImg.complete && selectedImg.naturalWidth !== 0;
    const rightLoaded = !matchedImg.src || (matchedImg.complete && matchedImg.naturalWidth !== 0);

    if (leftLoaded && rightLoaded) drawBoxes();
    else {
        selectedImg.onload = () => { if (!matchedImg.src || matchedImg.complete) drawBoxes(); };
        matchedImg.onload = () => { if (selectedImg.complete) drawBoxes(); };
    }

    await loadHistory();
});

function drawBBox(bbox, imgEl, overlayEl) {
    if (!bbox) { overlayEl.style.display = "none"; return; }
    const naturalW = imgEl.naturalWidth;
    const naturalH = imgEl.naturalHeight;
    const dispW = imgEl.clientWidth;
    const dispH = imgEl.clientHeight;

    // scale by width or height depending on aspect
    const scaleX = dispW / naturalW;
    const scaleY = dispH / naturalH;

    overlayEl.style.left = (bbox.x * scaleX) + "px";
    overlayEl.style.top = (bbox.y * scaleY) + "px";
    overlayEl.style.width = (bbox.w * scaleX) + "px";
    overlayEl.style.height = (bbox.h * scaleY) + "px";
    overlayEl.style.display = "block";
}

function showPopup(text) {
    popup.classList.remove("hidden");
    popup.querySelector(".popup-text").textContent = text;
    setTimeout(()=> popup.classList.add("show"), 20);
    setTimeout(()=> {
        popup.classList.remove("show");
        setTimeout(()=> popup.classList.add("hidden"), 350);
    }, 1800);
}

zoomBtn.addEventListener("click", () => {
    zoomLeft.src = selectedImg.src || "";
    zoomRight.src = matchedImg.src || "";
    zoomOverlay.classList.remove("hidden");
});
closeZoom && closeZoom.addEventListener("click", () => zoomOverlay.classList.add("hidden"));

historyList.addEventListener("click", (e) => {
    const row = e.target.closest(".history-row");
    if (!row) return;
    zoomLeft.src = row.dataset.queryimg || "";
    zoomRight.src = row.dataset.matchedimg || "";
    zoomOverlay.classList.remove("hidden");
});

async function loadHistory(){
    const res = await fetch("/history");
    const data = await res.json();
    historyList.innerHTML = "";
    data.forEach(item => {
        const div = document.createElement("div");
        div.className = "history-row";
        div.dataset.queryimg = ""; // we don't keep query images on server
        div.dataset.matchedimg = item.matched_image ? ("/images/" + item.matched_image) : "";
        const thumb = document.createElement("img");
        thumb.className = "history-thumb";
        thumb.src = item.matched_image ? ("/images/" + item.matched_image) : "";
        const meta = document.createElement("div");
        meta.className = "history-meta";
        const time = document.createElement("div");
        time.className = "time";
        time.textContent = new Date(item.timestamp).toLocaleString();
        const msg = document.createElement("div");
        msg.className = "msg";
        msg.textContent = item.msg + (item.similarity ? ("  —  " + Number(item.similarity).toFixed(4)) : "");
        meta.appendChild(msg);
        meta.appendChild(time);
        div.appendChild(thumb);
        div.appendChild(meta);
        historyList.appendChild(div);
    });
}

themeToggle.addEventListener("click", () => {
    if (document.body.classList.contains("dark")) {
        document.body.classList.remove("dark");
        document.body.classList.add("light");
        themeToggle.textContent = "🌙";
    } else {
        document.body.classList.remove("light");
        document.body.classList.add("dark");
        themeToggle.textContent = "🌞";
    }
});

clearHistoryBtn.addEventListener("click", async () => {
    if (!confirm("Clear server history?")) return;
    await fetch("/clear_history", {method:"POST"});
    await loadHistory();
});

document.addEventListener("DOMContentLoaded", async () => {
    document.body.classList.add("light");
    await loadHistory();
});
