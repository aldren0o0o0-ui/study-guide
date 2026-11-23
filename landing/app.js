(() => {
  // Replace this URL if you later host a direct .apk file (e.g. GitHub Releases)
  const APK_DOWNLOAD_URL =
    "https://expo.dev/accounts/aldren07/projects/study-guide/builds/0d8b68b2-bc13-4e1d-8675-1f956983633f";
  const API_HEALTH_URL =
    "https://study-guide-wcf4.onrender.com/api/v1/health?check_db=true";

  const downloadBtn = document.getElementById("downloadApkBtn");
  const copyBtn = document.getElementById("copyLinkBtn");
  const copyFeedback = document.getElementById("copyFeedback");
  const qrImage = document.getElementById("qrImage");
  const statusDot = document.getElementById("apiStatusDot");
  const statusText = document.getElementById("apiStatusText");
  const footerYear = document.getElementById("footerYear");

  if (downloadBtn) {
    downloadBtn.setAttribute("href", APK_DOWNLOAD_URL);
  }

  if (qrImage) {
    qrImage.src =
      "https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=" +
      encodeURIComponent(APK_DOWNLOAD_URL);
  }

  if (footerYear) {
    footerYear.textContent = String(new Date().getFullYear());
  }

  if (copyBtn && copyFeedback) {
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(APK_DOWNLOAD_URL);
        copyFeedback.textContent = "Install link copied to clipboard!";
        setTimeout(() => {
          copyFeedback.textContent =
            "Supports Android 8.0+ • Auto-syncs when online • Works offline after first login";
        }, 2500);
      } catch {
        copyFeedback.textContent = APK_DOWNLOAD_URL;
      }
    });
  }

  async function checkBackendHealth() {
    try {
      const res = await fetch(API_HEALTH_URL);
      const body = await res.json();
      if (res.ok && body?.data?.status === "ok" && body?.data?.database === "ok") {
        statusDot?.classList.add("online");
        if (statusText) statusText.textContent = "API & Database Online";
      } else {
        if (statusText) statusText.textContent = "Server Waking Up";
      }
    } catch {
      if (statusText) statusText.textContent = "Server Starting";
    }
  }

  void checkBackendHealth();
})();
