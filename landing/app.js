(() => {
  // Direct APK artifact URL (downloads the .apk file directly instead of opening the Expo webpage)
  const DIRECT_APK_URL =
    "https://expo.dev/artifacts/eas/X3lgXoCmbMp5W47pw33s0SIJpWdHq727pIMzOHeWPpg.apk";
  const API_HEALTH_URL =
    "https://study-guide-wcf4.onrender.com/api/v1/health?check_db=true";

  const downloadBtn = document.getElementById("downloadApkBtn");
  const downloadBtnLabel = document.getElementById("downloadBtnLabel");
  const downloadToast = document.getElementById("downloadToast");
  const copyBtn = document.getElementById("copyLinkBtn");
  const copyBtnText = document.getElementById("copyBtnText");
  const qrToggleBtn = document.getElementById("qrToggleBtn");
  const qrModal = document.getElementById("qrModal");
  const qrCloseBtn = document.getElementById("qrCloseBtn");
  const qrImage = document.getElementById("qrImage");
  const statusDot = document.getElementById("apiStatusDot");
  const statusText = document.getElementById("apiStatusText");
  const footerYear = document.getElementById("footerYear");

  if (footerYear) {
    footerYear.textContent = String(new Date().getFullYear());
  }

  if (downloadBtn) {
    downloadBtn.setAttribute("href", DIRECT_APK_URL);
    downloadBtn.addEventListener("click", () => {
      if (downloadBtnLabel) {
        downloadBtnLabel.textContent = "Downloading APK...";
        setTimeout(() => {
          downloadBtnLabel.textContent = "Download & Install APK";
        }, 3000);
      }
      if (downloadToast) {
        downloadToast.classList.add("is-visible");
      }
    });
  }

  if (qrImage) {
    qrImage.src =
      "https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=" +
      encodeURIComponent(DIRECT_APK_URL);
  }

  if (qrToggleBtn && qrModal) {
    qrToggleBtn.addEventListener("click", () => {
      qrModal.hidden = false;
    });
  }

  if (qrCloseBtn && qrModal) {
    qrCloseBtn.addEventListener("click", () => {
      qrModal.hidden = true;
    });
    qrModal.addEventListener("click", (event) => {
      if (event.target === qrModal) {
        qrModal.hidden = true;
      }
    });
  }

  if (copyBtn && copyBtnText) {
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(DIRECT_APK_URL);
        copyBtnText.textContent = "Copied";
        setTimeout(() => {
          copyBtnText.textContent = "Copy Link";
        }, 2200);
      } catch {
        copyBtnText.textContent = "Copy Failed";
      }
    });
  }

  // Interactive phone mockup tabs
  const tabs = document.querySelectorAll(".screen-tab");
  const panels = document.querySelectorAll(".screen-panel");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.getAttribute("data-tab");
      tabs.forEach((t) => {
        const active = t === tab;
        t.classList.toggle("is-active", active);
        t.setAttribute("aria-selected", String(active));
      });
      panels.forEach((panel) => {
        panel.classList.toggle(
          "is-active",
          panel.getAttribute("data-panel") === target,
        );
      });
    });
  });

  // Scroll-reveal observer
  const revealItems = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const delay = Number(entry.target.getAttribute("data-delay") || 0);
            setTimeout(() => {
              entry.target.classList.add("is-visible");
            }, delay);
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12 },
    );
    revealItems.forEach((el) => observer.observe(el));
  } else {
    revealItems.forEach((el) => el.classList.add("is-visible"));
  }

  // Live backend status check
  async function checkBackendHealth() {
    try {
      const res = await fetch(API_HEALTH_URL);
      const body = await res.json();
      if (res.ok && body?.data?.status === "ok" && body?.data?.database === "ok") {
        statusDot?.classList.add("online");
        if (statusText) statusText.textContent = "API Online";
      } else {
        if (statusText) statusText.textContent = "Waking Up";
      }
    } catch {
      if (statusText) statusText.textContent = "Offline";
    }
  }

  void checkBackendHealth();
})();
