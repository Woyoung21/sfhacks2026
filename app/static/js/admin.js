(function () {
  const modeEl = document.getElementById('default-mode');
  const confEl = document.getElementById('hint-conf');
  const cacheSyncEl = document.getElementById('cache-sync');
  const saveBtn = document.getElementById('save-admin');
  const statusEl = document.getElementById('admin-status');

  async function loadConfig() {
    const res = await fetch('/api/admin');
    const data = await res.json();
    const cfg = data.config || {};

    modeEl.value = cfg.default_mode || 'eco';
    confEl.value = cfg.vector_hint_min_conf ?? 0.7;
    cacheSyncEl.checked = !!cfg.cache_write_sync;
    statusEl.textContent = JSON.stringify(data, null, 2);
  }

  async function saveConfig() {
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';
    try {
      const payload = {
        default_mode: modeEl.value,
        vector_hint_min_conf: Number(confEl.value),
        cache_write_sync: cacheSyncEl.checked,
      };

      const res = await fetch('/api/admin', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      statusEl.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      statusEl.textContent = `error: ${err.message}`;
    } finally {
      saveBtn.disabled = false;
      saveBtn.textContent = 'Save Admin Config';
    }
  }

  saveBtn.addEventListener('click', saveConfig);
  loadConfig().catch((err) => {
    statusEl.textContent = `error: ${err.message}`;
  });
})();
