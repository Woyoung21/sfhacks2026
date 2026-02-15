(function () {
  const queryEl = document.getElementById('query');
  const sendBtn = document.getElementById('send');
  const modeToggle = document.getElementById('mode-toggle');

  const tierEl = document.getElementById('tier-used');
  const latencyEl = document.getElementById('latency');
  const energyEl = document.getElementById('energy');
  const responseEl = document.getElementById('response');

  const ringTier = document.getElementById('ring-tier');
  const ringLatency = document.getElementById('ring-latency');
  const ringEnergy = document.getElementById('ring-energy');

  let mode = 'eco';

  function setRingProgress(el, value01) {
    if (!el) return;
    const pct = Math.max(0, Math.min(1, Number(value01) || 0));
    el.style.setProperty('--pct', `${(pct * 100).toFixed(1)}%`);
  }

  modeToggle.addEventListener('click', (e) => {
    const btn = e.target.closest('.mode-btn');
    if (!btn) return;

    mode = btn.dataset.mode;
    modeToggle.querySelectorAll('.mode-btn').forEach((b) => {
      b.classList.remove('active');
      b.setAttribute('aria-selected', 'false');
    });
    btn.classList.add('active');
    btn.setAttribute('aria-selected', 'true');
  });

  async function sendQuery() {
    const query = queryEl.value.trim();
    if (!query) return;

    sendBtn.disabled = true;
    sendBtn.textContent = 'Routing...';

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, mode }),
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'request failed');
      }

      const tierNum = Number(data.tier_used || 0);
      const latency = Number(data.latency_ms || 0);
      const energy = Number(data.energy_kwh || 0);

      tierEl.textContent = `${data.tier_name} (T${tierNum})`;
      latencyEl.textContent = `${latency.toFixed(1)} ms`;
      energyEl.textContent = `${energy.toFixed(4)} kWh`;
      responseEl.textContent = data.response || '(empty response)';

      setRingProgress(ringTier, tierNum / 3);
      setRingProgress(ringLatency, Math.min(latency / 12000, 1));
      setRingProgress(ringEnergy, Math.min(energy / 0.01, 1));
    } catch (err) {
      responseEl.textContent = `Request failed: ${err.message}`;
      setRingProgress(ringTier, 0);
      setRingProgress(ringLatency, 0);
      setRingProgress(ringEnergy, 0);
    } finally {
      sendBtn.disabled = false;
      sendBtn.textContent = 'Run Route';
    }
  }

  sendBtn.addEventListener('click', sendQuery);

  queryEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  });

  setRingProgress(ringTier, 0);
  setRingProgress(ringLatency, 0);
  setRingProgress(ringEnergy, 0);
})();
