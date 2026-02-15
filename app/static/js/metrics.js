(function () {
  const el = {
    total: document.getElementById('m-total'),
    cache: document.getElementById('m-cache'),
    escalations: document.getElementById('m-escalations'),
    latency: document.getElementById('m-latency'),
    energy: document.getElementById('m-energy'),
    saved: document.getElementById('m-saved'),
    t1: document.getElementById('t1'),
    t2: document.getElementById('t2'),
    t3: document.getElementById('t3'),
    b1: document.getElementById('b1'),
    b2: document.getElementById('b2'),
    b3: document.getElementById('b3'),
    recent: document.getElementById('recent'),
  };

  function setBar(node, count, total) {
    const pct = total > 0 ? (count / total) * 100 : 0;
    node.style.width = `${pct.toFixed(1)}%`;
  }

  function renderRecent(items) {
    if (!items || !items.length) {
      el.recent.textContent = 'No requests yet.';
      return;
    }
    el.recent.innerHTML = items
      .slice()
      .reverse()
      .map((r) => {
        const q = (r.query || '').replace(/</g, '&lt;');
        const reason = (r.routing_reason || '').replace(/</g, '&lt;');
        return `
          <div class="recent-item">
            <div class="recent-meta">${r.timestamp} • ${r.tier_name} • ${r.latency_ms.toFixed(1)}ms • ${r.energy_kwh}kWh</div>
            <div class="recent-text">${q}</div>
            <div class="recent-meta">${reason}</div>
          </div>`;
      })
      .join('');
  }

  let inFlight = false;

  async function tick() {
    if (inFlight || document.hidden) return;
    inFlight = true;
    try {
      const res = await fetch('/api/metrics');
      const data = await res.json();
      const live = data.live || {};
      const tiers = live.tier_counts || {};

      el.total.textContent = live.total_requests ?? 0;
      el.cache.textContent = live.cache_hits ?? 0;
      el.escalations.textContent = live.escalations ?? 0;
      el.latency.textContent = `${live.avg_latency_ms ?? 0} ms`;
      el.energy.textContent = `${live.total_energy_kwh ?? 0} kWh`;
      el.saved.textContent = `${live.energy_saved_kwh ?? 0} kWh (${live.energy_saved_pct ?? 0}%)`;

      const total = Number(live.total_requests || 0);
      const t1 = Number(tiers[1] || 0);
      const t2 = Number(tiers[2] || 0);
      const t3 = Number(tiers[3] || 0);
      el.t1.textContent = t1;
      el.t2.textContent = t2;
      el.t3.textContent = t3;
      setBar(el.b1, t1, total);
      setBar(el.b2, t2, total);
      setBar(el.b3, t3, total);

      renderRecent(data.recent_requests || []);
    } catch (err) {
      el.recent.textContent = `metrics error: ${err.message}`;
    } finally {
      inFlight = false;
    }
  }

  tick();
  setInterval(tick, 8000);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) tick();
  });
})();
