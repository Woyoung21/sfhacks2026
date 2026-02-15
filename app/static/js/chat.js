(function () {
  const queryEl = document.getElementById('query');
  const clearBtn = document.getElementById('clear-btn');
  const modeToggle = document.getElementById('mode-toggle');

  let mode = 'eco';

  // Auto-expand textarea height
  function autoExpandTextarea() {
    queryEl.style.height = '50px';
    const scrollHeight = queryEl.scrollHeight;
    queryEl.style.height = scrollHeight + 'px';
  }

  queryEl.addEventListener('input', autoExpandTextarea);

  // Clear button functionality
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      queryEl.value = '';
      queryEl.style.height = '50px';
      queryEl.focus();
    });
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

  function sendQuery() {
    const query = queryEl.value.trim();
    if (!query) return;

    const encodedQuery = encodeURIComponent(query);
    window.location.href = `/results?query=${encodedQuery}&mode=${mode}`;
  }

  queryEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  });
})();
