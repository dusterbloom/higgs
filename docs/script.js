(function () {
  const revealItems = document.querySelectorAll('.reveal');
  const counters = document.querySelectorAll('[data-count]');

  const revealObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('visible');
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.12 });

  revealItems.forEach((item) => revealObserver.observe(item));

  function animateCounter(el) {
    const end = Number.parseFloat(el.dataset.count);
    if (!Number.isFinite(end)) return;

    const decimals = el.dataset.count.includes('.') ? 3 : 0;
    const start = performance.now();
    const duration = 900;

    function tick(now) {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      const value = end * eased;
      el.textContent = value.toFixed(decimals).replace(/0+$/, '').replace(/\.$/, '');
      if (t < 1) requestAnimationFrame(tick);
    }

    requestAnimationFrame(tick);
  }

  const counterObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      animateCounter(entry.target);
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.7 });

  counters.forEach((counter) => counterObserver.observe(counter));
})();
