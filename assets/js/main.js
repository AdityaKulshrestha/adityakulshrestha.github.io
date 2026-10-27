const side = document.querySelector('.toc-side');
if (side) {
  const pairs = [...side.querySelectorAll('a[href^="#"]')]
    .map((a) => [a, document.getElementById(decodeURIComponent(a.hash.slice(1)))])
    .filter(([, heading]) => heading);
  let queued = false;
  const update = () => {
    queued = false;
    let current = pairs[0];
    for (const pair of pairs) if (pair[1].getBoundingClientRect().top <= 120) current = pair;
    pairs.forEach(([a]) => a.classList.toggle('active', a === current[0]));
  };
  if (pairs.length) {
    addEventListener('scroll', () => { if (!queued) { queued = true; requestAnimationFrame(update); } }, { passive: true });
    update();
  }
}

document.querySelectorAll('[data-slideshow]').forEach((root) => {
  const track = root.querySelector('.track');
  const step = (dir) => {
    const atEnd = track.scrollLeft + track.clientWidth >= track.scrollWidth - 4;
    if (dir > 0 && atEnd) track.scrollTo({ left: 0, behavior: 'smooth' });
    else track.scrollBy({ left: dir * track.clientWidth, behavior: 'smooth' });
  };
  root.querySelector('.prev').addEventListener('click', () => step(-1));
  root.querySelector('.next').addEventListener('click', () => step(1));

  let paused = false;
  root.addEventListener('mouseenter', () => (paused = true));
  root.addEventListener('mouseleave', () => (paused = false));
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
    setInterval(() => { if (!paused && !document.hidden) step(1); }, 4500);
  }
});
