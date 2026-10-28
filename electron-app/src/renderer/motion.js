// Motion layer: every animation in the app goes through here so timing stays consistent
// and prefers-reduced-motion is honoured in one place.
(function () {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const hasGsap = typeof window.gsap !== 'undefined';
    const EASE = 'expo.out';

    if (hasGsap) gsap.defaults({ ease: EASE, duration: 0.6 });

    function run(fn, fallback) {
        if (!hasGsap || reduced) { if (fallback) fallback(); return null; }
        return fn();
    }

    // Opening sequence: shell settles in once, quickly
    function intro() {
        const reveal = () => document.documentElement.classList.remove('is-booting');
        run(() => {
            const tl = gsap.timeline({ defaults: { duration: 0.7 } });
            // .from() tweens render their start state immediately, so un-hide right after building
            tl.eventCallback('onStart', reveal);
            setTimeout(reveal, 0);
            tl.from('.topbar > *', { opacity: 0, y: -6, stagger: 0.04 })
              .from('.stage-head > *', { opacity: 0, y: 10, stagger: 0.06 }, 0.08)
              .from('.plot-frame', { opacity: 0, y: 14, scale: 0.995 }, 0.14)
              .from('.readout-cell', { opacity: 0, y: 10, stagger: 0.05 }, 0.24)
              .from('.inspector .panel', { opacity: 0, x: 12, stagger: 0.06 }, 0.2)
              .from('.empty-content > *', { opacity: 0, y: 12, stagger: 0.06 }, 0.3);
            return tl;
        }, reveal);
    }

    // Synthetic seismogram for the splash: noise floor, P arrival, larger S arrival, coda
    function seismogramPath(width, height, pX, sX) {
        let seed = 7;
        const rnd = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280 - 0.5; };
        const mid = height / 2;
        let d = `M0 ${mid}`;
        for (let x = 2; x <= width; x += 2) {
            let a = 0.03;
            if (x > pX) a += 0.32 * Math.exp(-(x - pX) / 38);
            if (x > sX) a += 0.9 * Math.exp(-(x - sX) / 70);
            d += `L${x} ${(mid + rnd() * a * height).toFixed(1)}`;
        }
        return d;
    }

    // Launch sequence, then hands over to intro(). Total about 2 s.
    function splash(done) {
        const el = document.getElementById('splash');
        if (!el) { done(); return; }
        if (!hasGsap || reduced) { el.remove(); done(); return; }

        const path = document.getElementById('splash-path');
        path.setAttribute('d', seismogramPath(720, 120, 268, 392));
        const len = path.getTotalLength();

        const word = el.querySelector('.splash-word');
        word.innerHTML = word.textContent.split('').map(c => `<span class="ch">${c}</span>`).join('');

        const tl = gsap.timeline({ onComplete: () => el.remove() });
        tl.set(path, { strokeDasharray: len, strokeDashoffset: len })
          .from('.splash-baseline', { scaleX: 0, transformOrigin: '0 50%', duration: 0.6, ease: 'power3.out' })
          .to(path, { strokeDashoffset: 0, duration: 1.1, ease: 'power2.inOut' }, 0.15)
          .to('.splash-pick-p', { opacity: 1, duration: 0.25 }, 0.62)
          .to('.splash-pick-s', { opacity: 1, duration: 0.25 }, 0.8)
          .from('.splash-mark', { opacity: 0, scale: 0.7, rotate: -8, duration: 0.6, ease: 'back.out(2)' }, 0.55)
          .from('.splash-word .ch', { opacity: 0, y: 14, duration: 0.5, stagger: 0.025 }, 0.62)
          .from('.splash-caption', { opacity: 0, y: 6, duration: 0.5 }, 0.95)
          // Hand-off: splash lifts away while the workspace builds in underneath
          .to(el, { opacity: 0, y: -16, duration: 0.55, ease: 'power2.in' }, 1.6)
          .add(done, 1.75);

        // Any key or click skips straight to the app
        const skip = () => { tl.progress(1); window.removeEventListener('keydown', skip); };
        window.addEventListener('keydown', skip, { once: true });
        el.addEventListener('click', skip, { once: true });
    }

    // Animated number. format(value) returns the display string.
    function countUp(el, to, format, opts = {}) {
        if (!el) return;
        const target = Number(to);
        if (!Number.isFinite(target)) { el.textContent = format(to); return; }
        run(() => {
            const state = { v: opts.from ?? 0 };
            return gsap.to(state, {
                v: target, duration: opts.duration ?? 1.1, delay: opts.delay ?? 0,
                onUpdate: () => { el.textContent = format(state.v); },
                onComplete: () => { el.textContent = format(target); }
            });
        }, () => { el.textContent = format(target); });
    }

    function meter(el, pct, delay = 0) {
        if (!el) return;
        const w = `${Math.max(0, Math.min(100, pct))}%`;
        run(() => gsap.fromTo(el, { width: '0%' }, { width: w, duration: 1.2, delay }),
            () => { el.style.width = w; });
    }

    // Waveform traces draw in from left to right
    function revealTraces(plotDiv) {
        const layers = plotDiv.querySelectorAll('.cartesianlayer .scatterlayer');
        run(() => gsap.fromTo(layers,
            { clipPath: 'inset(0 100% 0 0)' },
            { clipPath: 'inset(0 0% 0 0)', duration: 1.4, ease: 'power3.inOut', stagger: 0.08, clearProps: 'clipPath' }));
    }

    // Pick markers drop in after the traces
    function revealPicks(plotDiv, delay = 0) {
        const shapes = plotDiv.querySelectorAll('.layer-above .shapelayer path, .layer-below .shapelayer path');
        const notes = plotDiv.querySelectorAll('.annotation[data-index]');
        const pickNotes = Array.from(notes).filter(n => n.querySelector('rect.bg'));
        run(() => {
            const tl = gsap.timeline({ delay });
            tl.fromTo(shapes, { opacity: 0 }, { opacity: 1, duration: 0.6, stagger: 0.08, clearProps: 'opacity' })
              .fromTo(pickNotes, { opacity: 0, y: -8 }, { opacity: 1, y: 0, duration: 0.6, stagger: 0.1, clearProps: 'opacity,transform' }, 0.05);
            return tl;
        });
    }

    function fadeIn(el, opts = {}) {
        if (!el) return;
        el.hidden = false;
        run(() => gsap.fromTo(el, { opacity: 0, y: opts.y ?? 12 }, { opacity: 1, y: 0, duration: 0.5, clearProps: 'transform' }));
    }

    function fadeOut(el, delay = 0) {
        if (!el) return Promise.resolve();
        return new Promise(resolve => {
            run(() => gsap.to(el, {
                opacity: 0, y: 8, duration: 0.4, delay, ease: 'power2.in',
                onComplete: () => { el.hidden = true; gsap.set(el, { clearProps: 'opacity,transform' }); resolve(); }
            }), () => { el.hidden = true; resolve(); });
        });
    }

    // Empty state gives way to the plot
    function leaveEmpty(frame, emptyEl, onSwap) {
        run(() => gsap.to(emptyEl, {
            opacity: 0, scale: 0.985, duration: 0.35, ease: 'power2.in',
            onComplete: () => { onSwap(); gsap.set(emptyEl, { clearProps: 'all' }); }
        }), onSwap);
    }

    function stagger(els, delay = 0) {
        if (!els || !els.length) return;
        run(() => gsap.from(els, { opacity: 0, y: 8, duration: 0.55, stagger: 0.06, delay, clearProps: 'opacity,transform' }));
    }

    function openDrawer(root) {
        root.hidden = false;
        const drawer = root.querySelector('.drawer');
        const scrim = root.querySelector('.drawer-scrim');
        run(() => {
            const tl = gsap.timeline();
            tl.fromTo(scrim, { opacity: 0 }, { opacity: 1, duration: 0.3, ease: 'power1.out' })
              .fromTo(drawer, { xPercent: 100 }, { xPercent: 0, duration: 0.55 }, 0)
              .from(drawer.querySelectorAll('.drawer-body > *'), { opacity: 0, x: 16, stagger: 0.04, duration: 0.45 }, 0.12);
            return tl;
        });
    }

    function closeDrawer(root) {
        const drawer = root.querySelector('.drawer');
        const scrim = root.querySelector('.drawer-scrim');
        run(() => {
            const tl = gsap.timeline({ onComplete: () => { root.hidden = true; gsap.set([drawer, scrim], { clearProps: 'all' }); } });
            tl.to(drawer, { xPercent: 100, duration: 0.35, ease: 'power3.in' })
              .to(scrim, { opacity: 0, duration: 0.3 }, 0);
            return tl;
        }, () => { root.hidden = true; });
    }

    function toastIn(el) {
        run(() => gsap.fromTo(el, { opacity: 0, y: 16, scale: 0.97 }, { opacity: 1, y: 0, scale: 1, duration: 0.5 }));
    }

    function toastOut(el) {
        run(() => gsap.to(el, {
            opacity: 0, x: -24, duration: 0.3, ease: 'power2.in',
            onComplete: () => el.remove()
        }), () => el.remove());
    }

    // Smooth scrolling for the inspector column only
    function initLenis(wrapper, content) {
        if (reduced || typeof window.Lenis === 'undefined' || !hasGsap) return null;
        const lenis = new Lenis({ wrapper, content, lerp: 0.12, wheelMultiplier: 0.9 });
        gsap.ticker.add(time => lenis.raf(time * 1000));
        gsap.ticker.lagSmoothing(0);
        return lenis;
    }

    // Faint live seismograph behind the empty state
    function emptyTrace(canvas) {
        const ctx = canvas.getContext('2d');
        let w = 0, h = 0, dpr = 1, running = true, t = 0;
        const samples = [];
        const colors = {};

        // Token colours, mixed to the trace's opacities; refreshed when the theme changes
        function readColors() {
            const css = getComputedStyle(document.documentElement);
            colors.faint = `color-mix(in srgb, ${css.getPropertyValue('--text').trim()} 12%, transparent)`;
            colors.head = `color-mix(in srgb, ${css.getPropertyValue('--accent-line').trim()} 45%, transparent)`;
            // Canvas needs concrete colours: resolve color-mix through a probe element
            const probe = document.createElement('span');
            document.body.appendChild(probe);
            ['faint', 'head'].forEach(k => { probe.style.color = colors[k]; colors[k] = getComputedStyle(probe).color; });
            probe.remove();
        }
        readColors();
        document.addEventListener('themechange', () => { readColors(); if (reduced) { running = true; frame(); running = false; } });

        function resize() {
            dpr = window.devicePixelRatio || 1;
            w = canvas.clientWidth; h = canvas.clientHeight;
            canvas.width = w * dpr; canvas.height = h * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }

        // Noise floor with an occasional P then S burst
        let burst = 0, burstAmp = 0;
        function sample() {
            t += 1;
            if (burst <= 0 && Math.random() < 0.004) { burst = 260; burstAmp = 0.5 + Math.random() * 0.5; }
            let v = (Math.random() - 0.5) * 0.06 + Math.sin(t * 0.05) * 0.015;
            if (burst > 0) {
                const age = 260 - burst;
                const p = age < 90 ? Math.exp(-age / 40) * 0.35 : 0;
                const s = age >= 90 ? Math.exp(-(age - 90) / 55) : 0;
                v += (Math.random() - 0.5) * burstAmp * (p + s) * 1.6;
                burst -= 1;
            }
            return v;
        }

        function frame() {
            if (!running) return;
            if (canvas.offsetParent !== null) {
                for (let i = 0; i < 3; i++) samples.push(sample());
                const max = Math.ceil(w / 1.5);
                while (samples.length > max) samples.shift();
                ctx.clearRect(0, 0, w, h);
                // Trace runs along the lower part of the frame, clear of the copy
                const mid = h * 0.8;
                const grad = ctx.createLinearGradient(0, 0, w, 0);
                grad.addColorStop(0, 'transparent');
                grad.addColorStop(0.7, colors.faint);
                grad.addColorStop(1, colors.head);
                ctx.strokeStyle = grad;
                ctx.lineWidth = 1;
                ctx.beginPath();
                const offset = w - samples.length * 1.5;
                samples.forEach((v, i) => {
                    const x = offset + i * 1.5, y = mid + Math.max(-1, Math.min(1, v)) * h * 0.13;
                    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
                });
                ctx.stroke();
            }
            requestAnimationFrame(frame);
        }

        resize();
        window.addEventListener('resize', resize);
        // Prefill so the trace spans the canvas from the first frame
        const fill = Math.ceil((w || 1200) / 1.5);
        for (let i = 0; i < fill; i++) samples.push(sample());
        if (reduced) {
            frame(); running = false;
        } else {
            frame();
        }
        return { stop() { running = false; } };
    }

    // Theme swap: brief cross-fade on colours, then let canvas and plot re-read tokens
    function themeSwap(apply) {
        const root = document.documentElement;
        if (!reduced) root.classList.add('theme-switching');
        apply();
        document.dispatchEvent(new Event('themechange'));
        setTimeout(() => root.classList.remove('theme-switching'), 360);
    }

    window.Motion = {
        themeSwap, splash,
        reduced, intro, countUp, meter, revealTraces, revealPicks, fadeIn, fadeOut,
        leaveEmpty, stagger, openDrawer, closeDrawer, toastIn, toastOut, initLenis, emptyTrace
    };
})();
