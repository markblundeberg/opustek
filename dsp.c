// Sample-by-sample DSP kernels for overtone. Everything that needs a
// recursive per-sample loop lives here; numpy does the rest.
// Build: gcc -O2 -shared -fPIC -o libdsp.so dsp.c -lm  (dsp.py does it)
#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline double polyblep(double t, double dt) {
    if (t < dt) { t /= dt; return t + t - t * t - 1.0; }
    if (t > 1.0 - dt) { t = (t - 1.0) / dt; return t * t + t + t + 1.0; }
    return 0.0;
}

// Sawtooth in [-1, 1] with a per-sample frequency. blep=0 gives the naive
// (aliasing) saw, kept so the alias check can be shown to fail.
void saw(const double *f, double *out, int n, double sr, double phase, int blep) {
    for (int i = 0; i < n; i++) {
        double dt = f[i] / sr;
        double v = 2.0 * phase - 1.0;
        if (blep) v -= polyblep(phase, dt);
        out[i] = v;
        phase += dt;
        if (phase >= 1.0) phase -= 1.0;
    }
}

// Four-pole transistor-ladder-style lowpass (TPT one-poles, tanh on the
// input and in each stage, one-sample feedback). fc per sample.
void ladder(const double *in, double *out, int n, const double *fc,
            double res, double drive, double sr) {
    double s[4] = {0, 0, 0, 0}, y = 0.0;
    double nd = drive > 0 ? tanh(drive) : 1.0;
    for (int i = 0; i < n; i++) {
        double f = fc[i] < 0.45 * sr ? fc[i] : 0.45 * sr;
        double g = tan(M_PI * f / sr);
        double G = g / (1.0 + g);
        double x = in[i] - 4.0 * res * y;
        x = drive > 0 ? tanh(drive * x) / nd : x;
        for (int k = 0; k < 4; k++) {
            double v = (x - s[k]) * G;
            double o = v + s[k];
            s[k] = o + v;
            x = drive > 0 ? tanh(o) : o;
        }
        y = x;
        out[i] = y * (1.0 + res);  // rough make-up for resonance loss
    }
}

// State-variable bandpass (Simper TPT), unity peak gain, fc per sample.
void svf_bp(const double *in, double *out, int n, const double *fc, double q, double sr) {
    double ic1 = 0, ic2 = 0, k = 1.0 / q;
    for (int i = 0; i < n; i++) {
        double f = fc[i] < 0.45 * sr ? fc[i] : 0.45 * sr;
        double g = tan(M_PI * f / sr);
        double a1 = 1.0 / (1.0 + g * (g + k)), a2 = g * a1, a3 = g * a2;
        double v3 = in[i] - ic2;
        double v1 = a1 * ic1 + a2 * v3;
        double v2 = ic2 + a2 * ic1 + a3 * v3;
        ic1 = 2.0 * v1 - ic1;
        ic2 = 2.0 * v2 - ic2;
        out[i] = k * v1;
    }
}

// Stereo ping-pong delay with a damped, cross-fed feedback path.
void pingpong(const double *inL, const double *inR, double *outL, double *outR,
              int n, int D, double fb, double damp) {
    double *bl = calloc(D, sizeof(double)), *br = calloc(D, sizeof(double));
    double lpL = 0, lpR = 0;
    int w = 0;
    for (int i = 0; i < n; i++) {
        double dl = bl[w], dr = br[w];
        lpL += damp * (dl - lpL);
        lpR += damp * (dr - lpR);
        outL[i] = dl; outR[i] = dr;
        bl[w] = inL[i] + fb * lpR;   // cross-feed: L echoes come back on R
        br[w] = inR[i] + fb * lpL;
        if (++w == D) w = 0;
    }
    free(bl); free(br);
}

// 8-line feedback delay network reverb with input diffusion, Householder
// mixing, per-line damping and slow delay modulation for a lusher tail.
void fdn(const double *inL, const double *inR, double *outL, double *outR,
         int n, double sr, double rt60, double damp_hz, double mod_depth) {
    static const int base[8] = {1499, 1889, 2381, 2789, 3253, 3739, 4271, 4799};
    static const int apl[4] = {142, 107, 379, 277};
    double sc = sr / 48000.0;
    int len[8], cap[8], w[8];
    double *buf[8], g[8], lp[8], phs[8];
    for (int j = 0; j < 8; j++) {
        len[j] = (int)(base[j] * sc);
        cap[j] = len[j] + 64;
        buf[j] = calloc(cap[j], sizeof(double));
        w[j] = 0;
        g[j] = pow(10.0, -3.0 * len[j] / (sr * rt60));
        lp[j] = 0;
        phs[j] = j * 0.7;
    }
    double *ab[4]; int al[4], aw[4];
    for (int j = 0; j < 4; j++) {
        al[j] = (int)(apl[j] * sc);
        ab[j] = calloc(al[j], sizeof(double));
        aw[j] = 0;
    }
    double dc = 1.0 - exp(-2.0 * M_PI * damp_hz / sr);
    for (int i = 0; i < n; i++) {
        // input diffusion: two allpasses per channel
        double xl = inL[i], xr = inR[i];
        for (int j = 0; j < 4; j++) {
            double *x = (j < 2) ? &xl : &xr;
            double d = ab[j][aw[j]];
            double v = *x + 0.6 * d;
            ab[j][aw[j]] = v;
            *x = d - 0.6 * v;
            if (++aw[j] == al[j]) aw[j] = 0;
        }
        double y[8], sum = 0;
        for (int j = 0; j < 8; j++) {
            double m = mod_depth * sc * (1.0 + sin(phs[j]));
            phs[j] += 2.0 * M_PI * (0.11 + 0.037 * j) / sr;
            double rp = w[j] - len[j] + m;
            while (rp < 0) rp += cap[j];
            int i0 = (int)rp; double fr = rp - i0;
            int i1 = i0 + 1; if (i1 >= cap[j]) i1 -= cap[j];
            double d = buf[j][i0] * (1 - fr) + buf[j][i1] * fr;
            lp[j] += dc * (d - lp[j]);
            y[j] = lp[j] * g[j];
            sum += y[j];
        }
        double L = 0, R = 0;
        for (int j = 0; j < 8; j++) {
            double fbk = y[j] - 0.25 * sum;  // Householder, N=8
            double inj = (j % 2 == 0) ? xl : xr;
            buf[j][w[j]] = fbk + inj;
            if (++w[j] == cap[j]) w[j] = 0;
            if (j % 2 == 0) L += y[j]; else R += y[j];
        }
        outL[i] = L * 0.35; outR[i] = R * 0.35;
    }
    for (int j = 0; j < 8; j++) free(buf[j]);
    for (int j = 0; j < 4; j++) free(ab[j]);
}

// Gain smoother for the limiter: instant attack, exponential release.
void release_smooth(const double *g, double *out, int n, double rel) {
    double s = 1.0;
    for (int i = 0; i < n; i++) {
        s += (1.0 - s) * rel;
        if (g[i] < s) s = g[i];
        out[i] = s;
    }
}

// Tape-style echo: a delay whose time wobbles (wow), so each repeat bends
// in pitch, with damped feedback. Run it per channel with different phases.
void tapedelay(const double *in, double *out, int n, double sr, double delay_s,
               double wow_ms, double wow_hz, double fb, double damp_hz, double phase) {
    int cap = (int)((delay_s + wow_ms * 2e-3) * sr) + 8;
    double *buf = calloc(cap, sizeof(double));
    double lp = 0, dc = 1.0 - exp(-2.0 * M_PI * damp_hz / sr);
    double ph = phase;
    int w = 0;
    for (int i = 0; i < n; i++) {
        double d = (delay_s + wow_ms * 1e-3 * (1.0 + sin(ph)) ) * sr;
        ph += 2.0 * M_PI * wow_hz / sr;
        double rp = w - d;
        while (rp < 0) rp += cap;
        int i0 = (int)rp; double fr = rp - i0;
        int i1 = i0 + 1; if (i1 >= cap) i1 -= cap;
        double y = buf[i0] * (1 - fr) + buf[i1] * fr;
        lp += dc * (y - lp);
        out[i] = lp;
        buf[w] = in[i] + fb * tanh(lp);
        if (++w == cap) w = 0;
    }
    free(buf);
}

// Hard-sync sawtooth: a slave saw at f*ratio restarts whenever the master at
// f completes a cycle, so its pitch stays the master's and its brightness
// peak moves with the ratio. Both the slave's own wrap and the sync reset are
// band-limited with polyBLEP; the output runs one sample late so the
// residual before each step can be added.
static inline void blep_add(double *prev, double *cur, double x, double h) {
    // x: time from the step to the current sample, in samples, in [0, 1)
    *prev += h * x * x * 0.5;
    *cur  -= h * (1.0 - x) * (1.0 - x) * 0.5;
}
void syncsaw(const double *f, const double *ratio, double *out, int n, double sr, double phase) {
    double pm = phase, ps = phase, prev = 0.0;
    for (int i = 0; i < n; i++) {
        double dtm = f[i] / sr, dts = dtm * ratio[i];
        pm += dtm;
        double psn = ps + dts;                      // slave, unwrapped
        double cur;
        if (pm >= 1.0) {
            pm -= 1.0;
            double xm = pm / dtm;                   // master wrapped xm samples ago
            double at = psn - xm * dts;             // slave phase at that instant
            double pend = 0.0, xw = -1.0;
            if (at >= 1.0) {                        // slave wrapped first, inside this sample
                xw = xm + (at - 1.0) / dts;
                at -= 1.0;
            }
            double before = 2.0 * at - 1.0;
            ps = xm * dts;
            while (ps >= 1.0) ps -= 1.0;
            cur = 2.0 * ps - 1.0;
            if (xw >= 0.0 && xw < 1.0) blep_add(&prev, &cur, xw, -2.0);
            blep_add(&prev, &cur, xm, (-1.0) - before);
            (void)pend;
        } else {
            ps = psn;
            double xs = -1.0;
            if (ps >= 1.0) { ps -= 1.0; xs = ps / dts; }
            cur = 2.0 * ps - 1.0;
            if (xs >= 0.0) blep_add(&prev, &cur, xs, -2.0);
        }
        if (i > 0) out[i - 1] = prev;
        prev = cur;
    }
    if (n > 0) out[n - 1] = prev;
}

// Phaser: a chain of first-order allpasses whose corner sweeps with a slow
// LFO (log between fmin and fmax), with feedback; output = (dry + wet) / 2,
// so the notches move through the spectrum.
void phaser(const double *in, double *out, int n, double sr, double rate, double fmin,
            double fmax, int stages, double fb, double phase) {
    double z[16] = {0}, y = 0.0, ph = phase;
    if (stages > 16) stages = 16;
    for (int i = 0; i < n; i++) {
        double l = 0.5 + 0.5 * sin(ph);
        ph += 2.0 * M_PI * rate / sr;
        double fc = fmin * pow(fmax / fmin, l);
        double t = tan(M_PI * fc / sr);
        double a = (t - 1.0) / (t + 1.0);
        double x = in[i] + fb * y;
        for (int s = 0; s < stages; s++) {
            double v = a * x + z[s];
            z[s] = x - a * v;
            x = v;
        }
        y = x;
        out[i] = 0.5 * (in[i] + x);
    }
}

// Karplus-Strong string/bar: an excitation fed into a delay loop with a
// two-point average (half a sample of delay) and loss. The loop length is
// fractional (linear interpolation) so the pitch is exact: rounding it to
// whole samples puts the high notes up to 12 cents out of tune.
// int_delay=1 rounds it anyway, for the tuning check's self-test.
void ks(const double *exc, double *out, int n, double sr, double freq, double fb,
        double bright, int int_delay) {
    double D = sr / freq - 0.5 * (1.0 - bright);   // the (partial) average adds this much delay
    if (int_delay) D = (double)(int)(sr / freq);
    int cap = (int)D + 4;
    double *buf = calloc(cap, sizeof(double));
    double prev = 0.0;
    int w = 0;
    for (int i = 0; i < n; i++) {
        double rp = w - D;
        while (rp < 0) rp += cap;
        int i0 = (int)rp; double fr = rp - i0;
        int i1 = i0 + 1; if (i1 >= cap) i1 -= cap;
        double d = buf[i0] * (1 - fr) + buf[i1] * fr;
        double avg = bright * d + (1 - bright) * 0.5 * (d + prev);   // bright=1: no averaging
        prev = d;
        double y = exc[i] + fb * avg;
        buf[w] = y;
        out[i] = y;
        if (++w == cap) w = 0;
    }
    free(buf);
}
