# utils/train_tracker.py
# Made with the help of ChatGPT5
import os
import csv
import time
from pathlib import Path
from typing import Optional

import matplotlib
# Prefer interactive backend if available; otherwise fall back to Agg
if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND", "") == "":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


class StatTracker:
    """
    Lightweight training/validation tracker with live plots via matplotlib draw().
    - Records: step, l1, lpips, total_loss, lr, val_lpips
    - Writes: <out_dir>/logs/train_log.csv
    - Plots:  2 panels -> Train (losses/LR) and Validation (LPIPS)
    - Headless-safe: if backend is Agg, it will save PNGs periodically instead of showing a GUI.
    """

    def __init__(
        self,
        out_dir: Path,
        redraw_every: int = 50,
        smoothing: float = 0.0,  # EMA smoothing for plotted loss curves (0 = off)
        is_main: bool = True,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "logs" / "train_log.csv"
        self.png_path = self.out_dir / "plots.png"
        self.redraw_every = max(1, int(redraw_every))
        self.is_main = bool(is_main)
        self.smoothing = float(smoothing)

        # Buffers
        self.steps = []
        self.l1 = []
        self.lp = []
        self.tot = []
        self.lr = []

        self.val_steps = []
        self.val_lp = []

        # Smoothed
        self._ema_l1 = None
        self._ema_lp = None
        self._ema_tot = None

        # Timing
        self._t0 = time.time()
        self._last_redraw_step = -10**9

        # CSV init
        if self.is_main:
            (self.out_dir / "logs").mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists():
                with open(self.csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["step", "loss_l1", "loss_lpips", "loss_total", "lr", "val_lpips"])

        # Plot init (only on main)
        self._headless = matplotlib.get_backend().lower() == "agg"
        self.fig = None
        self.ax_train = None
        self.ax_val = None
        self.lines = {}  # name -> Line2D

        if self.is_main:
            self._init_plot()

    # --------------- public API ---------------

    def log_train(self, step: int, loss_l1: float, loss_lp: float, total_loss: Optional[float], lr: float):
        """Record a training step. total_loss can be None; we’ll compute loss_l1+loss_lp if so."""
        if total_loss is None:
            total_loss = float(loss_l1) + float(loss_lp)

        self.steps.append(int(step))
        self.l1.append(float(loss_l1))
        self.lp.append(float(loss_lp))
        self.tot.append(float(total_loss))
        self.lr.append(float(lr))

        # EMA smoothing (for display only)
        if self.smoothing > 0:
            self._ema_l1 = self._ema(loss_l1, self._ema_l1, self.smoothing)
            self._ema_lp = self._ema(loss_lp, self._ema_lp, self.smoothing)
            self._ema_tot = self._ema(total_loss, self._ema_tot, self.smoothing)

        # Append row to CSV
        if self.is_main:
            self._append_csv(step, loss_l1, loss_lp, total_loss, lr, val_lp=None)

        # Redraw if needed
        if self.is_main and (step - self._last_redraw_step) >= self.redraw_every:
            self.redraw()

    def log_val(self, step: int, avg_lpips: float):
        """Record a validation measurement."""
        self.val_steps.append(int(step))
        self.val_lp.append(float(avg_lpips))
        if self.is_main:
            # Add a CSV line with val filled, train columns left as last-known or blank
            self._append_csv(step, "", "", "", "", val_lp=avg_lpips)
            self.redraw(force=True)

    def redraw(self, force: bool = False):
        """Update live plot using draw()/pause(). In headless, save a PNG instead."""
        if not self.is_main or self.fig is None:
            return
        if not force and len(self.steps) > 0 and (self.steps[-1] - self._last_redraw_step) < self.redraw_every:
            return

        self._update_train_axes()
        self._update_val_axes()

        # Tighten and draw
        self.fig.tight_layout()
        if self._headless:
            self.fig.savefig(self.png_path, dpi=150)
        else:
            plt.draw()
            # a tiny pause to process GUI events; does not stall training
            plt.pause(0.001)

        self._last_redraw_step = self.steps[-1] if self.steps else self._last_redraw_step

    def save_fig(self, path: Optional[Path] = None):
        if not self.is_main or self.fig is None:
            return
        p = Path(path) if path is not None else self.png_path
        self.fig.savefig(p, dpi=150)

    def close(self):
        if self.fig is not None and not self._headless:
            try:
                plt.close(self.fig)
            except Exception:
                pass

    # --------------- internals ---------------

    def _init_plot(self):
        plt.ion()  # enable interactive mode (safe if headless; ignored by Agg)
        self.fig = plt.figure(figsize=(10, 5))
        self.ax_train = self.fig.add_subplot(1, 2, 1)
        self.ax_val = self.fig.add_subplot(1, 2, 2)

        # Create persistent line objects (avoid replot overhead)
        (l1_line,) = self.ax_train.plot([], [], label="L1")
        (lp_line,) = self.ax_train.plot([], [], label="LPIPS")
        (tot_line,) = self.ax_train.plot([], [], label="Total")
        (lr_line,) = self.ax_train.plot([], [], label="LR (scaled)")

        self.lines["l1"] = l1_line
        self.lines["lp"] = lp_line
        self.lines["tot"] = tot_line
        self.lines["lr"] = lr_line

        (v_line,) = self.ax_val.plot([], [], label="Val LPIPS")
        self.lines["val"] = v_line

        self.ax_train.set_title("Train losses / LR")
        self.ax_train.set_xlabel("Step")
        self.ax_train.set_ylabel("Loss")
        self.ax_train.legend(loc="upper right")

        self.ax_val.set_title("Validation LPIPS")
        self.ax_val.set_xlabel("Step")
        self.ax_val.set_ylabel("LPIPS")
        self.ax_val.legend(loc="upper right")

        # First draw so window appears
        if not self._headless:
            plt.draw()
            plt.pause(0.001)

    def _update_train_axes(self):
        x = self.steps
        if not x:
            return

        # choose raw or smoothed for plotting
        l1 = self._series(self.l1, self._ema_l1)
        lp = self._series(self.lp, self._ema_lp)
        tot = self._series(self.tot, self._ema_tot)

        # Scale LR to fit (divide by its max to keep it ~[0,1] then multiply by median(tot) for visibility)
        if self.lr:
            lrmax = max(self.lr)
            scale = max(1e-12, lrmax)
            base = max(1e-6, (sum(tot) / len(tot)) if tot else 1.0)
            lr_scaled = [v / scale * base for v in self.lr]
        else:
            lr_scaled = []

        self.lines["l1"].set_data(x, l1)
        self.lines["lp"].set_data(x, lp)
        self.lines["tot"].set_data(x, tot)
        self.lines["lr"].set_data(x, lr_scaled)

        # Update limits
        xmin, xmax = min(x), max(x)
        self.ax_train.set_xlim(xmin, xmax if xmax > xmin else xmin + 1)

        y_vals = []
        y_vals += l1 if l1 else []
        y_vals += lp if lp else []
        y_vals += tot if tot else []
        y_vals += lr_scaled if lr_scaled else []
        if y_vals:
            ymin, ymax = min(y_vals), max(y_vals)
            pad = 0.05 * (ymax - ymin + 1e-12)
            self.ax_train.set_ylim(ymin - pad, ymax + pad)

        # progress subtitle
        elapsed = time.time() - self._t0
        self.ax_train.set_title(f"Train losses / LR  |  steps={xmax}  |  {elapsed/60.0:.1f} min")

    def _update_val_axes(self):
        xv = self.val_steps
        yv = self.val_lp
        if not xv:
            return

        self.lines["val"].set_data(xv, yv)

        xmin, xmax = min(xv), max(xv)
        self.ax_val.set_xlim(xmin, xmax if xmax > xmin else xmin + 1)

        ymin, ymax = min(yv), max(yv)
        pad = 0.05 * (ymax - ymin + 1e-12)
        self.ax_val.set_ylim(ymin - pad, ymax + pad)

        best = min(yv)
        self.ax_val.set_title(f"Validation LPIPS  |  best={best:.4f}")

    def _append_csv(self, step, loss_l1, loss_lp, total_loss, lr, val_lp):
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([step, loss_l1, loss_lp, total_loss, lr, val_lp if val_lp is not None else ""])

    @staticmethod
    def _ema(x, prev, alpha):
        return x if prev is None else (alpha * prev + (1 - alpha) * x)

    def _series(self, raw_list, ema_value):
        if self.smoothing <= 0 or not raw_list:
            return raw_list
        # Rebuild smoothed series efficiently from last known EMA
        out = []
        ema = None
        for v in raw_list:
            ema = v if ema is None else (self.smoothing * ema + (1 - self.smoothing) * v)
            out.append(ema)
        return out