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
    - Also writes epoch timing to <out_dir>/logs/epochs_log.csv and total timing to runtime.txt
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
        self.epoch_csv_path = self.out_dir / "logs" / "epochs_log.csv"
        self.runtime_path = self.out_dir / "runtime.txt"
        self.svg_path = self.out_dir / "plots.svg"
        self.redraw_every = max(1, int(redraw_every))
        self.is_main = bool(is_main)
        self.smoothing = float(smoothing)

        # Buffers
        self.steps = []
        self.l1 = []
        self.lp = []
        self.uv = []
        self.tot = []
        self.lr = []

        self.val_steps = []
        self.val_lp = []
        self.val_l1 = []
        self.val_uv = []
        self.val_tot = []

        # Smoothed
        self._ema_l1 = None
        self._ema_lp = None
        self._ema_tot = None

        # Timing
        self._t0 = time.time()
        self._run_start_wall = None
        self._last_redraw_step = -10**9

        # CSV init
        if self.is_main:
            (self.out_dir / "logs").mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists():
                with open(self.csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["step", "loss_l1", "loss_lpips", "loss_uv", "loss_total", "lr", "val_l1", "val_lpips", "val_uv", "val_total"])

        # Plot init (only on main)
        self._headless = matplotlib.get_backend().lower() == "agg"
        self.fig = None
        self.ax_train = None
        self.ax_val = None
        self.lines = {}  # name -> Line2D

        if self.is_main:
            self._init_plot()

    # --------------- public API ---------------

    def start_run(self):
        """Mark run start and initialize epoch CSV."""
        if not self.is_main:
            return
        self._run_start_wall = time.time()
        if not self.epoch_csv_path.exists():
            with open(self.epoch_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "duration_sec", "steps", "steps_per_sec", "total_elapsed_sec"])

    def end_run(self, total_duration_sec: float):
        """Write total runtime."""
        if not self.is_main:
            return
        try:
            with open(self.runtime_path, "w") as f:
                f.write(f"total_duration_sec,{total_duration_sec:.6f}\n")
                f.write(f"total_duration_min,{total_duration_sec/60.0:.6f}\n")
        except Exception:
            pass

    def log_epoch(self, epoch: int, duration_sec: float, steps: int, steps_per_sec: float):
        """Append per-epoch timing to epoch CSV."""
        if not self.is_main:
            return
        elapsed = (time.time() - self._run_start_wall) if self._run_start_wall else duration_sec
        with open(self.epoch_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([int(epoch), float(duration_sec), int(steps), float(steps_per_sec), float(elapsed)])

    def log_train(self, step: int, loss_l1: float, loss_lp: float, loss_uv: Optional[float], total_loss: Optional[float], lr: float):
        """Record a training step. total_loss can be None; we’ll compute loss_l1+loss_lp if so."""
        if total_loss is None:
            total_loss = float(loss_l1) + float(loss_lp)
        if loss_uv is None:
            loss_uv = 0.0

        self.steps.append(int(step))
        self.l1.append(float(loss_l1))
        self.lp.append(float(loss_lp))
        self.tot.append(float(total_loss))
        self.lr.append(float(lr))
        self.uv.append(float(loss_uv))

        # EMA smoothing (for display only)
        if self.smoothing > 0:
            self._ema_l1 = self._ema(loss_l1, self._ema_l1, self.smoothing)
            self._ema_lp = self._ema(loss_lp, self._ema_lp, self.smoothing)
            self._ema_tot = self._ema(total_loss, self._ema_tot, self.smoothing)

        # Append row to CSV
        if self.is_main:
            self._append_csv(step, loss_l1, loss_lp, loss_uv, total_loss, lr, val_l1=None, val_lp=None, val_uv=None, val_total=None)

        # Redraw if needed
        if self.is_main and (step - self._last_redraw_step) >= self.redraw_every:
            self.redraw()

    def log_val(self, step: int, loss_l1: float, loss_lp: float, loss_uv: float, loss_total: float):
        """Record validation metrics (L1, LPIPS, UV, Total) and update plot."""
        step = int(step)
        self.val_steps.append(step)
        self.val_l1.append(float(loss_l1))
        self.val_lp.append(float(loss_lp))
        self.val_uv.append(float(loss_uv))
        self.val_tot.append(float(loss_total))

        if self.is_main:
            self._append_csv(step,
                            loss_l1=None, loss_lp=None, loss_uv=None, total_loss=None, lr=None,
                            val_l1=loss_l1, val_lp=loss_lp, val_uv=loss_uv, val_total=loss_total)
            self.redraw(force=True)

    def redraw(self, force: bool = False):
        """Update live plot using draw()/pause(). In headless, save a SVG instead."""
        if not self.is_main or self.fig is None:
            return
        if not force and len(self.steps) > 0 and (self.steps[-1] - self._last_redraw_step) < self.redraw_every:
            return

        self._update_train_axes()
        self._update_val_axes()

        self.fig.tight_layout()
        if self._headless:
            self.fig.savefig(self.svg_path, dpi=150)
        else:
            plt.draw()
            plt.pause(0.001)

        self._last_redraw_step = self.steps[-1] if self.steps else self._last_redraw_step

    def save_fig(self, path: Optional[Path] = None):
        if not self.is_main or self.fig is None:
            return
        p = Path(path) if path is not None else self.svg_path
        self.fig.savefig(p, dpi=150)

    def close(self):
        if self.fig is not None and not self._headless:
            try:
                plt.close(self.fig)
            except Exception:
                pass

    # --------------- internals ---------------

    def _init_plot(self):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 5))
        self.ax_train = self.fig.add_subplot(1, 2, 1)
        self.ax_val = self.fig.add_subplot(1, 2, 2)

        (l1_line,) = self.ax_train.plot([], [], label="L1")
        (lp_line,) = self.ax_train.plot([], [], label="LPIPS")
        (uv_line,) = self.ax_train.plot([], [], label="UV")
        (tot_line,) = self.ax_train.plot([], [], label="Total")
        (lr_line,) = self.ax_train.plot([], [], label="LR (scaled)")

        self.lines["l1"] = l1_line
        self.lines["lp"] = lp_line
        self.lines["uv"] = uv_line
        self.lines["tot"] = tot_line
        self.lines["lr"] = lr_line

        (vl1_line,)  = self.ax_val.plot([], [], label="Val L1")
        (vlp_line,)  = self.ax_val.plot([], [], label="Val LPIPS")
        (vuv_line,)  = self.ax_val.plot([], [], label="Val UV")
        (vtot_line,) = self.ax_val.plot([], [], label="Val Total")

        self.lines["val_l1"]  = vl1_line
        self.lines["val_lp"]  = vlp_line
        self.lines["val_uv"]  = vuv_line
        self.lines["val_tot"] = vtot_line

        self.ax_val.set_title("Validation losses")
        self.ax_val.set_xlabel("Step")
        self.ax_val.set_ylabel("Loss")
        self.ax_val.legend(loc="upper right")

        if not self._headless:
            plt.draw()
            plt.pause(0.001)

    def _update_train_axes(self):
        x = self.steps
        if not x:
            return

        l1 = self._series(self.l1, self._ema_l1)
        lp = self._series(self.lp, self._ema_lp)
        tot = self._series(self.tot, self._ema_tot)
        uv = self.uv

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
        self.lines["uv"].set_data(x, uv)

        xmin, xmax = min(x), max(x)
        self.ax_train.set_xlim(xmin, xmax if xmax > xmin else xmin + 1)

        y_vals = []
        y_vals += l1 if l1 else []
        y_vals += lp if lp else []
        y_vals += tot if tot else []
        y_vals += lr_scaled if lr_scaled else []
        y_vals += uv if uv else []
        if y_vals:
            ymin, ymax = min(y_vals), max(y_vals)
            pad = 0.05 * (ymax - ymin + 1e-12)
            self.ax_train.set_ylim(ymin - pad, ymax + pad)

        elapsed = time.time() - self._t0
        self.ax_train.set_title(f"Train losses / LR  |  steps={xmax}  |  {elapsed/60.0:.1f} min")

    def _update_val_axes(self):
        xv = self.val_steps
        if not xv:
            return

        y_l1  = self.val_l1
        y_lp  = self.val_lp
        y_uv  = self.val_uv
        y_tot = self.val_tot

        # Set data for each line
        self.lines["val_l1"].set_data(xv, y_l1)
        self.lines["val_lp"].set_data(xv, y_lp)
        self.lines["val_uv"].set_data(xv, y_uv)
        self.lines["val_tot"].set_data(xv, y_tot)

        # X limits
        xmin, xmax = min(xv), max(xv)
        self.ax_val.set_xlim(xmin, xmax if xmax > xmin else xmin + 1)

        # Y limits from all series
        y_all = []
        if y_l1:  y_all += y_l1
        if y_lp:  y_all += y_lp
        if y_uv:  y_all += y_uv
        if y_tot: y_all += y_tot
        if y_all:
            ymin, ymax = min(y_all), max(y_all)
            pad = 0.05 * (ymax - ymin + 1e-12)
            self.ax_val.set_ylim(ymin - pad, ymax + pad)

        best_lp = min(y_lp) if y_lp else float("nan")
        last_tot = y_tot[-1] if y_tot else float("nan")
        self.ax_val.set_title(f"Validation losses  |  best LPIPS={best_lp:.4f}  |  last TOTAL={last_tot:.4f}")

    def _append_csv(self, step, loss_l1, loss_lp, loss_uv, total_loss, lr, val_l1, val_lp, val_uv, val_total):
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                step,
                loss_l1 if loss_l1 is not None else "",
                loss_lp if loss_lp is not None else "",
                loss_uv if loss_uv is not None else "",
                total_loss if total_loss is not None else "",
                lr if lr is not None else "",
                val_l1 if val_l1 is not None else "",
                val_lp if val_lp is not None else "",
                val_uv if val_uv is not None else "",
                val_total if val_total is not None else "",
            ])

    @staticmethod
    def _ema(x, prev, alpha):
        return x if prev is None else (alpha * prev + (1 - alpha) * x)

    def _series(self, raw_list, ema_value):
        if self.smoothing <= 0 or not raw_list:
            return raw_list
        out = []
        ema = None
        for v in raw_list:
            ema = v if ema is None else (self.smoothing * ema + (1 - self.smoothing) * v)
            out.append(ema)
        return out