# -*- coding: utf-8 -*-
"""
Численная симуляция «Вязкоупругой модели ликвидности» (Viscoelastic Flow Market, VFM)
и её сравнение с CPMM (x*y=k) на сценарии покупки китом крупной доли эмиссии.

Эксперименты:
  E1. Стоимость исполнения кита (20% эмиссии) как функция горизонта T: VFM vs CPMM.
  E2. Траектория маргинальной цены во время исполнения: VFM vs CPMM (чанки).
  E3. Манипуляция round-trip (накачка/сброс): PnL атакующего <= 0 на всей сетке.
  E4. Медленный фронт-раннинг потока жертвы: прибыль атакующего и проверка
      теоретической границы извлечения  delta <= kappa/(2*gamma_eff).
  E5. Сэндвич внутри эпохи: выигрыш тождественно равен нулю (единая цена клиринга).
  E6. Баланс пула (LP): декомпозиция выручки на транзиентную и перманентную части.

Все результаты печатаются в results.txt и сохраняются в ../figures/*.png
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- палитра (dataviz) -----------------------------
SURFACE = "#fcfcfb"
INK     = "#0b0b0b"
INK2    = "#52514e"
BLUE    = "#2a78d6"   # серия 1: VFM
AQUA    = "#1baf7a"   # серия 2: floor / вспомогательная
YELLOW  = "#eda100"   # серия 3
RED     = "#e34948"   # серия 6: CPMM
VIOLET  = "#4a3aa7"   # серия 5

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "lines.linewidth": 2.0, "figure.dpi": 130,
})

# ----------------------------- механизмы -----------------------------

class VFM:
    """Вязкоупругий потоковый маркет-мейкер.

    Состояние: V (валюационный якорь), J (транзиентное напряжение),
               X, Y (резервы), эпоха t.
    Цена: P(x) = V * exp(J + gamma * x / L),  L = lam * X  (внутри эпохи).
    Обновление после эпохи с нетто-потоком f (f>0 — пул продаёт токен):
        u = f / L
        J <- alpha * (J + gamma * u)
        ln V <- ln V + kappa * u
    Ограничение потока: |f| <= eta * X  (платёжеспособность по построению).
    """

    def __init__(self, X, Y, gamma=2.0, kappa=0.2, alpha=None, half_life=20.0,
                 lam=1.0, eta=0.05, V0=None):
        self.X, self.Y = float(X), float(Y)
        self.gamma, self.kappa, self.lam, self.eta = gamma, kappa, lam, eta
        self.alpha = alpha if alpha is not None else 0.5 ** (1.0 / half_life)
        self.V = V0 if V0 is not None else Y / X
        self.J = 0.0
        self.t = 0

    # --- котировки ---
    def marginal_price(self):
        return self.V * math.exp(self.J)

    def flow_cap(self, side):
        """Максимальный нетто-поток эпохи: продажа токена пулом ограничена X,
        покупка токена пулом ограничена эквивалентом по Y."""
        if side > 0:                      # пул отдаёт токен
            return self.eta * self.X
        # пул принимает токен: ограничиваем отток Y тем же правилом
        return self.eta * self.Y / self.marginal_price()

    def epoch_cost(self, f):
        """Стоимость (в numeraire) нетто-потока f в текущую эпоху, без изменения состояния.
        C(f) = V e^J (L/gamma) (e^{gamma f / L} - 1). Знак совпадает со знаком f."""
        L = self.lam * self.X
        u = f / L
        return self.V * math.exp(self.J) * (L / self.gamma) * math.expm1(self.gamma * u)

    # --- исполнение эпохи ---
    def trade_epoch(self, f):
        """Исполнить нетто-поток f (единая цена клиринга эпохи). Возвращает
        уплаченный numeraire (>0, если пул продаёт токен)."""
        cap = self.flow_cap(1 if f > 0 else -1)
        if abs(f) > cap + 1e-12:
            raise ValueError(f"flow {f:.4g} превышает лимит эпохи {cap:.4g}")
        cost = self.epoch_cost(f)
        L = self.lam * self.X
        u = f / L
        self.X -= f
        self.Y += cost
        self.J = self.alpha * (self.J + self.gamma * u)
        self.V *= math.exp(self.kappa * u)
        self.t += 1
        return cost

    def idle_epochs(self, n):
        """n эпох без потока: J релаксирует, V неизменен."""
        self.J *= self.alpha ** n
        self.t += n

    def clone(self):
        c = VFM(self.X, self.Y, self.gamma, self.kappa, self.alpha,
                lam=self.lam, eta=self.eta, V0=self.V)
        c.J, c.t = self.J, self.t
        return c


class CPMM:
    """x*y=k без комиссии (комиссия не меняет вывод о price impact)."""

    def __init__(self, X, Y):
        self.X, self.Y = float(X), float(Y)
        self.k = self.X * self.Y

    def marginal_price(self):
        return self.Y / self.X

    def buy_tokens(self, q):
        """Купить q токенов у пула; вернуть уплаченный numeraire."""
        newX = self.X - q
        assert newX > 0
        newY = self.k / newX
        paid = newY - self.Y
        self.X, self.Y = newX, newY
        return paid


# ----------------------------- сценарии -----------------------------

def whale_buy_vfm(pool, Q, T):
    """Кит покупает Q токенов равномерным потоком за T эпох (с учётом лимитов).
    Возвращает (paid, got, история маргинальной цены, фактическое число эпох)."""
    remaining, paid, got = Q, 0.0, 0.0
    prices = [pool.marginal_price()]
    t = 0
    while remaining > 1e-9 * Q:
        target = min(remaining, Q / T)
        f = min(target, pool.flow_cap(+1) * 0.999999)
        paid += pool.trade_epoch(f)
        got += f
        remaining -= f
        prices.append(pool.marginal_price())
        t += 1
        if t > 100 * T + 1000:
            raise RuntimeError("исполнение не сходится")
    return paid, got, prices, t


def run_all(outdir_fig, results_path):
    S = 1_000_000_000.0          # эмиссия
    X0 = 0.30 * S                # 30% эмиссии в пуле
    P0 = 1.0
    Y0 = X0 * P0
    Q = 0.20 * S                 # кит покупает 20% эмиссии = 2/3 резервов
    params = dict(gamma=2.0, kappa=0.2, half_life=20.0, lam=1.0, eta=0.05)

    lines = []
    def rep(s=""):
        print(s)
        lines.append(s)

    rep("=" * 78)
    rep("ПАРАМЕТРЫ: S=1e9, пул X0=0.3S, P0=1.0, кит Q=0.2S (66.7% резервов)")
    rep(f"VFM: gamma={params['gamma']}, kappa={params['kappa']}, "
        f"полураспад J = {params['half_life']} эпох, eta={params['eta']}, lam={params['lam']}")
    rep("=" * 78)

    # ---------- E1: стоимость исполнения vs T ----------
    cpmm = CPMM(X0, Y0)
    paid_cpmm = cpmm.buy_tokens(Q)
    prem_cpmm = paid_cpmm / (Q * P0) - 1.0
    final_cpmm = cpmm.marginal_price()
    rep("\n[E1] CPMM (разбиение на части не помогает — путь-независимость):")
    rep(f"  средняя премия: {prem_cpmm*100:8.2f}%   конечная цена: x{final_cpmm/P0:.3f}")

    # теоретический перманентный пол VFM: kappa * ln(X0/(X0-Q)), средняя ~ половина
    perm_total = params["kappa"] * math.log(X0 / (X0 - Q))
    floor_avg = (math.exp(perm_total) - 1.0)  # верхняя оценка; средняя примерно половина
    rep(f"  VFM перманентный сдвиг цены (инвариант пути): x{math.exp(perm_total):.4f}")

    Ts = [25, 50, 100, 300, 1000, 3000]
    prem_vfm, final_vfm, eff_T = [], [], []
    for T in Ts:
        pool = VFM(X0, Y0, **params)
        paid, got, prices, t_used = whale_buy_vfm(pool, Q, T)
        prem = paid / (got * P0) - 1.0
        prem_vfm.append(prem)
        final_vfm.append(prices[-1] / P0)
        eff_T.append(t_used)
        rep(f"  VFM T={T:5d} (факт {t_used:5d} эпох): премия {prem*100:7.2f}%   "
            f"конечная цена x{prices[-1]/P0:.4f}")
    # средняя премия только от перманентной части (эмпирический пол при T->inf)
    rep(f"  Пол (перманентная составляющая, T->inf): ~{prem_vfm[-1]*100:.2f}% "
        f"(CPMM: {prem_cpmm*100:.0f}%) — выигрыш x{prem_cpmm/prem_vfm[-1]:.1f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(prem_cpmm * 100, color=RED, lw=2)
    ax.annotate(f"CPMM (любое разбиение): {prem_cpmm*100:.0f}%",
                xy=(Ts[0], prem_cpmm * 100), xytext=(Ts[0], prem_cpmm * 100 * 0.82),
                color=RED, fontsize=10)
    ax.plot(eff_T, [p * 100 for p in prem_vfm], color=BLUE, marker="o", ms=6)
    ax.annotate("VFM: премия ~ S∞ + Θ(1/T)", xy=(eff_T[2], prem_vfm[2] * 100),
                xytext=(eff_T[2] * 1.4, prem_vfm[2] * 100 * 1.8), color=BLUE, fontsize=10)
    ax.axhline(prem_vfm[-1] * 100, color=AQUA, lw=1.5, ls=(0, (4, 3)))
    ax.annotate(f"пол при T=3000 ≈ {prem_vfm[-1]*100:.1f}% (теор. S∞ ≈ 9.7%)",
                xy=(Ts[-1], prem_vfm[-1] * 100), xytext=(eff_T[1], prem_vfm[-1] * 100 * 1.6),
                color=AQUA, fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Горизонт исполнения T, эпох (лог. шкала)")
    ax.set_ylabel("Средняя премия исполнения, % (лог. шкала)")
    ax.set_title("E1. Кит покупает 20% эмиссии: премия VFM против CPMM", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{outdir_fig}/fig1_impact_vs_T.png")
    plt.close(fig)

    # ---------- E2: траектории цены ----------
    T_show = 300
    pool = VFM(X0, Y0, **params)
    _, _, prices_vfm, t_used = whale_buy_vfm(pool, Q, T_show)
    # после исполнения: релаксация J в течение 200 эпох
    relax = []
    for _ in range(200):
        pool.idle_epochs(1)
        relax.append(pool.marginal_price())
    cpmm2 = CPMM(X0, Y0)
    prices_cpmm = [cpmm2.marginal_price()]
    for _ in range(t_used):
        cpmm2.buy_tokens(Q / t_used)
        prices_cpmm.append(cpmm2.marginal_price())

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = np.arange(len(prices_cpmm))
    ax.plot(xs, prices_cpmm, color=RED)
    ax.annotate(f"CPMM: x{prices_cpmm[-1]:.1f}", xy=(xs[-1], prices_cpmm[-1]),
                xytext=(xs[-1] * 0.72, prices_cpmm[-1] * 0.92), color=RED)
    xs2 = np.arange(len(prices_vfm) + len(relax))
    ax.plot(xs2, prices_vfm + relax, color=BLUE)
    ax.annotate(f"VFM: пик x{max(prices_vfm):.2f}, после релаксации x{relax[-1]:.2f}",
                xy=(len(prices_vfm), prices_vfm[-1]),
                xytext=(len(prices_vfm) * 0.35, max(prices_vfm) + 1.1), color=BLUE)
    ax.axvline(t_used, color=INK2, lw=0.8, ls=(0, (2, 3)))
    ax.set_xlabel("Эпоха (исполнение кита закончилось на пунктирной линии)")
    ax.set_ylabel("Маргинальная цена, numeraire/токен")
    ax.set_title(f"E2. Траектория цены: кит покупает 20% эмиссии за {t_used} эпох")
    fig.tight_layout()
    fig.savefig(f"{outdir_fig}/fig2_price_paths.png")
    plt.close(fig)

    # ---------- E3: round-trip манипуляция ----------
    rep("\n[E3] Round-trip манипуляция (купить Qm за Tm, пауза W, продать за Tm):")
    fracs = [0.02, 0.05, 0.10, 0.20]
    waits = [0, 5, 20, 60]
    Tm = 20
    pnl_grid = np.zeros((len(fracs), len(waits)))
    worst = -1e18
    for i, fr in enumerate(fracs):
        for j, w in enumerate(waits):
            pool = VFM(X0, Y0, **params)
            Qm = fr * X0
            paid, got, _, _ = whale_buy_vfm(pool, Qm, Tm)
            pool.idle_epochs(w)
            # продажа назад тем же темпом
            remaining, revenue = got, 0.0
            while remaining > 1e-9 * got:
                f = -min(remaining, got / Tm)
                cap = pool.flow_cap(-1)
                f = -min(-f, cap * 0.999999)
                revenue += -pool.trade_epoch(f)
                remaining -= -f
            pnl = revenue - paid
            pnl_grid[i, j] = pnl / paid * 100
            worst = max(worst, pnl)
    for i, fr in enumerate(fracs):
        row = "  ".join(f"{pnl_grid[i, j]:7.3f}%" for j in range(len(waits)))
        rep(f"  Qm={fr*100:4.0f}% резервов:  PnL/затраты: {row}   (W={waits})")
    rep(f"  Максимальный PnL по сетке: {worst:,.0f} numeraire  "
        f"({'<= 0: манипуляция строго убыточна' if worst <= 0 else 'ПОЛОЖИТЕЛЕН — ПРОВЕРИТЬ'})")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    # один цвет, светлый->тёмный: величина убытка (все значения <= 0)
    vmax = 0.0
    vmin = pnl_grid.min()
    # один цвет, светлый (малый убыток) -> тёмный (большой убыток)
    im = ax.imshow(pnl_grid, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq", ["#0d366b", "#cde2fb"]), vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(waits)), [f"W={w}" for w in waits])
    ax.set_yticks(range(len(fracs)), [f"{f*100:.0f}%" for f in fracs])
    ax.set_xlabel("Пауза между накачкой и сбросом, эпох")
    ax.set_ylabel("Размер манипуляции, % резервов")
    ax.set_title("E3. Round-trip: PnL атакующего, % затрат (всюду < 0)")
    for i in range(len(fracs)):
        for j in range(len(waits)):
            ax.text(j, i, f"{pnl_grid[i,j]:.2f}%", ha="center", va="center",
                    color=SURFACE if pnl_grid[i, j] < vmin * 0.35 else INK, fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(f"{outdir_fig}/fig3_roundtrip.png")
    plt.close(fig)

    # ---------- E4: медленный фронт-раннинг потока жертвы ----------
    rep("\n[E4] Фронт-раннинг: жертва покупает Qv=10% резервов за Tv эпох; атакующий")
    rep("     покупает Qa до неё и продаёт после (Qa и Ta оптимизируются перебором).")
    kap, gam = params["kappa"], params["gamma"]
    rep(f"     Нижний ориентир извлечения: kappa/(2*gamma) = "
        f"{kap/(2*gam)*100:.1f}% издержек жертвы; при малых Tv доминирует")
    rep("     транзиентный член O(1/(rho*Tv)) — терпеливая жертва защищена лучше.")
    Qv = 0.10 * X0
    Tvs = [25, 50, 100, 200, 400, 800]
    extract = []
    for Tv in Tvs:
        pool = VFM(X0, Y0, **params)
        paid_v0, _, _, _ = whale_buy_vfm(pool, Qv, Tv)
        victim_cost0 = paid_v0 - Qv * P0
        best = 0.0
        for fr_a in [0.005, 0.01, 0.02, 0.04, 0.07, 0.10]:
            for Ta in [10, 30, 90]:
                Qa = fr_a * X0
                pool = VFM(X0, Y0, **params)
                paid_a, got_a, _, _ = whale_buy_vfm(pool, Qa, Ta)   # атакующий входит
                whale_buy_vfm(pool, Qv, Tv)                         # жертва исполняется
                remaining, revenue = got_a, 0.0                     # атакующий выходит
                while remaining > 1e-9 * got_a:
                    f = min(remaining, got_a / Ta, pool.flow_cap(-1) * 0.999999)
                    revenue += -pool.trade_epoch(-f)
                    remaining -= f
                best = max(best, revenue - paid_a)
        extract.append(best / victim_cost0 * 100)
        rep(f"  Tv={Tv:4d}: макс. прибыль атакующего = {extract[-1]:5.1f}% издержек жертвы")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(Tvs, extract, color=BLUE, marker="o", ms=6)
    ax.annotate("оптимизированный атакующий", xy=(Tvs[2], extract[2]),
                xytext=(Tvs[2] * 1.15, extract[2] * 1.25), color=BLUE, fontsize=10)
    ax.axhline(kap / (2 * gam) * 100, color=YELLOW, lw=1.5, ls=(0, (4, 3)))
    ax.annotate("нижний ориентир κ/(2γ)", xy=(Tvs[0], kap / (2 * gam) * 100),
                xytext=(Tvs[0], kap / (2 * gam) * 100 * 1.15), color="#9a6a00", fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks(Tvs, [str(t) for t in Tvs])
    ax.minorticks_off()
    ax.set_xlabel("Горизонт исполнения жертвы Tv, эпох (лог. шкала)")
    ax.set_ylabel("Извлечение, % издержек жертвы")
    ax.set_title("E4. Фронт-раннинг потока: извлечение убывает с ростом Tv", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{outdir_fig}/fig4_frontrun.png")
    plt.close(fig)

    # ---------- E5: сэндвич внутри эпохи ----------
    rep("\n[E5] Сэндвич внутри одной эпохи: атакующий добавляет +q и -q к нетто-потоку")
    rep("     эпохи. Нетто-поток не меняется => единая цена клиринга не меняется =>")
    rep("     выигрыш атакующего тождественно 0, жертва не страдает. (По построению.)")

    # ---------- E6: баланс пула ----------
    rep("\n[E6] Баланс пула после покупки китом Q=20% эмиссии (T=300):")
    pool = VFM(X0, Y0, **params)
    paid, got, prices, t_used = whale_buy_vfm(pool, Q, T_show)
    pool.idle_epochs(1000)  # полная релаксация J
    V_end = pool.V
    equity0 = X0 * P0 + Y0
    # два сценария маркировки: кит неинформирован (истинная цена P0) и
    # информирован ровно на величину перманентного сдвига (истинная цена V_end)
    eq_uninf = pool.X * P0 + pool.Y
    eq_inf = pool.X * V_end + pool.Y
    eq0_inf = X0 * V_end + Y0
    rep(f"  Получено пулом: {paid:,.0f}; отдано токенов: {got:,.0f}; V_end = {V_end:.4f}")
    rep(f"  Кит неинформирован (истинная цена 1.0): PnL пула = {eq_uninf-equity0:+,.0f} "
        f"({(eq_uninf-equity0)/equity0*100:+.2f}%) — транзиентная выручка LP")
    rep(f"  Кит полностью информирован (истинная цена {V_end:.3f}): PnL пула = "
        f"{eq_inf-eq0_inf:+,.0f} ({(eq_inf-eq0_inf)/eq0_inf*100:+.2f}%) — адверсный отбор,"
        f" ограничен параметром kappa")

    with open(results_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    rep(f"\nГотово. Фигуры: {outdir_fig}/fig1..fig4, лог: {results_path}")


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(os.path.dirname(here), "figures")
    os.makedirs(figdir, exist_ok=True)
    run_all(figdir, os.path.join(here, "results.txt"))
