// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title  VFM — Viscoelastic Flow Market (эталонный ЭСКИЗ ядра)
/// @notice Иллюстрация O(1)-реализуемости модели из PAPER.md.
///         НЕ АУДИРОВАНО, НЕ ДЛЯ ПРОДАКШЕНА: перенос токенов, доли LP,
///         округления и библиотека фиксированной точки намеренно опущены
///         или упрощены. Все величины — фиксированная точка 1e18 (WAD).
///
/// Состояние пула — 7 постоянных слотов (аксиома A6):
///   V (валюационный якорь), J (вязкоупругое напряжение), X, Y (резервы),
///   W (накопитель средней цены для ленивых потоков), lastEpoch, params.
///
/// Формулы (нумерация как в PAPER.md §7):
///   (1) C(f) = V e^J (L/γ)(e^{γf/L} − 1),  L = λX
///   (2) J ← α (J + γ f/L)
///   (3) V ← V · e^{κ f/L}
///   (4) |f| ≤ η X  (симметрично по Y)
contract VFM {
    int256 internal constant WAD = 1e18;

    // --- параметры (immutable после деплоя; базовая конфигурация PAPER §12) ---
    int256 public immutable gamma;   // вязкость, WAD           (2e18)
    int256 public immutable kappa;   // упругость, WAD          (0.2e18)
    int256 public immutable alpha;   // релаксация за эпоху, WAD (0.5^(1/20) ≈ 0.9659e18)
    int256 public immutable lam;     // глубина L = lam*X, WAD  (1e18)
    int256 public immutable eta;     // лимит потока эпохи, WAD (0.05e18)
    uint256 public immutable epochBlocks; // блоков в эпохе

    // --- состояние (7 слотов) ---
    int256 public V;          // якорь, WAD
    int256 public J;          // напряжение, WAD (лог-величина)
    int256 public X;          // резерв токена, WAD
    int256 public Y;          // резерв нумерария, WAD
    int256 public W;          // накопитель средней цены, WAD
    uint256 public lastEpoch; // последняя обработанная эпоха
    int256 public pendingF;   // агрегируемый нетто-поток текущей эпохи, WAD

    struct Stream {           // 2 слота на заявку-поток
        int192 ratePerEpoch;  // подписанный темп (токенов/эпоху), WAD
        uint32 startEpoch;
        uint32 endEpoch;
        int256 wStart;        // снимок W при открытии (ленивый расчёт)
    }
    mapping(uint256 => Stream) public streams;
    uint256 public nextStreamId;

    constructor(int256 X0, int256 Y0, int256 g, int256 k, int256 a, int256 l, int256 e, uint256 eb) {
        X = X0; Y = Y0; V = (Y0 * WAD) / X0;
        gamma = g; kappa = k; alpha = a; lam = l; eta = e; epochBlocks = eb;
        lastEpoch = block.number / eb;
    }

    // ------------------------------------------------------------------
    //  Ядро: закрытие эпох (O(1) на эпоху; догон пустых эпох — O(log n))
    // ------------------------------------------------------------------
    function _settleEpochs() internal {
        uint256 cur = block.number / epochBlocks;
        if (cur == lastEpoch) return;

        // 1. Исполнить накопленный нетто-поток прошлой эпохи по единой цене.
        int256 f = _clipFlow(pendingF);
        if (f != 0) {
            int256 L = (lam * X) / WAD;
            int256 u = (f * WAD) / L;
            //     C(f) = V e^J (L/γ)(e^{γu} − 1)          — формула (1)
            int256 cost = mulWad(mulWad(V, expWad(J)),
                                 (L * (expWad((gamma * u) / WAD) - WAD)) / gamma);
            X -= f;
            Y += cost;
            W += (cost * WAD) / f;                       // ленивый накопитель
            J = (alpha * (J + (gamma * u) / WAD)) / WAD; // формула (2)
            V = mulWad(V, expWad((kappa * u) / WAD));    // формула (3)
            pendingF = 0;
        }

        // 2. Догон пустых эпох: J ← J · α^n  (возведение квадрированием).
        uint256 idle = cur - lastEpoch - 1;
        if (idle > 0) J = mulWad(J, powWad(alpha, idle));
        lastEpoch = cur;
    }

    /// формула (4): |f| ≤ η·X, симметрично η·Y по нумерарию
    function _clipFlow(int256 f) internal view returns (int256) {
        int256 capX = (eta * X) / WAD;
        int256 capY = (((eta * Y) / WAD) * WAD) / marginalPrice();
        int256 cap = f > 0 ? capX : capY;
        if (f > cap) return cap;
        if (f < -cap) return -cap;
        return f;
    }

    // ------------------------------------------------------------------
    //  Публичный интерфейс (эскиз; трансферы токенов опущены)
    // ------------------------------------------------------------------
    function marginalPrice() public view returns (int256) {
        return mulWad(V, expWad(J)); // P = V e^J
    }

    /// Открыть заявку-поток: обменивать |rate| токенов/эпоху в течение T эпох.
    function openStream(int192 rate, uint32 T) external returns (uint256 id) {
        _settleEpochs();
        id = nextStreamId++;
        uint32 s = uint32(lastEpoch + 1);
        streams[id] = Stream(rate, s, s + T, W);
        // Примечание эскиза: pendingF пополняется темпом потока при закрытии
        // каждой эпохи; в полной реализации — через агрегатор активных темпов
        // (две переменные: суммарный темп и очередь истечений), также O(1).
    }

    /// Атомарный своп = вырожденный поток T=1: попадает в нетто-поток эпохи.
    function swap(int256 f) external {
        _settleEpochs();
        pendingF += f; // клиринг по единой цене эпохи — защита от сэндвича (T5)
    }

    /// Расчёт закрытого потока: владелец должен/получает rate·(W_end − W_start).
    function settleStream(uint256 id) external view returns (int256 owedNumeraire) {
        Stream memory s = streams[id];
        owedNumeraire = (int256(s.ratePerEpoch) * (W - s.wStart)) / WAD;
    }

    // ------------------------------------------------------------------
    //  Фиксированная точка (эскиз; в продакшене — проверенная библиотека)
    // ------------------------------------------------------------------
    function mulWad(int256 a, int256 b) internal pure returns (int256) {
        return (a * b) / WAD;
    }

    /// e^x, WAD. Аргумент по построению мал: |x| ≤ γη + |J|, J ограничено
    /// геометрической суммой γη·α/(1−α) — ряд Тейлора 8 членов достаточен.
    function expWad(int256 x) internal pure returns (int256 r) {
        r = WAD;
        int256 term = WAD;
        for (int256 i = 1; i <= 8; ++i) {
            term = (term * x) / (WAD * i);
            r += term;
        }
    }

    /// a^n, WAD — возведение квадрированием, O(log n).
    function powWad(int256 a, uint256 n) internal pure returns (int256 r) {
        r = WAD;
        while (n > 0) {
            if (n & 1 == 1) r = mulWad(r, a);
            a = mulWad(a, a);
            n >>= 1;
        }
    }
}
