import React, {
  useState,
  useEffect,
  useCallback,
  useRef,
  useMemo,
} from "react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

// =============================================================================
// SEEDED RANDOM GENERATOR (Mulberry32)
// =============================================================================

const createRNG = (seed) => {
  let s = seed;
  return () => {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

// =============================================================================
// CONSTANTS & GAME THEORY
// =============================================================================

const FIXED_PARAMS = {
  R: 30,
  B: 10,
  volunteerSpeed: 1.2,
  targetSpeed: 0.4,
  responseTime: 50,
  aoiThreshold: 25,
  maxTime: 400,
};

const PRESETS = {
  low: { name: "Low Cost", c_ratio: 0.25, N: 100, L: 400 },
  moderate: { name: "Moderate", c_ratio: 0.5, N: 100, L: 500 },
  high: { name: "High Cost", c_ratio: 0.7, N: 100, L: 600 },
  critical: { name: "Critical", c_ratio: 0.88, N: 100, L: 700 },
};

const computeRho = (R, L) => (Math.PI * R * R) / (L * L);

const computeNashK = (N, rho, B, c) => {
  const Brho = B * rho;
  if (c >= Brho) return 0;
  if (c <= 0) return N;
  const k = 1 + Math.log(c / Brho) / Math.log(1 - rho);
  return Math.max(0, Math.min(N, Math.floor(k)));
};

const computeOptimalK = (N, rho, B, c) => {
  const NBrho = N * B * rho;
  if (c >= NBrho) return 0;
  if (c <= 0) return N;
  const k = 1 + Math.log(c / NBrho) / Math.log(1 - rho);
  return Math.max(0, Math.min(N, Math.floor(k)));
};

const computePdet = (k, rho) => {
  if (k <= 0) return 0;
  return 1 - Math.pow(1 - rho, k);
};

const distance = (x1, y1, x2, y2) => Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);

// =============================================================================
// MONTE CARLO COVERAGE ESTIMATION
// =============================================================================

const estimateCoverage = (volunteers, L, R, samples = 800) => {
  if (!volunteers || volunteers.length === 0) return 0;
  const activeVols = volunteers.filter((v) => v.active);
  if (activeVols.length === 0) return 0;

  let covered = 0;
  for (let i = 0; i < samples; i++) {
    const px = Math.random() * L;
    const py = Math.random() * L;
    for (const v of activeVols) {
      if (distance(px, py, v.x, v.y) <= R) {
        covered++;
        break;
      }
    }
  }
  return covered / samples;
};

// =============================================================================
// SIMULATION ENGINE
// =============================================================================

class SimulationEngine {
  constructor(config, rng) {
    this.config = config;
    this.rng = rng;
    this.initialize();
  }

  initialize() {
    const { N, L, k } = this.config;
    const R = FIXED_PARAMS.R;
    const rng = this.rng;

    this.volunteers = [];
    for (let i = 0; i < N; i++) {
      this.volunteers.push({
        id: i,
        x: rng() * L,
        y: rng() * L,
        waypointX: rng() * L,
        waypointY: rng() * L,
        active: false,
      });
    }

    const indices = [...Array(N).keys()];
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    for (let i = 0; i < k; i++) {
      this.volunteers[indices[i]].active = true;
    }

    const margin = R;
    this.target = {
      x: margin + rng() * (L - 2 * margin),
      y: margin + rng() * (L - 2 * margin),
      vx: (rng() - 0.5) * FIXED_PARAMS.targetSpeed * 2,
      vy: (rng() - 0.5) * FIXED_PARAMS.targetSpeed * 2,
    };

    this.time = 0;
    this.aoi = 0;
    this.aoiHistory = [];
    this.detections = 0;
    this.rescueStarted = false;
    this.rescueStartTime = null;
    this.rescueProgress = 0;
    this.outcome = "searching";
  }

  step() {
    if (this.outcome !== "searching") return this.getState();

    const { L } = this.config;
    const {
      R,
      volunteerSpeed,
      targetSpeed,
      responseTime,
      aoiThreshold,
      maxTime,
    } = FIXED_PARAMS;

    for (const v of this.volunteers) {
      if (!v.active) continue;
      const dx = v.waypointX - v.x;
      const dy = v.waypointY - v.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < 8) {
        v.waypointX = Math.random() * L;
        v.waypointY = Math.random() * L;
      } else {
        v.x += (dx / dist) * Math.min(volunteerSpeed, dist);
        v.y += (dy / dist) * Math.min(volunteerSpeed, dist);
      }
    }

    if (Math.random() < 0.03) {
      const angle = Math.random() * Math.PI * 2;
      this.target.vx = Math.cos(angle) * targetSpeed;
      this.target.vy = Math.sin(angle) * targetSpeed;
    }

    const margin = R;
    let newX = this.target.x + this.target.vx;
    let newY = this.target.y + this.target.vy;

    if (newX < margin || newX > L - margin) this.target.vx *= -1;
    if (newY < margin || newY > L - margin) this.target.vy *= -1;

    this.target.x = Math.max(margin, Math.min(L - margin, newX));
    this.target.y = Math.max(margin, Math.min(L - margin, newY));

    let detected = false;
    for (const v of this.volunteers) {
      if (!v.active) continue;
      if (distance(v.x, v.y, this.target.x, this.target.y) <= R) {
        detected = true;
        break;
      }
    }

    if (detected) {
      this.aoi = 0;
      this.detections++;
      if (!this.rescueStarted) {
        this.rescueStarted = true;
        this.rescueStartTime = this.time;
      }
    } else {
      this.aoi++;
    }

    this.aoiHistory.push(this.aoi);

    if (this.rescueStarted) {
      if (this.aoi > aoiThreshold) {
        this.rescueStarted = false;
        this.rescueStartTime = null;
        this.rescueProgress = 0;
      } else {
        this.rescueProgress = Math.min(
          1,
          (this.time - this.rescueStartTime) / responseTime
        );
        if (this.time - this.rescueStartTime >= responseTime) {
          this.outcome = "rescued";
        }
      }
    }

    this.time++;
    if (this.time >= maxTime && this.outcome === "searching") {
      this.outcome = "timeout";
    }

    return this.getState();
  }

  runToCompletion() {
    while (this.outcome === "searching") {
      this.step();
    }
    return this.getState();
  }

  getState() {
    return {
      volunteers: this.volunteers,
      target: this.target,
      time: this.time,
      aoi: this.aoi,
      aoiHistory: [...this.aoiHistory],
      detections: this.detections,
      rescueStarted: this.rescueStarted,
      rescueProgress: this.rescueProgress,
      outcome: this.outcome,
    };
  }
}

// =============================================================================
// SHARED SIMULATION FACTORY
// =============================================================================

const createSimulationPair = (N, L, kNash, kOpt, seed) => {
  const rng = createRNG(seed);

  const baseVolunteers = [];
  for (let i = 0; i < N; i++) {
    baseVolunteers.push({
      x: rng() * L,
      y: rng() * L,
      waypointX: rng() * L,
      waypointY: rng() * L,
    });
  }

  const indices = [...Array(N).keys()];
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const margin = FIXED_PARAMS.R;
  const target = {
    x: margin + rng() * (L - 2 * margin),
    y: margin + rng() * (L - 2 * margin),
    vx: (rng() - 0.5) * FIXED_PARAMS.targetSpeed * 2,
    vy: (rng() - 0.5) * FIXED_PARAMS.targetSpeed * 2,
  };

  const nashVolunteers = baseVolunteers.map((v, i) => ({
    ...v,
    id: i,
    active: false,
  }));
  for (let i = 0; i < kNash; i++) {
    nashVolunteers[indices[i]].active = true;
  }

  const optVolunteers = baseVolunteers.map((v, i) => ({
    ...v,
    id: i,
    active: false,
  }));
  for (let i = 0; i < kOpt; i++) {
    optVolunteers[indices[i]].active = true;
  }

  return {
    nash: { volunteers: nashVolunteers, target: { ...target } },
    opt: { volunteers: optVolunteers, target: { ...target } },
  };
};

// =============================================================================
// STYLES
// =============================================================================

const colors = {
  bg: "#0c0c0c",
  surface: "#161616",
  border: "#262626",
  borderDark: "#303030",
  textPrimary: "#e5e5e5",
  textSecondary: "#a3a3a3",
  textMuted: "#6b6b6b",
  nash: "#d4705a",
  nashBg: "#1a1412",
  nashLight: "rgba(212, 112, 90, 0.1)",
  optimal: "#5a9a7a",
  optimalBg: "#121a16",
  optimalLight: "rgba(90, 154, 122, 0.1)",
  accent: "#6b8ccc",
  warning: "#c9a55a",
};

const fonts = {
  mono: '"IBM Plex Mono", "SF Mono", "Consolas", monospace',
  sans: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
};

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function EmergencySimulation() {
  const [preset, setPreset] = useState("moderate");
  const [costRatio, setCostRatio] = useState(0.5);
  const [numVolunteers, setNumVolunteers] = useState(100);
  const [areaSize, setAreaSize] = useState(500);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [currentSeed, setCurrentSeed] = useState(() =>
    Math.floor(Math.random() * 1000000)
  );

  const [batchSize, setBatchSize] = useState(50);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchProgress, setBatchProgress] = useState(0);
  const [batchResults, setBatchResults] = useState(null);

  const [simNash, setSimNash] = useState(null);
  const [simOpt, setSimOpt] = useState(null);
  const [stateNash, setStateNash] = useState(null);
  const [stateOpt, setStateOpt] = useState(null);
  const [running, setRunning] = useState(false);

  const [coverageNash, setCoverageNash] = useState(0);
  const [coverageOpt, setCoverageOpt] = useState(0);

  const canvasNashRef = useRef(null);
  const canvasOptRef = useRef(null);
  const intervalRef = useRef(null);

  const rho = useMemo(() => computeRho(FIXED_PARAMS.R, areaSize), [areaSize]);
  const cost = useMemo(
    () => costRatio * FIXED_PARAMS.B * rho,
    [costRatio, rho]
  );
  const kNash = useMemo(
    () => computeNashK(numVolunteers, rho, FIXED_PARAMS.B, cost),
    [numVolunteers, rho, cost]
  );
  const kOpt = useMemo(
    () => computeOptimalK(numVolunteers, rho, FIXED_PARAMS.B, cost),
    [numVolunteers, rho, cost]
  );

  const pDetCurveData = useMemo(() => {
    const data = [];
    for (
      let k = 0;
      k <= numVolunteers;
      k += Math.max(1, Math.floor(numVolunteers / 40))
    ) {
      data.push({ k, pDet: computePdet(k, rho) * 100 });
    }
    if (data[data.length - 1].k !== numVolunteers) {
      data.push({
        k: numVolunteers,
        pDet: computePdet(numVolunteers, rho) * 100,
      });
    }
    return data;
  }, [numVolunteers, rho]);

  const aoiChartData = useMemo(() => {
    if (!stateNash || !stateOpt) return [];
    const maxLen = Math.max(
      stateNash.aoiHistory.length,
      stateOpt.aoiHistory.length
    );
    const data = [];
    const startIdx = Math.max(0, maxLen - 80);
    for (let i = startIdx; i < maxLen; i++) {
      data.push({
        t: i,
        nash: stateNash.aoiHistory[i] ?? null,
        opt: stateOpt.aoiHistory[i] ?? null,
      });
    }
    return data;
  }, [stateNash, stateOpt]);

  const applyPreset = useCallback((key) => {
    const p = PRESETS[key];
    setPreset(key);
    setCostRatio(p.c_ratio);
    setNumVolunteers(p.N);
    setAreaSize(p.L);
  }, []);

  const initSimulations = useCallback(() => {
    setRunning(false);
    setBatchResults(null);

    const seed = currentSeed;
    const { nash, opt } = createSimulationPair(
      numVolunteers,
      areaSize,
      kNash,
      kOpt,
      seed
    );

    const engineNash = new SimulationEngine(
      { N: numVolunteers, L: areaSize, k: kNash },
      createRNG(seed)
    );
    engineNash.volunteers = nash.volunteers.map((v) => ({ ...v }));
    engineNash.target = { ...nash.target };
    engineNash.time = 0;
    engineNash.aoi = 0;
    engineNash.aoiHistory = [];
    engineNash.detections = 0;
    engineNash.rescueStarted = false;
    engineNash.rescueStartTime = null;
    engineNash.rescueProgress = 0;
    engineNash.outcome = "searching";

    const engineOpt = new SimulationEngine(
      { N: numVolunteers, L: areaSize, k: kOpt },
      createRNG(seed)
    );
    engineOpt.volunteers = opt.volunteers.map((v) => ({ ...v }));
    engineOpt.target = { ...opt.target };
    engineOpt.time = 0;
    engineOpt.aoi = 0;
    engineOpt.aoiHistory = [];
    engineOpt.detections = 0;
    engineOpt.rescueStarted = false;
    engineOpt.rescueStartTime = null;
    engineOpt.rescueProgress = 0;
    engineOpt.outcome = "searching";

    setSimNash(engineNash);
    setSimOpt(engineOpt);
    setStateNash(engineNash.getState());
    setStateOpt(engineOpt.getState());

    setCoverageNash(
      estimateCoverage(nash.volunteers, areaSize, FIXED_PARAMS.R)
    );
    setCoverageOpt(estimateCoverage(opt.volunteers, areaSize, FIXED_PARAMS.R));
  }, [numVolunteers, areaSize, kNash, kOpt, currentSeed]);

  const handleReset = useCallback(() => {
    setCurrentSeed(Math.floor(Math.random() * 1000000));
  }, []);

  useEffect(() => {
    initSimulations();
  }, [currentSeed, initSimulations]);

  useEffect(() => {
    initSimulations();
  }, [numVolunteers, areaSize, kNash, kOpt]);

  const stepSimulation = useCallback(() => {
    if (!simNash || !simOpt) return;

    const sNash = simNash.step();
    const sOpt = simOpt.step();

    setStateNash({ ...sNash });
    setStateOpt({ ...sOpt });

    if (sNash.time % 20 === 0) {
      setCoverageNash(
        estimateCoverage(sNash.volunteers, areaSize, FIXED_PARAMS.R)
      );
      setCoverageOpt(
        estimateCoverage(sOpt.volunteers, areaSize, FIXED_PARAMS.R)
      );
    }

    if (sNash.outcome !== "searching" && sOpt.outcome !== "searching") {
      setRunning(false);
    }
  }, [simNash, simOpt, areaSize]);

  useEffect(() => {
    if (running) {
      const interval = Math.max(16, 50 / playbackSpeed);
      intervalRef.current = setInterval(stepSimulation, interval);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [running, playbackSpeed, stepSimulation]);

  const runBatch = useCallback(async () => {
    setBatchRunning(true);
    setBatchProgress(0);
    setBatchResults(null);

    const results = {
      nash: { successes: 0, totalTime: 0, totalDetections: 0 },
      opt: { successes: 0, totalTime: 0, totalDetections: 0 },
    };

    const runChunk = (startIdx, endIdx) => {
      return new Promise((resolve) => {
        setTimeout(() => {
          for (let i = startIdx; i < endIdx; i++) {
            const seed = currentSeed + i * 12345;
            const rng = createRNG(seed);

            const engNash = new SimulationEngine(
              { N: numVolunteers, L: areaSize, k: kNash },
              rng
            );
            const resNash = engNash.runToCompletion();
            if (resNash.outcome === "rescued") results.nash.successes++;
            results.nash.totalTime += resNash.time;
            results.nash.totalDetections += resNash.detections;

            const rng2 = createRNG(seed);
            const engOpt = new SimulationEngine(
              { N: numVolunteers, L: areaSize, k: kOpt },
              rng2
            );
            const resOpt = engOpt.runToCompletion();
            if (resOpt.outcome === "rescued") results.opt.successes++;
            results.opt.totalTime += resOpt.time;
            results.opt.totalDetections += resOpt.detections;
          }
          resolve();
        }, 0);
      });
    };

    const chunkSize = 10;
    for (let i = 0; i < batchSize; i += chunkSize) {
      await runChunk(i, Math.min(i + chunkSize, batchSize));
      setBatchProgress(Math.min(i + chunkSize, batchSize));
    }

    setBatchResults({
      runs: batchSize,
      nash: {
        successRate: (results.nash.successes / batchSize) * 100,
        avgTime: results.nash.totalTime / batchSize,
        avgDetections: results.nash.totalDetections / batchSize,
      },
      opt: {
        successRate: (results.opt.successes / batchSize) * 100,
        avgTime: results.opt.totalTime / batchSize,
        avgDetections: results.opt.totalDetections / batchSize,
      },
    });

    setBatchRunning(false);
  }, [batchSize, numVolunteers, areaSize, kNash, kOpt, currentSeed]);

  const renderCanvas = useCallback(
    (canvas, state, isNash) => {
      if (!canvas || !state) return;
      const ctx = canvas.getContext("2d");
      const W = canvas.width;
      const H = canvas.height;
      const scale = W / areaSize;
      const R = FIXED_PARAMS.R * scale;

      // Background
      ctx.fillStyle = "#0c0c0c";
      ctx.fillRect(0, 0, W, H);

      // Grid
      ctx.strokeStyle = "#1f1f1f";
      ctx.lineWidth = 0.5;
      const gridStep = 50;
      for (let x = 0; x < areaSize; x += gridStep) {
        ctx.beginPath();
        ctx.moveTo(x * scale, 0);
        ctx.lineTo(x * scale, H);
        ctx.stroke();
      }
      for (let y = 0; y < areaSize; y += gridStep) {
        ctx.beginPath();
        ctx.moveTo(0, y * scale);
        ctx.lineTo(W, y * scale);
        ctx.stroke();
      }

      // Border
      ctx.strokeStyle = "#262626";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, W - 1, H - 1);

      const primaryColor = isNash ? colors.nash : colors.optimal;
      const primaryLight = isNash
        ? "rgba(212, 112, 90, 0.08)"
        : "rgba(90, 154, 122, 0.08)";

      // Detection zones
      ctx.fillStyle = primaryLight;
      state.volunteers
        .filter((v) => v.active)
        .forEach((v) => {
          ctx.beginPath();
          ctx.arc(v.x * scale, v.y * scale, R, 0, Math.PI * 2);
          ctx.fill();
        });

      // Detection zone borders
      ctx.strokeStyle = isNash
        ? "rgba(212, 112, 90, 0.2)"
        : "rgba(90, 154, 122, 0.2)";
      ctx.lineWidth = 0.5;
      state.volunteers
        .filter((v) => v.active)
        .forEach((v) => {
          ctx.beginPath();
          ctx.arc(v.x * scale, v.y * scale, R, 0, Math.PI * 2);
          ctx.stroke();
        });

      // Inactive volunteers
      ctx.fillStyle = "#404040";
      state.volunteers
        .filter((v) => !v.active)
        .forEach((v) => {
          ctx.beginPath();
          ctx.arc(v.x * scale, v.y * scale, 2, 0, Math.PI * 2);
          ctx.fill();
        });

      // Active volunteers
      ctx.fillStyle = primaryColor;
      state.volunteers
        .filter((v) => v.active)
        .forEach((v) => {
          ctx.beginPath();
          ctx.arc(v.x * scale, v.y * scale, 3, 0, Math.PI * 2);
          ctx.fill();
        });

      // Target
      const tx = state.target.x * scale;
      const ty = state.target.y * scale;

      if (state.rescueStarted) {
        ctx.beginPath();
        ctx.arc(tx, ty, 12, 0, Math.PI * 2);
        ctx.strokeStyle = colors.warning;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(tx, ty, 6, 0, Math.PI * 2);
      ctx.fillStyle = colors.warning;
      ctx.fill();
      ctx.strokeStyle = "#0c0c0c";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    },
    [areaSize]
  );

  useEffect(() => {
    renderCanvas(canvasNashRef.current, stateNash, true);
    renderCanvas(canvasOptRef.current, stateOpt, false);
  }, [stateNash, stateOpt, renderCanvas]);

  const bothFinished =
    stateNash?.outcome !== "searching" && stateOpt?.outcome !== "searching";

  return (
    <div
      style={{
        minHeight: "100vh",
        background: colors.bg,
        color: colors.textPrimary,
        fontFamily: fonts.sans,
        fontSize: "13px",
        lineHeight: 1.5,
      }}
    >
      {/* Header */}
      <header
        style={{
          padding: "14px 20px",
          borderBottom: `1px solid ${colors.border}`,
          background: colors.surface,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <h1
            style={{
              fontSize: "15px",
              fontWeight: 600,
              margin: 0,
              color: colors.textPrimary,
            }}
          >
            Emergency Crowd-Finding Simulation
          </h1>
          <p
            style={{
              margin: "2px 0 0",
              color: colors.textSecondary,
              fontSize: "11px",
            }}
          >
            Nash Equilibrium vs Social Optimum — Age of Information Analysis
          </p>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <label style={{ color: colors.textSecondary, fontSize: "11px" }}>
            Batch:
          </label>
          <select
            value={batchSize}
            onChange={(e) => setBatchSize(parseInt(e.target.value))}
            disabled={batchRunning}
            style={{
              background: colors.surface,
              border: `1px solid ${colors.border}`,
              borderRadius: "3px",
              color: colors.textPrimary,
              padding: "5px 8px",
              fontSize: "11px",
              fontFamily: fonts.mono,
            }}
          >
            {[10, 25, 50, 100, 200].map((n) => (
              <option key={n} value={n}>
                {n} runs
              </option>
            ))}
          </select>
          <button
            onClick={runBatch}
            disabled={batchRunning}
            style={{
              padding: "5px 12px",
              borderRadius: "3px",
              border: `1px solid ${colors.border}`,
              background: batchRunning ? colors.bg : colors.surface,
              color: colors.textPrimary,
              cursor: batchRunning ? "wait" : "pointer",
              fontSize: "11px",
            }}
          >
            {batchRunning
              ? `Running ${batchProgress}/${batchSize}...`
              : "Run Batch"}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "200px 1fr",
          height: "calc(100vh - 53px)",
        }}
      >
        {/* Sidebar */}
        <aside
          style={{
            padding: "14px",
            borderRight: `1px solid ${colors.border}`,
            overflowY: "auto",
            background: colors.surface,
          }}
        >
          {/* Scenario */}
          <Section title="Scenario">
            <div
              style={{ display: "flex", flexDirection: "column", gap: "3px" }}
            >
              {Object.entries(PRESETS).map(([key, p]) => (
                <button
                  key={key}
                  onClick={() => applyPreset(key)}
                  disabled={running || batchRunning}
                  style={{
                    padding: "6px 8px",
                    borderRadius: "3px",
                    border: `1px solid ${
                      preset === key ? colors.accent : colors.border
                    }`,
                    background:
                      preset === key ? `${colors.accent}08` : colors.surface,
                    color:
                      preset === key ? colors.accent : colors.textSecondary,
                    cursor: running || batchRunning ? "not-allowed" : "pointer",
                    fontSize: "11px",
                    textAlign: "left",
                    opacity: running || batchRunning ? 0.5 : 1,
                  }}
                >
                  {p.name}
                </button>
              ))}
            </div>
          </Section>

          {/* Parameters */}
          <Section title="Parameters">
            <ParamRow
              label="Cost c/Bρ"
              value={costRatio}
              onChange={setCostRatio}
              min={0.1}
              max={0.95}
              step={0.05}
              disabled={running || batchRunning}
              format={(v) => v.toFixed(2)}
            />
            <ParamRow
              label="Area L"
              value={areaSize}
              onChange={setAreaSize}
              min={300}
              max={800}
              step={50}
              disabled={running || batchRunning}
              format={(v) => `${v}×${v}`}
            />
            <div style={{ marginTop: "6px" }}>
              <label
                style={{
                  fontSize: "10px",
                  color: colors.textMuted,
                  display: "block",
                  marginBottom: "4px",
                }}
              >
                Volunteers N
              </label>
              <div style={{ display: "flex", gap: "3px" }}>
                {[50, 100, 150, 200].map((n) => (
                  <button
                    key={n}
                    onClick={() => setNumVolunteers(n)}
                    disabled={running || batchRunning}
                    style={{
                      flex: 1,
                      padding: "4px 0",
                      borderRadius: "2px",
                      border: `1px solid ${
                        numVolunteers === n ? colors.accent : colors.border
                      }`,
                      background:
                        numVolunteers === n
                          ? `${colors.accent}08`
                          : colors.surface,
                      color:
                        numVolunteers === n
                          ? colors.accent
                          : colors.textSecondary,
                      cursor:
                        running || batchRunning ? "not-allowed" : "pointer",
                      fontSize: "10px",
                      fontFamily: fonts.mono,
                      opacity: running || batchRunning ? 0.5 : 1,
                    }}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>
          </Section>

          {/* Equilibrium */}
          <Section title="Equilibrium">
            <table
              style={{
                width: "100%",
                fontSize: "10px",
                borderCollapse: "collapse",
              }}
            >
              <tbody>
                <DataRow label="ρ" value={rho.toFixed(5)} />
                <DataRow label="k* (Nash)" value={kNash} color={colors.nash} />
                <DataRow label="k_opt" value={kOpt} color={colors.optimal} />
                <DataRow
                  label="Gap"
                  value={kOpt - kNash}
                  color={colors.warning}
                />
                <DataRow
                  label="P(k*)"
                  value={`${(computePdet(kNash, rho) * 100).toFixed(1)}%`}
                  color={colors.nash}
                />
                <DataRow
                  label="P(k_opt)"
                  value={`${(computePdet(kOpt, rho) * 100).toFixed(1)}%`}
                  color={colors.optimal}
                />
              </tbody>
            </table>
          </Section>

          {/* P_det Chart */}
          <Section title="P_det(k)">
            <div style={{ height: 100 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={pDetCurveData}
                  margin={{ top: 5, right: 5, bottom: 5, left: -15 }}
                >
                  <CartesianGrid strokeDasharray="2 2" stroke={colors.border} />
                  <XAxis
                    dataKey="k"
                    tick={{ fill: colors.textMuted, fontSize: 8 }}
                    axisLine={{ stroke: colors.border }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: colors.textMuted, fontSize: 8 }}
                    domain={[0, 100]}
                    tickFormatter={(v) => `${v}%`}
                    axisLine={{ stroke: colors.border }}
                    tickLine={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="pDet"
                    stroke={colors.textSecondary}
                    strokeWidth={1.5}
                    dot={false}
                  />
                  <ReferenceLine
                    x={kNash}
                    stroke={colors.nash}
                    strokeWidth={1}
                    strokeDasharray="3 3"
                  />
                  <ReferenceLine
                    x={kOpt}
                    stroke={colors.optimal}
                    strokeWidth={1}
                    strokeDasharray="3 3"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Section>

          {/* Controls */}
          <Section title="Controls">
            <div style={{ display: "flex", gap: "4px", marginBottom: "8px" }}>
              {!running && !bothFinished && (
                <ControlBtn onClick={() => setRunning(true)}>Start</ControlBtn>
              )}
              {running && (
                <ControlBtn onClick={() => setRunning(false)}>Pause</ControlBtn>
              )}
              <ControlBtn onClick={handleReset}>Reset</ControlBtn>
            </div>
            <div>
              <label
                style={{
                  fontSize: "10px",
                  color: colors.textMuted,
                  display: "block",
                  marginBottom: "3px",
                }}
              >
                Speed
              </label>
              <div style={{ display: "flex", gap: "3px" }}>
                {[1, 2, 4, 8].map((s) => (
                  <button
                    key={s}
                    onClick={() => setPlaybackSpeed(s)}
                    style={{
                      flex: 1,
                      padding: "3px 0",
                      borderRadius: "2px",
                      border: `1px solid ${
                        playbackSpeed === s ? colors.accent : colors.border
                      }`,
                      background:
                        playbackSpeed === s
                          ? `${colors.accent}08`
                          : colors.surface,
                      color:
                        playbackSpeed === s ? colors.accent : colors.textMuted,
                      cursor: "pointer",
                      fontSize: "9px",
                      fontFamily: fonts.mono,
                    }}
                  >
                    {s}×
                  </button>
                ))}
              </div>
            </div>
          </Section>
        </aside>

        {/* Main Area */}
        <main style={{ padding: "14px", overflowY: "auto" }}>
          {/* Canvases */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "14px",
              marginBottom: "14px",
            }}
          >
            <SimPanel
              title="Nash Equilibrium"
              subtitle={`k* = ${kNash}`}
              canvasRef={canvasNashRef}
              state={stateNash}
              coverage={coverageNash}
              color={colors.nash}
              bgColor={colors.nashBg}
            />
            <SimPanel
              title="Social Optimum"
              subtitle={`k_opt = ${kOpt}`}
              canvasRef={canvasOptRef}
              state={stateOpt}
              coverage={coverageOpt}
              color={colors.optimal}
              bgColor={colors.optimalBg}
            />
          </div>

          {/* Stats Row */}
          {stateNash && stateOpt && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(5, 1fr)",
                gap: "1px",
                background: colors.border,
                borderRadius: "3px",
                overflow: "hidden",
                marginBottom: "14px",
              }}
            >
              <StatCell
                label="Time"
                nash={stateNash.time}
                opt={stateOpt.time}
              />
              <StatCell
                label="AoI"
                nash={stateNash.aoi}
                opt={stateOpt.aoi}
                warn={FIXED_PARAMS.aoiThreshold}
              />
              <StatCell
                label="Detections"
                nash={stateNash.detections}
                opt={stateOpt.detections}
              />
              <StatCell
                label="Coverage"
                nash={`${(coverageNash * 100).toFixed(0)}%`}
                opt={`${(coverageOpt * 100).toFixed(0)}%`}
              />
              <StatCell
                label="Status"
                nash={stateNash.outcome}
                opt={stateOpt.outcome}
                isStatus
              />
            </div>
          )}

          {/* AoI Chart */}
          {aoiChartData.length > 0 && (
            <div
              style={{
                background: colors.surface,
                borderRadius: "3px",
                border: `1px solid ${colors.border}`,
                padding: "10px 12px",
                marginBottom: "14px",
              }}
            >
              <div
                style={{
                  fontSize: "10px",
                  color: colors.textMuted,
                  marginBottom: "6px",
                }}
              >
                Age of Information
              </div>
              <div style={{ height: 80 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={aoiChartData}
                    margin={{ top: 0, right: 0, bottom: 0, left: -15 }}
                  >
                    <CartesianGrid
                      strokeDasharray="2 2"
                      stroke={colors.border}
                    />
                    <XAxis
                      dataKey="t"
                      tick={{ fill: colors.textMuted, fontSize: 8 }}
                      axisLine={{ stroke: colors.border }}
                      tickLine={false}
                    />
                    <YAxis
                      tick={{ fill: colors.textMuted, fontSize: 8 }}
                      axisLine={{ stroke: colors.border }}
                      tickLine={false}
                    />
                    <ReferenceLine
                      y={FIXED_PARAMS.aoiThreshold}
                      stroke={colors.warning}
                      strokeDasharray="3 3"
                      strokeWidth={1}
                    />
                    <Area
                      type="monotone"
                      dataKey="nash"
                      stroke={colors.nash}
                      fill={colors.nashLight}
                      strokeWidth={1}
                    />
                    <Area
                      type="monotone"
                      dataKey="opt"
                      stroke={colors.optimal}
                      fill={colors.optimalLight}
                      strokeWidth={1}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Batch Results */}
          {batchResults && (
            <div
              style={{
                background: colors.surface,
                borderRadius: "3px",
                border: `1px solid ${colors.border}`,
                padding: "10px 12px",
              }}
            >
              <div
                style={{
                  fontSize: "10px",
                  color: colors.textMuted,
                  marginBottom: "8px",
                }}
              >
                Batch Results (n = {batchResults.runs})
              </div>
              <table
                style={{
                  width: "100%",
                  fontSize: "11px",
                  borderCollapse: "collapse",
                }}
              >
                <thead>
                  <tr>
                    <th
                      style={{
                        textAlign: "left",
                        fontWeight: 500,
                        color: colors.textMuted,
                        paddingBottom: "6px",
                      }}
                    ></th>
                    <th
                      style={{
                        textAlign: "right",
                        fontWeight: 500,
                        color: colors.nash,
                        paddingBottom: "6px",
                      }}
                    >
                      Nash
                    </th>
                    <th
                      style={{
                        textAlign: "right",
                        fontWeight: 500,
                        color: colors.optimal,
                        paddingBottom: "6px",
                      }}
                    >
                      Optimal
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td
                      style={{ color: colors.textSecondary, padding: "2px 0" }}
                    >
                      Success Rate
                    </td>
                    <td style={{ textAlign: "right", fontFamily: fonts.mono }}>
                      {batchResults.nash.successRate.toFixed(1)}%
                    </td>
                    <td style={{ textAlign: "right", fontFamily: fonts.mono }}>
                      {batchResults.opt.successRate.toFixed(1)}%
                    </td>
                  </tr>
                  <tr>
                    <td
                      style={{ color: colors.textSecondary, padding: "2px 0" }}
                    >
                      Avg. Time
                    </td>
                    <td style={{ textAlign: "right", fontFamily: fonts.mono }}>
                      {batchResults.nash.avgTime.toFixed(0)}
                    </td>
                    <td style={{ textAlign: "right", fontFamily: fonts.mono }}>
                      {batchResults.opt.avgTime.toFixed(0)}
                    </td>
                  </tr>
                  <tr>
                    <td
                      style={{ color: colors.textSecondary, padding: "2px 0" }}
                    >
                      Avg. Detections
                    </td>
                    <td style={{ textAlign: "right", fontFamily: fonts.mono }}>
                      {batchResults.nash.avgDetections.toFixed(1)}
                    </td>
                    <td style={{ textAlign: "right", fontFamily: fonts.mono }}>
                      {batchResults.opt.avgDetections.toFixed(1)}
                    </td>
                  </tr>
                </tbody>
              </table>
              <div style={{ marginTop: "10px" }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    marginBottom: "4px",
                  }}
                >
                  <span
                    style={{
                      width: "50px",
                      fontSize: "9px",
                      color: colors.nash,
                    }}
                  >
                    Nash
                  </span>
                  <div
                    style={{
                      flex: 1,
                      height: "6px",
                      background: colors.bg,
                      borderRadius: "1px",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: `${batchResults.nash.successRate}%`,
                        background: colors.nash,
                        borderRadius: "1px",
                      }}
                    />
                  </div>
                </div>
                <div
                  style={{ display: "flex", alignItems: "center", gap: "8px" }}
                >
                  <span
                    style={{
                      width: "50px",
                      fontSize: "9px",
                      color: colors.optimal,
                    }}
                  >
                    Optimal
                  </span>
                  <div
                    style={{
                      flex: 1,
                      height: "6px",
                      background: colors.bg,
                      borderRadius: "1px",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: `${batchResults.opt.successRate}%`,
                        background: colors.optimal,
                        borderRadius: "1px",
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

function Section({ title, children }) {
  return (
    <div style={{ marginBottom: "14px" }}>
      <h3
        style={{
          fontSize: "9px",
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          color: colors.textMuted,
          marginBottom: "6px",
        }}
      >
        {title}
      </h3>
      {children}
    </div>
  );
}

function ParamRow({
  label,
  value,
  onChange,
  min,
  max,
  step,
  disabled,
  format,
}) {
  return (
    <div style={{ marginBottom: "8px" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: "2px",
        }}
      >
        <label style={{ fontSize: "10px", color: colors.textSecondary }}>
          {label}
        </label>
        <span
          style={{
            fontSize: "10px",
            color: colors.textPrimary,
            fontFamily: fonts.mono,
          }}
        >
          {format(value)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        disabled={disabled}
        style={{
          width: "100%",
          accentColor: colors.accent,
          opacity: disabled ? 0.5 : 1,
          height: "3px",
        }}
      />
    </div>
  );
}

function DataRow({ label, value, color }) {
  return (
    <tr>
      <td style={{ padding: "2px 0", color: colors.textSecondary }}>{label}</td>
      <td
        style={{
          padding: "2px 0",
          textAlign: "right",
          fontFamily: fonts.mono,
          fontWeight: 500,
          color: color || colors.textPrimary,
        }}
      >
        {value}
      </td>
    </tr>
  );
}

function ControlBtn({ onClick, children }) {
  return (
    <button
      onClick={onClick}
      style={{
        flex: 1,
        padding: "5px 10px",
        borderRadius: "3px",
        border: `1px solid ${colors.border}`,
        background: colors.surface,
        color: colors.textPrimary,
        cursor: "pointer",
        fontSize: "10px",
      }}
    >
      {children}
    </button>
  );
}

function SimPanel({
  title,
  subtitle,
  canvasRef,
  state,
  coverage,
  color,
  bgColor,
}) {
  return (
    <div
      style={{
        background: colors.surface,
        borderRadius: "3px",
        border: `1px solid ${colors.border}`,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          padding: "8px 10px",
          borderBottom: `1px solid ${colors.border}`,
          background: bgColor,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <span style={{ fontSize: "11px", fontWeight: 500, color }}>
            {title}
          </span>
          <span
            style={{
              fontSize: "10px",
              color: colors.textMuted,
              marginLeft: "6px",
            }}
          >
            {subtitle}
          </span>
        </div>
        {state && (
          <span
            style={{
              fontSize: "9px",
              padding: "2px 6px",
              borderRadius: "2px",
              fontFamily: fonts.mono,
              background:
                state.outcome === "rescued"
                  ? colors.optimalLight
                  : state.outcome === "timeout"
                  ? colors.nashLight
                  : colors.bg,
              color:
                state.outcome === "rescued"
                  ? colors.optimal
                  : state.outcome === "timeout"
                  ? colors.nash
                  : colors.textMuted,
            }}
          >
            {state.outcome}
          </span>
        )}
      </div>
      <div style={{ padding: "8px" }}>
        <canvas
          ref={canvasRef}
          width={320}
          height={320}
          style={{ width: "100%", aspectRatio: "1", display: "block" }}
        />
      </div>
      {state && (
        <div
          style={{
            padding: "6px 10px",
            borderTop: `1px solid ${colors.border}`,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            fontSize: "9px",
            color: colors.textMuted,
          }}
        >
          <span>Coverage: {(coverage * 100).toFixed(1)}%</span>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span>Rescue</span>
            <div
              style={{
                width: "60px",
                height: "3px",
                background: colors.bg,
                borderRadius: "1px",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${state.rescueProgress * 100}%`,
                  background: color,
                  borderRadius: "1px",
                }}
              />
            </div>
            <span>{(state.rescueProgress * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}
    </div>
  );
}

function StatCell({ label, nash, opt, warn, isStatus }) {
  const formatVal = (v) => {
    if (isStatus)
      return v === "rescued" ? "OK" : v === "timeout" ? "FAIL" : "...";
    return v;
  };
  const nashWarn = warn && typeof nash === "number" && nash > warn;
  const optWarn = warn && typeof opt === "number" && opt > warn;

  return (
    <div
      style={{
        background: colors.surface,
        padding: "8px",
        textAlign: "center",
      }}
    >
      <div
        style={{
          fontSize: "9px",
          color: colors.textMuted,
          marginBottom: "2px",
        }}
      >
        {label}
      </div>
      <div style={{ fontSize: "11px", fontFamily: fonts.mono }}>
        <span style={{ color: nashWarn ? colors.nash : colors.nash }}>
          {formatVal(nash)}
        </span>
        <span style={{ color: colors.textMuted, margin: "0 4px" }}>/</span>
        <span style={{ color: optWarn ? colors.nash : colors.optimal }}>
          {formatVal(opt)}
        </span>
      </div>
    </div>
  );
}
