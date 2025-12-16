import React, { useState, useEffect, useCallback, useRef } from 'react';

const DEFAULT_CONFIG = {
  areaWidth: 650,
  areaHeight: 450,
  nVolunteers: 50,
  detectionRadius: 35,
  volunteerSpeed: 2.5,
  targetSpeed: 1,
  targetMobile: true,  // Now actually used
  benefit: 100,
  cost: 5,
  maxTime: 60,
  rescueThreshold: 5,
};

const distance = (x1, y1, x2, y2) => Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
const computeRho = (R, W, H) => (Math.PI * R * R) / (W * H);

const computeNashK = (config) => {
  const rho = computeRho(config.detectionRadius, config.areaWidth, config.areaHeight);
  const Brho = config.benefit * rho;
  if (config.cost >= Brho) return 0;
  const k = 1 + Math.log(config.cost / Brho) / Math.log(1 - rho);
  return Math.max(0, Math.min(config.nVolunteers, Math.floor(k)));
};

const computeStackelbergK = (config) => {
  const rho = computeRho(config.detectionRadius, config.areaWidth, config.areaHeight);
  const NBrho = config.nVolunteers * config.benefit * rho;
  if (config.cost >= NBrho) return 0;
  const k = 1 + Math.log(config.cost / NBrho) / Math.log(1 - rho);
  return Math.max(0, Math.min(config.nVolunteers, Math.floor(k)));
};

// Pure function: compute new volunteer positions
const moveVolunteers = (volunteers, config) => {
  return volunteers.map(v => {
    const dx = v.waypoint.x - v.x;
    const dy = v.waypoint.y - v.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    
    if (dist < 10) {
      return { 
        ...v, 
        waypoint: { 
          x: Math.random() * config.areaWidth, 
          y: Math.random() * config.areaHeight 
        }
      };
    }
    
    const speed = config.volunteerSpeed;
    return {
      ...v,
      x: v.x + (dx / dist) * Math.min(speed, dist),
      y: v.y + (dy / dist) * Math.min(speed, dist),
    };
  });
};

// Pure function: compute new target position
const moveTarget = (target, config) => {
  // If target is not mobile, return unchanged
  if (!config.targetMobile) {
    return target;
  }
  
  let { x, y, vx, vy } = target;
  
  // Random direction change
  if (Math.random() < 0.05) {
    const angle = Math.random() * Math.PI * 2;
    vx = Math.cos(angle) * config.targetSpeed;
    vy = Math.sin(angle) * config.targetSpeed;
  }
  
  x += vx;
  y += vy;
  
  // Bounce off walls
  if (x < 0 || x > config.areaWidth) vx *= -1;
  if (y < 0 || y > config.areaHeight) vy *= -1;
  
  return {
    x: Math.max(0, Math.min(config.areaWidth, x)),
    y: Math.max(0, Math.min(config.areaHeight, y)),
    vx,
    vy
  };
};

// Pure function: check detection with given positions
const checkDetection = (volunteers, target, detectionRadius) => {
  for (const v of volunteers) {
    if (v.active && distance(v.x, v.y, target.x, target.y) <= detectionRadius) {
      return true;
    }
  }
  return false;
};

export default function EmergencySimulation() {
  const [config] = useState(DEFAULT_CONFIG);
  const [mode, setMode] = useState('stackelberg');
  const [running, setRunning] = useState(false);
  const [time, setTime] = useState(0);
  const [aoi, setAoi] = useState(0);
  const [aoiHistory, setAoiHistory] = useState([]);
  const [detections, setDetections] = useState(0);
  const [outcome, setOutcome] = useState('waiting');
  const [volunteers, setVolunteers] = useState([]);
  const [target, setTarget] = useState({ x: 325, y: 225, vx: 0, vy: 0 });
  const [kActive, setKActive] = useState(0);
  const [flash, setFlash] = useState(false);
  
  const canvasRef = useRef(null);
  const timeoutRef = useRef(null);

  const timeRef = useRef(0);
  const aoiRef = useRef(0);
  const detectionsRef = useRef(0);

  const initSimulation = useCallback(() => {
    const newVolunteers = [];
    for (let i = 0; i < config.nVolunteers; i++) {
      newVolunteers.push({
        id: i,
        x: Math.random() * config.areaWidth,
        y: Math.random() * config.areaHeight,
        waypoint: { x: Math.random() * config.areaWidth, y: Math.random() * config.areaHeight },
        active: false,
      });
    }
    
    const k = mode === 'nash' ? computeNashK(config) : computeStackelbergK(config);
    const activeIndices = new Set();
    while (activeIndices.size < k && activeIndices.size < config.nVolunteers) {
      activeIndices.add(Math.floor(Math.random() * config.nVolunteers));
    }
    activeIndices.forEach(i => { newVolunteers[i].active = true; });
    
    setVolunteers(newVolunteers);
    setKActive(k);
    
    // Target starts in interior
    const margin = config.detectionRadius;
    const initVx = config.targetMobile ? (Math.random() - 0.5) * config.targetSpeed * 2 : 0;
    const initVy = config.targetMobile ? (Math.random() - 0.5) * config.targetSpeed * 2 : 0;
    
    setTarget({
      x: margin + Math.random() * (config.areaWidth - 2 * margin),
      y: margin + Math.random() * (config.areaHeight - 2 * margin),
      vx: initVx,
      vy: initVy,
    });
    
    // Reset simulation counters (refs are the single source of truth per tick)
    timeRef.current = 0;
    aoiRef.current = 0;
    detectionsRef.current = 0;

    setTime(0);
    setAoi(0);
    setAoiHistory([]);
    setDetections(0);
    setOutcome('waiting');
    setRunning(false);
  }, [config, mode]);

  useEffect(() => { initSimulation(); }, [mode, initSimulation]);

  // Main simulation step - uses pure functions to avoid stale state
  const simulationStep = useCallback(() => {
    if (outcome !== 'searching') return;
    
    // Compute new positions FIRST (pure functions, no state dependency issues)
    const newVolunteers = moveVolunteers(volunteers, config);
    const newTarget = moveTarget(target, config);
    
    // Check detection with NEW positions (not stale state)
    const detected = checkDetection(newVolunteers, newTarget, config.detectionRadius);
    
    // Now update all state
    setVolunteers(newVolunteers);
    setTarget(newTarget);
    
    // Deterministic counters: compute next values from refs, then commit once per tick
    const nextDetections = detectionsRef.current + (detected ? 1 : 0);
    detectionsRef.current = nextDetections;
    setDetections(nextDetections);

    const nextAoi = detected ? 0 : aoiRef.current + 1;
    aoiRef.current = nextAoi;
    setAoi(nextAoi);
    setAoiHistory(h => [...h.slice(-59), nextAoi]);

    if (detected) {
      setFlash(true);
      setTimeout(() => setFlash(false), 150);
    }

    const nextTime = timeRef.current + 1;
    timeRef.current = nextTime;
    setTime(nextTime);

    if (nextTime >= config.maxTime) {
      const success = nextDetections > 0;
      setOutcome(success ? 'rescued' : 'failed');
      setRunning(false);
    }
  }, [outcome, volunteers, target, config]);

  // Animation loop
  useEffect(() => {
    if (running && outcome === 'searching') {
      timeoutRef.current = setTimeout(simulationStep, 80);
    }
    return () => clearTimeout(timeoutRef.current);
  }, [running, outcome, simulationStep]);

  // Canvas rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    ctx.fillStyle = flash ? '#d4edda' : '#f0f4f0';
    ctx.fillRect(0, 0, config.areaWidth, config.areaHeight);
    
    // Grid
    ctx.strokeStyle = '#ddd';
    for (let x = 0; x < config.areaWidth; x += 50) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, config.areaHeight); ctx.stroke();
    }
    for (let y = 0; y < config.areaHeight; y += 50) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(config.areaWidth, y); ctx.stroke();
    }
    
    // Detection zones for active volunteers
    volunteers.filter(v => v.active).forEach(v => {
      ctx.beginPath();
      ctx.arc(v.x, v.y, config.detectionRadius, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(59, 130, 246, 0.1)';
      ctx.fill();
    });
    
    // Volunteers
    volunteers.forEach(v => {
      ctx.beginPath();
      ctx.arc(v.x, v.y, v.active ? 6 : 4, 0, Math.PI * 2);
      ctx.fillStyle = v.active ? '#3b82f6' : '#9ca3af';
      ctx.fill();
      if (v.active) {
        ctx.strokeStyle = '#1e40af';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });
    
    // Target
    ctx.beginPath();
    ctx.arc(target.x, target.y, 12, 0, Math.PI * 2);
    ctx.fillStyle = '#ef4444';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = 'white';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('?', target.x, target.y);
    
  }, [volunteers, target, config, flash]);

  const nashK = computeNashK(config);
  const stackK = computeStackelbergK(config);

  return (
    <div className="bg-gray-900 text-white p-4 min-h-screen">
      <h1 className="text-2xl font-bold text-center mb-1">🚨 Emergency Crowd-Finding</h1>
      <p className="text-center text-gray-400 text-sm mb-4">Game Theory in Search & Rescue</p>
      
      <div className="flex flex-wrap gap-4 justify-center">
        <div className="bg-gray-800 rounded-lg p-4 w-56">
          <h2 className="font-bold mb-3">Controls</h2>
          
          <div className="mb-3">
            <div className="text-xs text-gray-400 mb-1">Mode</div>
            <div className="grid grid-cols-2 gap-1">
              <button onClick={() => setMode('nash')} disabled={running}
                className={`py-2 text-xs rounded font-medium ${mode === 'nash' ? 'bg-red-500' : 'bg-gray-700 hover:bg-gray-600'}`}>
                Nash
              </button>
              <button onClick={() => setMode('stackelberg')} disabled={running}
                className={`py-2 text-xs rounded font-medium ${mode === 'stackelberg' ? 'bg-green-500' : 'bg-gray-700 hover:bg-gray-600'}`}>
                Stackelberg
              </button>
            </div>
          </div>
          
          <div className="bg-gray-700 rounded p-2 mb-3 text-xs">
            <div className="flex justify-between">
              <span className={mode === 'nash' ? 'text-red-400 font-bold' : 'text-gray-500'}>Nash: k={nashK}</span>
              <span className={mode === 'stackelberg' ? 'text-green-400 font-bold' : 'text-gray-500'}>Opt: k={stackK}</span>
            </div>
          </div>
          
          <div className="flex gap-2 mb-4">
            {outcome === 'waiting' && (
              <button onClick={() => { setOutcome('searching'); setRunning(true); }}
                className="flex-1 bg-green-500 hover:bg-green-600 py-2 rounded font-bold text-sm">▶ Start</button>
            )}
            {outcome === 'searching' && (
              <button onClick={() => setRunning(!running)}
                className="flex-1 bg-yellow-500 hover:bg-yellow-600 py-2 rounded font-bold text-sm">
                {running ? '⏸' : '▶'}
              </button>
            )}
            <button onClick={initSimulation}
              className="flex-1 bg-gray-600 hover:bg-gray-500 py-2 rounded font-bold text-sm">🔄</button>
          </div>
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between border-b border-gray-700 pb-1">
              <span className="text-gray-400">Time</span>
              <span className="font-mono">{time}/{config.maxTime}</span>
            </div>
            <div className="flex justify-between border-b border-gray-700 pb-1">
              <span className="text-gray-400">Active</span>
              <span className="font-mono text-blue-400 font-bold">{kActive}/{config.nVolunteers}</span>
            </div>
            <div className="flex justify-between border-b border-gray-700 pb-1">
              <span className="text-gray-400">Detections</span>
              <span className="font-mono">{detections}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">AoI</span>
              <span className={`font-mono font-bold ${aoi > config.rescueThreshold ? 'text-red-400' : 'text-green-400'}`}>{aoi} min</span>
            </div>
          </div>
          
          <div className="mt-3 text-xs text-gray-500">
            Target: {config.targetMobile ? 'Mobile' : 'Stationary'}
          </div>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex justify-between items-center mb-2">
            <span className="font-bold text-sm">Search Area</span>
            <span className={`px-3 py-1 rounded-full text-xs font-bold ${
              outcome === 'waiting' ? 'bg-gray-600' :
              outcome === 'searching' ? 'bg-blue-500 animate-pulse' :
              outcome === 'rescued' ? 'bg-green-500' : 'bg-red-500'
            }`}>
              {outcome === 'waiting' ? '⏳ Ready' :
               outcome === 'searching' ? '🔍 Searching' :
               outcome === 'rescued' ? '✅ RESCUED' : '❌ FAILED'}
            </span>
          </div>
          
          <canvas ref={canvasRef} width={config.areaWidth} height={config.areaHeight}
            className="rounded border border-gray-600" />
          
          <div className="flex justify-center gap-4 mt-2 text-xs text-gray-400">
            <span>🔵 Active</span>
            <span>⚪ Inactive</span>
            <span>🔴 Target</span>
          </div>
          
          <div className="mt-3">
            <div className="text-xs text-gray-400 mb-1">AoI History (lower is better)</div>
            <div className="h-12 bg-gray-700 rounded relative overflow-hidden">
              <div className="absolute w-full border-t border-dashed border-red-500/50" style={{ bottom: '25%' }} />
              <div className="absolute bottom-0 left-0 right-0 flex items-end h-full">
                {aoiHistory.map((a, i) => (
                  <div key={i} className={`flex-1 ${a > config.rescueThreshold ? 'bg-red-500' : 'bg-green-500'}`}
                    style={{ height: `${Math.min(100, (a / 20) * 100)}%` }} />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="max-w-3xl mx-auto mt-4 bg-gray-800 rounded-lg p-4">
        <h2 className="font-bold mb-2">📖 What This Shows</h2>
        <p className="text-xs text-yellow-400 mb-2">
          ⚠️ Note: This is a <strong>qualitative demonstration</strong>, not a validation of the analytical formula. 
          Agent movement introduces temporal correlation not present in the i.i.d. theory.
        </p>
        <div className="grid md:grid-cols-2 gap-3 text-sm">
          <div className="bg-red-900/30 border border-red-700 rounded p-2">
            <strong className="text-red-400">Nash (Selfish):</strong>
            <p className="text-gray-300 text-xs mt-1">Volunteers only consider personal cost → k*={nashK} active</p>
          </div>
          <div className="bg-green-900/30 border border-green-700 rounded p-2">
            <strong className="text-green-400">Stackelberg (Incentivized):</strong>
            <p className="text-gray-300 text-xs mt-1">Platform offers incentives → k={stackK} active</p>
          </div>
        </div>
      </div>
    </div>
  );
}
